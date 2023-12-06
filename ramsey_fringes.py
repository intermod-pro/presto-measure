# -*- coding: utf-8 -*-
"""Measure a Ramsey fringes pattern by changing the frequency of two π/2 pulses and their delay."""
import ast
from typing import Any, List, Optional, Union

import h5py
import numpy as np
import numpy.typing as npt

from presto.hardware import AdcMode, DacMode
from presto import pulsed
from presto.utils import format_precision, rotate_opt, sin2, si_prefix_scale

from _base import Base

DAC_CURRENT = 32_000  # uA
IDX_LOW = 0
IDX_HIGH = -1


class RamseyFringes(Base):
    def __init__(
        self,
        readout_freq: float,
        control_freq_center: float,
        control_freq_span: float,
        control_freq_nr: int,
        readout_amp: float,
        control_amp: float,
        readout_duration: float,
        control_duration: float,
        sample_duration: float,
        delay_arr: Union[List[float], npt.NDArray[np.float64]],
        readout_port: int,
        control_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
        jpa_params: Optional[dict] = None,
        drag: float = 0.0,
    ) -> None:
        self.readout_freq = readout_freq
        self.control_freq_center = control_freq_center
        self.control_freq_span = control_freq_span
        self.control_freq_nr = control_freq_nr
        self.readout_amp = readout_amp
        self.control_amp = control_amp
        self.readout_duration = readout_duration
        self.control_duration = control_duration
        self.sample_duration = sample_duration
        self.delay_arr = np.atleast_1d(delay_arr).astype(np.float64)
        self.readout_port = readout_port
        self.control_port = control_port
        self.sample_port = sample_port
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages
        self.jpa_params = jpa_params
        self.drag = drag

        self.control_freq_arr = None  # replaced by run
        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run

    def run(
        self,
        presto_address: str,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
    ) -> str:
        # Instantiate interface class
        with pulsed.Pulsed(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            adc_mode=AdcMode.Mixed,
            dac_mode=DacMode.Mixed,
        ) as pls:
            # figure out frequencies
            assert self.control_freq_center > (self.control_freq_span / 2)
            assert self.control_freq_span < pls.get_fs("dac") / 2  # fits in HSB
            control_if_center = pls.get_fs("dac") / 4  # middle of HSB
            control_if_start = control_if_center - self.control_freq_span / 2
            control_if_stop = control_if_center + self.control_freq_span / 2
            control_if_arr = np.linspace(control_if_start, control_if_stop, self.control_freq_nr)
            control_nco = self.control_freq_center - control_if_center
            self.control_freq_arr = control_nco + control_if_arr

            pls.hardware.set_adc_attenuation(self.sample_port, 0.0)
            pls.hardware.set_dac_current(self.readout_port, DAC_CURRENT)
            pls.hardware.set_dac_current(self.control_port, DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.readout_port, 0)
            pls.hardware.set_inv_sinc(self.control_port, 0)

            pls.hardware.configure_mixer(
                self.readout_freq,
                in_ports=self.sample_port,
                out_ports=self.readout_port,
                sync=False,
            )  # sync in next call
            pls.hardware.configure_mixer(
                freq=control_nco, out_ports=self.control_port, sync=True
            )  # sync here

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************
            # Setup lookup tables for frequencies we want to sweep
            pls.setup_freq_lut(
                self.control_port,
                group=0,
                frequencies=control_if_arr,
                phases=np.full_like(control_if_arr, 0.0),
                phases_q=np.full_like(control_if_arr, -np.pi / 2),
            )  # HSB

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(self.readout_port, group=0, scales=self.readout_amp)
            pls.setup_scale_lut(self.control_port, group=0, scales=1.0)

            # Setup readout and control pulses
            # use setup_long_drive to create a pulse with square envelope
            # setup_long_drive supports smooth rise and fall transitions for the pulse,
            # but we keep it simple here
            readout_pulse = pls.setup_long_drive(
                self.readout_port,
                group=0,
                duration=self.readout_duration,
                amplitude=1.0 + 1j,
                envelope=False,
            )

            # number of samples in the control template
            control_ns = int(round(self.control_duration * pls.get_fs("dac")))
            control_envelope = self.control_amp * sin2(control_ns)
            # we loose 3 dB by using a nonzero IF so multiply the envelope by sqrt(2)
            control_envelope *= np.sqrt(2)
            control_pulse = pls.setup_template(
                self.control_port,
                group=0,
                template=control_envelope + 1j * control_envelope,
                envelope=True,
            )

            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            for delay in self.delay_arr:
                pls.reset_phase(T, self.control_port)
                pls.output_pulse(T, control_pulse)  # first pi/2 pulse
                T += self.control_duration
                T += delay
                pls.output_pulse(T, control_pulse)  # second pi/2 pulse
                T += self.control_duration
                pls.output_pulse(T, readout_pulse)  # Readout
                pls.store(T + self.readout_sample_delay)
                T += self.readout_duration
                T += self.wait_delay  # wait for decay
            pls.next_frequency(T, self.control_port)
            T += self.wait_delay

            # **************************
            # *** Run the experiment ***
            # **************************
            pls.run(period=T, repeat_count=self.control_freq_nr, num_averages=self.num_averages)
            self.t_arr, self.store_arr = pls.get_store_data()

        return self.save()

    def save(self, save_filename: Optional[str] = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "RamseyFringes":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = float(h5f.attrs["readout_freq"])  # type: ignore
            control_freq_center = float(h5f.attrs["control_freq_center"])  # type: ignore
            control_freq_span = float(h5f.attrs["control_freq_span"])  # type: ignore
            control_freq_nr = int(h5f.attrs["control_freq_nr"])  # type: ignore
            readout_amp = float(h5f.attrs["readout_amp"])  # type: ignore
            control_amp = float(h5f.attrs["control_amp"])  # type: ignore
            readout_duration = float(h5f.attrs["readout_duration"])  # type: ignore
            control_duration = float(h5f.attrs["control_duration"])  # type: ignore
            sample_duration = float(h5f.attrs["sample_duration"])  # type: ignore
            delay_arr: npt.NDArray[np.float64] = h5f["delay_arr"][()]  # type: ignore
            readout_port = int(h5f.attrs["readout_port"])  # type: ignore
            control_port = int(h5f.attrs["control_port"])  # type: ignore
            sample_port = int(h5f.attrs["sample_port"])  # type: ignore
            wait_delay = float(h5f.attrs["wait_delay"])  # type: ignore
            readout_sample_delay = float(h5f.attrs["readout_sample_delay"])  # type: ignore
            num_averages = int(h5f.attrs["num_averages"])  # type: ignore

            jpa_params: dict = ast.literal_eval(h5f.attrs["jpa_params"])  # type: ignore

            control_freq_arr: npt.NDArray[np.float64] = h5f["control_freq_arr"][()]  # type: ignore
            t_arr: npt.NDArray[np.float64] = h5f["t_arr"][()]  # type: ignore
            store_arr: npt.NDArray[np.complex128] = h5f["store_arr"][()]  # type: ignore

            try:
                drag = float(h5f.attrs["drag"])  # type: ignore
            except KeyError:
                drag = 0.0

        self = cls(
            readout_freq=readout_freq,
            control_freq_center=control_freq_center,
            control_freq_span=control_freq_span,
            control_freq_nr=control_freq_nr,
            readout_amp=readout_amp,
            control_amp=control_amp,
            readout_duration=readout_duration,
            control_duration=control_duration,
            sample_duration=sample_duration,
            delay_arr=delay_arr,
            readout_port=readout_port,
            control_port=control_port,
            sample_port=sample_port,
            wait_delay=wait_delay,
            readout_sample_delay=readout_sample_delay,
            num_averages=num_averages,
            jpa_params=jpa_params,
            drag=drag,
        )
        self.control_freq_arr = control_freq_arr
        self.t_arr = t_arr
        self.store_arr = store_arr

        return self

    def analyze(self, all_plots: bool = False):
        assert self.t_arr is not None
        assert self.store_arr is not None
        assert self.control_freq_arr is not None
        assert len(self.control_freq_arr) == self.control_freq_nr

        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit

        ret_fig = []

        t_low = self.t_arr[IDX_LOW]
        t_high = self.t_arr[IDX_HIGH]

        if all_plots:
            # Plot raw store data for first iteration as a check
            fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
            ax11, ax12 = ax1
            ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
            ax12.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
            ax11.plot(1e9 * self.t_arr, np.abs(self.store_arr[0, 0, :]))
            ax12.plot(1e9 * self.t_arr, np.angle(self.store_arr[0, 0, :]))
            ax12.set_xlabel("Time [ns]")
            fig1.show()
            ret_fig.append(fig1)

        # Analyze
        resp_arr = np.mean(self.store_arr[:, 0, IDX_LOW:IDX_HIGH], axis=-1)
        resp_arr.shape = (self.control_freq_nr, len(self.delay_arr))
        data = rotate_opt(resp_arr)
        plot_data = data.real

        data_max = np.abs(plot_data).max()
        unit = ""
        mult = 1.0
        if data_max < 1e-6:
            unit = "n"
            mult = 1e9
        elif data_max < 1e-3:
            unit = "μ"
            mult = 1e6
        elif data_max < 1e0:
            unit = "m"
            mult = 1e3
        plot_data *= mult

        # choose limits for colorbar
        cutoff = 0.0  # %
        lowlim = np.percentile(plot_data, cutoff)
        highlim = np.percentile(plot_data, 100.0 - cutoff)

        # extent
        x_min = 1e6 * self.delay_arr[0]
        x_max = 1e6 * self.delay_arr[-1]
        dx = 1e6 * (self.delay_arr[1] - self.delay_arr[0])
        y_min = 1e-9 * self.control_freq_arr[0]
        y_max = 1e-9 * self.control_freq_arr[-1]
        dy = 1e-9 * (self.control_freq_arr[1] - self.control_freq_arr[0])

        fig2 = plt.figure(figsize=(6.4, 9.6), constrained_layout=True)
        gs = fig2.add_gridspec(2, 1)
        gs1 = gs[0].subgridspec(1, 1)
        ax2: Any = gs1.subplots()
        ax3 = fig2.add_subplot(gs[1])

        im = ax2.imshow(
            plot_data,
            origin="lower",
            aspect="auto",
            interpolation="none",
            extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
            vmin=lowlim,
            vmax=highlim,
        )
        ax2.set_xlabel("Ramsey delay [μs]")
        ax2.set_ylabel("Control frequency [GHz]")
        cb = fig2.colorbar(im, ax=ax2)
        cb.set_label(f"Response I quadrature [{unit:s}FS]")
        # fig2.show()
        # ret_fig.append(fig2)

        fit_freq = np.zeros_like(self.control_freq_arr)
        err_freq = np.zeros_like(self.control_freq_arr)
        for jj in range(self.control_freq_nr):
            try:
                res, err = _fit_simple(self.delay_arr, plot_data[jj])
                fit_freq[jj] = np.abs(res[3])
                err_freq[jj] = err[3]
            except Exception:
                fit_freq[jj] = np.nan

        # n_fit = self.control_freq_nr // 4
        # pfit1 = np.polyfit(self.control_freq_arr[:n_fit], fit_freq[:n_fit], 1)
        # pfit2 = np.polyfit(self.control_freq_arr[-n_fit:], fit_freq[-n_fit:], 1)
        # x0 = np.roots(pfit1 - pfit2)[0]
        def detuning(x, x0):
            return np.abs(x - x0)

        good_idx = np.isfinite(fit_freq)
        guess = self.control_freq_arr[np.nanargmin(fit_freq)]
        popt, pcov = curve_fit(
            detuning,
            self.control_freq_arr[good_idx],
            fit_freq[good_idx],
            p0=[guess],
            sigma=err_freq[good_idx],
        )
        perr = np.sqrt(np.diag(pcov))
        x0_opt = popt[0]
        x0_err = perr[0]

        unit_x, mult_x = si_prefix_scale(self.control_freq_arr)
        unit_y, mult_y = si_prefix_scale(fit_freq)

        gray = "0.25" if plt.rcParams["axes.facecolor"] == "black" else "0.75"

        # fig3, ax3 = plt.subplots(tight_layout=True)
        ax3.plot(mult_x * self.control_freq_arr, mult_y * fit_freq, ".", c=gray, ms=12)
        ax3.set_ylabel(f"Fitted detuning [{unit_y}Hz]")
        ax3.set_xlabel(f"Control frequency [{unit_x}Hz]")
        fig2.show()
        _lims = ax3.axis()

        ax3.plot(
            mult_x * self.control_freq_arr,
            mult_y * detuning(self.control_freq_arr, *popt),
            "--",
            c="C1",
        )
        ax3.axhline(0.0, ls="--", c=gray)
        ax3.axvline(mult_x * x0_opt, ls="--", c=gray)
        ax3.axis(_lims)
        fig2.canvas.draw()
        print(f"Fitted qubit frequency: {format_precision(x0_opt, x0_err)} Hz")
        ret_fig.append(fig2)

        return ret_fig


def _func(t, offset, amplitude, T2, frequency, phase):
    return offset + amplitude * np.exp(-t / T2) * np.cos(2.0 * np.pi * frequency * t + phase)


def _fit_simple(x, y):
    from scipy.optimize import curve_fit

    pkpk = np.max(y) - np.min(y)
    offset = np.min(y) + pkpk / 2
    amplitude = 0.5 * pkpk
    T2 = 0.5 * (np.max(x) - np.min(x))
    freqs = np.fft.rfftfreq(len(x), x[1] - x[0])
    fft = np.fft.rfft(y)
    fft[0] = 0
    idx_max = np.argmax(np.abs(fft))
    frequency = freqs[idx_max]
    phase = np.angle(fft[idx_max])
    p0 = (
        offset,
        amplitude,
        T2,
        frequency,
        phase,
    )
    popt, pcov = curve_fit(
        _func,
        x,
        y,
        p0=p0,
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr
