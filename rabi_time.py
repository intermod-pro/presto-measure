# -*- coding: utf-8 -*-
"""
Measure Rabi oscillation by changing the amplitude of the control pulse.

The control pulse has a sin^2 envelope, while the readout pulse is square.
"""
import ast
import math
from typing import List, Tuple
import warnings

import h5py
import numpy as np

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.utils import format_precision, rotate_opt, sin2, recommended_dac_config

from _base import Base

DAC_CURRENT = 32_000  # uA
CONVERTER_CONFIGURATION = {
    "adc_mode": AdcMode.Mixed,
    "adc_fsample": AdcFSample.G2,
}
IDX_LOW = 0
IDX_HIGH = -1


class RabiTime(Base):
    def __init__(
        self,
        readout_freq: float,
        control_freq: float,
        readout_amp: float,
        control_amp_arr: List[float],
        readout_duration: float,
        control_duration_arr: List[float],
        sample_duration: float,
        readout_port: int,
        control_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
        drag: float = 0.0,
    ) -> None:
        self.readout_freq = readout_freq
        self.control_freq = control_freq
        self.readout_amp = readout_amp
        self.control_amp_arr = np.atleast_1d(control_amp_arr).astype(np.float64)
        self.readout_duration = readout_duration
        self.control_duration_arr = np.atleast_1d(control_duration_arr).astype(np.float64)
        self.sample_duration = sample_duration
        self.readout_port = readout_port
        self.control_port = control_port
        self.sample_port = sample_port
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages
        self.drag = drag

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run

    def run(
        self,
        presto_address: str,
        presto_port: int = None,
        ext_ref_clk: bool = False,
    ) -> str:
        with pulsed.Pulsed(address=presto_address, ext_ref_clk=ext_ref_clk) as pls:
            control_tile = pls.hardware._port_to_tile(self.control_port, "dac")
            readout_tile = pls.hardware._port_to_tile(self.readout_port, "dac")
        dac_mode_r, dac_fsample_r = recommended_dac_config(self.readout_freq)
        dac_mode_c, dac_fsample_c = recommended_dac_config(self.control_freq)
        if dac_mode_c == dac_mode_r and dac_fsample_c == dac_fsample_r:
            dac_mode = dac_mode_c
            dac_fsample = dac_fsample_c
        elif control_tile != readout_tile:
            dac_mode = [dac_mode_r, dac_mode_r, dac_mode_r, dac_mode_r]
            dac_fsample = [dac_fsample_r, dac_fsample_r, dac_fsample_r, dac_fsample_r]
            dac_mode[control_tile] = dac_mode_c
            dac_fsample[control_tile] = dac_fsample_c
        else:
            warnings.warn(
                "Warning: The qubit and readout frequency might not be able to be output on the same tile. Consider outputting qubit tone on a different tile or manually choose the dac_mode and dac_fsample. See presto.utils.recommended_dac_config for help."
            )
            dac_mode = dac_mode_c
            dac_fsample = dac_fsample_c
        # Instantiate interface class
        with pulsed.Pulsed(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            dac_fsample=dac_fsample,
            dac_mode=dac_mode,
            **CONVERTER_CONFIGURATION,
        ) as pls:
            assert pls.hardware is not None

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
                self.control_freq, out_ports=self.control_port, sync=True
            )  # sync here

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(self.readout_port, group=0, scales=self.readout_amp)
            pls.setup_scale_lut(self.control_port, group=0, scales=self.control_amp_arr)

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

            # use setup_long_drive to create a pulse with square envelope
            # setup_long_drive supports smooth rise and fall transitions for the pulse,
            # but we keep it simple here
            control_pulse = pls.setup_long_drive(
                self.control_port,
                group=0,
                duration=self.control_duration_arr[0],
                amplitude=1.0 + 1j,
                envelope=False,
            )

            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            for control_duration in self.control_duration_arr:
                control_pulse.set_total_duration(control_duration)  # Set control pulse length
                pls.output_pulse(T, control_pulse)
                T += control_duration
                pls.output_pulse(T, readout_pulse)  # Readout
                pls.store(T + self.readout_sample_delay)
                T += self.readout_duration
                T += self.wait_delay  # Wait for decay
            pls.next_scale(T, self.control_port, group=0)
            T += self.wait_delay

            # **************************
            # *** Run the experiment ***
            # **************************
            # repeat the whole sequence `nr_amps` times
            # then average `num_averages` times

            nr_amps = len(self.control_amp_arr)
            pls.run(period=T, repeat_count=nr_amps, num_averages=self.num_averages)
            self.t_arr, self.store_arr = pls.get_store_data()

        return self.save()

    def save(self, save_filename: str = None) -> str:
        return super().save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "RabiTime":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = h5f.attrs["readout_freq"]
            control_freq = h5f.attrs["control_freq"]
            readout_amp = h5f.attrs["readout_amp"]
            control_amp_arr = h5f["control_amp_arr"][()]
            readout_duration = h5f.attrs["readout_duration"]
            control_duration_arr = h5f["control_duration_arr"][()]
            sample_duration = h5f.attrs["sample_duration"]
            readout_port = h5f.attrs["readout_port"]
            control_port = h5f.attrs["control_port"]
            sample_port = h5f.attrs["sample_port"]
            wait_delay = h5f.attrs["wait_delay"]
            readout_sample_delay = h5f.attrs["readout_sample_delay"]
            num_averages = h5f.attrs["num_averages"]

            t_arr = h5f["t_arr"][()]
            store_arr = h5f["store_arr"][()]

            try:
                drag = h5f.attrs["drag"]
            except KeyError:
                drag = 0.0

        self = cls(
            readout_freq=readout_freq,
            control_freq=control_freq,
            readout_amp=readout_amp,
            control_amp_arr=control_amp_arr,
            readout_duration=readout_duration,
            control_duration_arr=control_duration_arr,
            sample_duration=sample_duration,
            readout_port=readout_port,
            control_port=control_port,
            sample_port=sample_port,
            wait_delay=wait_delay,
            readout_sample_delay=readout_sample_delay,
            num_averages=num_averages,
            drag=drag,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr

        return self

    def analyze(self, portrait: bool = True, all_plots: bool = False):
        if self.t_arr is None:
            raise RuntimeError
        if self.store_arr is None:
            raise RuntimeError

        import matplotlib.pyplot as plt

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

        # Analyze Rabi
        resp_arr = np.mean(self.store_arr[:, 0, IDX_LOW:IDX_HIGH], axis=-1)
        resp_arr.shape = (len(self.control_amp_arr), len(self.control_duration_arr))
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
        x_min = 1e6 * self.control_duration_arr[0]
        x_max = 1e6 * self.control_duration_arr[-1]
        dx = 1e6 * (self.control_duration_arr[1] - self.control_duration_arr[0])
        y_min = self.control_amp_arr[0]
        y_max = self.control_amp_arr[-1]
        dy = self.control_amp_arr[1] - self.control_amp_arr[0]

        if portrait:
            fig2 = plt.figure(tight_layout=True, figsize=(6.4, 9.6))
            ax1 = fig2.add_subplot(2, 1, 1)
            # fig1 = plt.figure(tight_layout=True)
            # ax1 = fig1.add_subplot(1, 1, 1)
        else:
            fig2 = plt.figure(tight_layout=True, figsize=(12.8, 4.8))
            ax1 = fig2.add_subplot(1, 2, 1)

        im = ax1.imshow(
            plot_data,
            origin="lower",
            aspect="auto",
            interpolation="none",
            extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
            vmin=lowlim,
            vmax=highlim,
        )
        ax1.set_xlabel("Control length [μs]")
        ax1.set_ylabel("Control amplitude [FS]")
        cb = fig2.colorbar(im)
        if portrait:
            cb.set_label(f"Response I quadrature [{unit:s}FS]")
        else:
            ax1.set_title(f"Response I quadrature [{unit:s}FS]")

        if portrait:
            ax2 = fig2.add_subplot(2, 1, 2)
        else:
            ax2 = fig2.add_subplot(1, 2, 2)
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()

        # Fit data
        fit_freq = np.zeros_like(self.control_amp_arr)
        for jj in range(len(self.control_amp_arr)):
            try:
                res, _err = _fit_period(self.control_duration_arr, plot_data[jj])
                fit_freq[jj] = 1 / np.abs(res[3])
            except Exception:
                fit_freq[jj] = np.nan
        # Fit Rabi rate
        pfit1 = np.polyfit(self.control_amp_arr, fit_freq, 1)

        ax2.plot(self.control_amp_arr, fit_freq, ".")
        ax2.set_ylabel("Fitted Rabi rate $\omega/2\pi$ [Hz]")
        ax2.set_xlabel("Control amplitude [FS]")
        ax2.plot(
            self.control_amp_arr,
            np.polyval(pfit1, self.control_amp_arr),
            "--",
        )
        _lims = ax2.axis()
        ax2.axis(_lims)
        fig2.canvas.draw()
        ret_fig.append(fig2)

        return ret_fig


def _func(t, offset, amplitude, T2, period, phase):
    frequency = 1 / period
    return offset + amplitude * np.exp(-t / T2) * np.cos(math.tau * frequency * t + phase)


def _fit_period(x: List[float], y: List[float]) -> Tuple[List[float], List[float]]:
    from scipy.optimize import curve_fit

    pkpk = np.max(y) - np.min(y)
    offset = np.min(y) + pkpk / 2
    amplitude = 0.5 * pkpk
    T2 = 0.5 * (np.max(x) - np.min(x))
    freqs = np.fft.rfftfreq(len(x), x[1] - x[0])
    fft = np.fft.rfft(y)
    frequency = freqs[1 + np.argmax(np.abs(fft[1:]))]
    period = 1 / frequency
    first = (y[0] - offset) / amplitude
    if first > 1.0:
        first = 1.0
    elif first < -1.0:
        first = -1.0
    phase = np.arccos(first)
    p0 = (
        offset,
        amplitude,
        T2,
        period,
        phase,
    )
    res = curve_fit(_func, x, y, p0=p0)
    popt = res[0]
    pcov = res[1]
    perr = np.sqrt(np.diag(pcov))
    offset, amplitude, T2, period, phase = popt
    return popt, perr
