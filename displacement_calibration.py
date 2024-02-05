# -*- coding: utf-8 -*-
"""Calibrate the amplitude of the displacement pulse."""
from typing import List, Optional

import h5py
import numpy as np

from presto.hardware import AdcMode, DacMode
from presto import pulsed
from presto.utils import format_precision, rotate_opt, sin2

from _base import Base, project

DAC_CURRENT = 32_000  # uA
CONVERTER_CONFIGURATION = {
    "adc_mode": AdcMode.Mixed,
    "dac_mode": DacMode.Mixed,
}
IDX_LOW = 0
IDX_HIGH = -1


class DisplacementCalibration(Base):
    def __init__(
        self,
        readout_freq: float,
        control_freq: float,
        control_df_arr: List[float],
        memory_freq: float,
        readout_amp: float,
        control_amp: float,
        memory_amp_arr: List[float],
        readout_duration: float,
        control_duration: float,
        memory_duration: float,
        sample_duration: float,
        readout_port: int,
        control_port: int,
        memory_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
    ) -> None:
        self.readout_freq = readout_freq
        self.control_freq = control_freq
        self.control_df_arr = control_df_arr
        self.memory_freq = memory_freq
        self.readout_amp = readout_amp
        self.control_amp = control_amp
        self.memory_amp_arr = np.atleast_1d(memory_amp_arr).astype(np.float64)
        self.readout_duration = readout_duration
        self.control_duration = control_duration
        self.memory_duration = memory_duration
        self.sample_duration = sample_duration
        self.readout_port = readout_port
        self.control_port = control_port
        self.memory_port = memory_port
        self.sample_port = sample_port
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run
        self.control_freq_arr = None  # replaced by run

    def run(
        self,
        presto_address: str,
        presto_port: int = None,
        ext_ref_clk: bool = False,
        save: bool = True,
    ) -> str:
        # Instantiate interface class
        with pulsed.Pulsed(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            **CONVERTER_CONFIGURATION,
        ) as pls:
            pls.hardware.set_adc_attenuation(self.sample_port, 0.0)
            pls.hardware.set_dac_current(self.readout_port, DAC_CURRENT)
            pls.hardware.set_dac_current(self.control_port, DAC_CURRENT)
            pls.hardware.set_dac_current(self.memory_port, DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.readout_port, 0)
            pls.hardware.set_inv_sinc(self.control_port, 0)
            pls.hardware.set_inv_sinc(self.memory_port, 0)

            pls.hardware.configure_mixer(
                self.readout_freq,
                in_ports=self.sample_port,
                out_ports=self.readout_port,
                sync=False,
            )  # sync in next call

            pls.hardware.configure_mixer(
                self.control_freq, out_ports=self.control_port, sync=False
            )
            pls.hardware.configure_mixer(
                self.memory_freq, out_ports=self.memory_port, sync=True
            )  # sync here

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(self.readout_port, group=0, scales=self.readout_amp)
            pls.setup_scale_lut(self.control_port, group=0, scales=self.control_amp * np.sqrt(2))
            pls.setup_scale_lut(self.memory_port, group=0, scales=self.memory_amp_arr)

            # Setup lookup tables for frequencies
            # intermediate frequency
            control_if_center = pls.get_fs("dac") / 4  # 250 MHz, middle of USB

            control_if_start = control_if_center + self.control_df_arr[0]
            control_if_stop = control_if_center + self.control_df_arr[-1]
            control_if_arr = np.linspace(
                control_if_start, control_if_stop, len(self.control_df_arr)
            )

            # up-conversion carrier
            control_nco = self.control_freq - control_if_center

            # final frequency array
            self.control_freq_arr = control_nco + control_if_arr

            pls.setup_freq_lut(
                self.control_port,
                group=0,
                frequencies=control_if_arr,
                phases=np.full_like(control_if_arr, 0.0),
                phases_q=np.full_like(control_if_arr, -np.pi / 2),
            )  # HSB

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
            control_envelope = sin2(control_ns)
            control_pulse = pls.setup_template(
                self.control_port,
                group=0,
                template=control_envelope + 1j * control_envelope,
                envelope=True,
            )

            memory_ns = int(round(self.memory_duration * pls.get_fs("dac")))
            memory_envelope = sin2(memory_ns)
            memory_pulse = pls.setup_template(
                self.memory_port,
                group=0,
                template=memory_envelope + 1j * memory_envelope,
                envelope=False,
            )

            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            for i in range(len(self.control_freq_arr)):
                pls.output_pulse(T, memory_pulse)  # displace memory
                T += self.memory_duration
                pls.select_frequency(T, i, self.control_port, group=0)
                pls.output_pulse(T, control_pulse)  # pi pulse
                T += self.control_duration
                pls.output_pulse(T, readout_pulse)  # readout
                pls.store(T + self.readout_sample_delay)
                T += self.readout_duration
                T += self.wait_delay  # Wait for decay
            pls.next_scale(T, self.memory_port)
            T += self.wait_delay

            # **************************
            # *** Run the experiment ***
            # **************************
            pls.run(
                period=T, repeat_count=len(self.memory_amp_arr), num_averages=self.num_averages
            )
            self.t_arr, self.store_arr = pls.get_store_data()

        return self.save()

    def save(self, save_filename: str = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "DisplacementCalibration":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = h5f.attrs["readout_freq"]
            control_freq = h5f.attrs["control_freq"]
            control_df_arr = h5f["control_df_arr"][()]
            memory_freq = h5f.attrs["memory_freq"]
            readout_amp = h5f.attrs["readout_amp"]
            control_amp = h5f.attrs["control_amp"]
            memory_amp_arr = h5f["memory_amp_arr"][()]
            readout_duration = h5f.attrs["readout_duration"]
            control_duration = h5f.attrs["control_duration"]
            memory_duration = h5f.attrs["memory_duration"]
            sample_duration = h5f.attrs["sample_duration"]
            readout_port = h5f.attrs["readout_port"]
            control_port = h5f.attrs["control_port"]
            memory_port = h5f.attrs["memory_port"]
            sample_port = h5f.attrs["sample_port"]
            wait_delay = h5f.attrs["wait_delay"]
            readout_sample_delay = h5f.attrs["readout_sample_delay"]
            num_averages = h5f.attrs["num_averages"]

            t_arr = h5f["t_arr"][()]
            store_arr = h5f["store_arr"][()]
            control_freq_arr = h5f["control_freq_arr"][()]

        self = cls(
            readout_freq=readout_freq,
            control_freq=control_freq,
            control_df_arr=control_df_arr,
            memory_freq=memory_freq,
            readout_amp=readout_amp,
            control_amp=control_amp,
            memory_amp_arr=memory_amp_arr,
            readout_duration=readout_duration,
            control_duration=control_duration,
            memory_duration=memory_duration,
            sample_duration=sample_duration,
            readout_port=readout_port,
            control_port=control_port,
            memory_port=memory_port,
            sample_port=sample_port,
            wait_delay=wait_delay,
            readout_sample_delay=readout_sample_delay,
            num_averages=num_averages,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr
        self.control_freq_arr = control_freq_arr

        return self

    def analyze_batch(self, reference_templates: Optional[tuple] = None):
        assert self.t_arr is not None
        assert self.store_arr is not None

        if reference_templates is None:
            resp_arr = np.mean(self.store_arr[:, 0, IDX_LOW:IDX_HIGH], axis=-1)
            data = np.real(rotate_opt(resp_arr))
        else:
            resp_arr = self.store_arr[:, 0, :]
            data = project(resp_arr, reference_templates)

        try:
            popt, perr = _fit_simple(self.delay_arr, data)
        except Exception as err:
            print(f"unable to fit T1: {err}")
            popt, perr = None, None

        return data, (popt, perr)

    def analyze(self, all_plots: bool = False, blit: bool = False, _do_fit: bool = True):
        assert self.t_arr is not None
        assert self.store_arr is not None

        import matplotlib.pyplot as plt

        try:
            import matplotlib.widgets as mwidgets
        except ImportError:
            _do_fit = False

        ret_fig = []
        t_low = self.t_arr[IDX_LOW]
        t_high = self.t_arr[IDX_HIGH]

        nr_amps = len(self.memory_amp_arr)
        self._AMP_IDX = nr_amps // 2

        if all_plots:
            # Plot raw store data for first iteration as a check
            fig0, ax0 = plt.subplots(2, 1, sharex=True, tight_layout=True)
            ax01, ax02 = ax0
            ax01.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
            ax02.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
            ax01.plot(1e9 * self.t_arr, np.abs(self.store_arr[0, 0, :]))
            ax02.plot(1e9 * self.t_arr, np.angle(self.store_arr[0, 0, :]))
            ax02.set_xlabel("Time [ns]")
            fig0.show()
            ret_fig.append(fig0)

        # Analyze
        resp_arr = np.mean(self.store_arr[:, 0, IDX_LOW:IDX_HIGH], axis=-1)
        resp_arr.shape = (len(self.memory_amp_arr), len(self.control_df_arr))
        resp_arr = rotate_opt(resp_arr) * np.exp(1j * np.pi)
        resp_arr = resp_arr.real

        # bigger plot just for I quadrature
        data_max = np.abs(resp_arr).max()
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

        resp_arr *= mult
        # Fit data
        if _do_fit:
            popt, perr, M, N = _fit_simple(
                self.control_freq_arr, resp_arr[self._AMP_IDX], self.control_freq
            )
        # choose limits for colorbar
        cutoff = 1.0  # %
        lowlim = np.percentile(resp_arr, cutoff)
        highlim = np.percentile(resp_arr, 100.0 - cutoff)

        # extent
        x_min = 1e-9 * self.control_freq_arr[0]
        x_max = 1e-9 * self.control_freq_arr[-1]
        dx = 1e-9 * (self.control_freq_arr[1] - self.control_freq_arr[0])
        y_min = self.memory_amp_arr[0]
        y_max = self.memory_amp_arr[-1]
        dy = self.memory_amp_arr[1] - self.memory_amp_arr[0]

        fig1 = plt.figure(tight_layout=True, figsize=(6.4, 9.6))
        if _do_fit:
            cc = 4
            aa = 3
        else:
            cc = 2
            aa = 2
        ax1 = fig1.add_subplot(2, 1, 1)
        im = ax1.imshow(
            resp_arr,
            origin="lower",
            aspect="auto",
            interpolation="none",
            extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
            vmin=lowlim,
            vmax=highlim,
        )
        line_sel = ax1.axhline(
            self.memory_amp_arr[self._AMP_IDX], ls="--", c="k", lw=3, animated=blit
        )
        # ax1.set_title(f"amp = {amp_arr[AMP_IDX]:.2e}")
        ax1.set_xlabel("Frequency [GHz]")
        ax1.set_ylabel("Memory drive amplitude [FS]")
        cb = fig1.colorbar(im)
        cb.set_label(f"I quadrature [{unit:s}FS]")

        ax2 = fig1.add_subplot(cc, 1, aa)

        (line_a,) = ax2.plot(
            1e-9 * self.control_freq_arr,
            resp_arr[self._AMP_IDX],
            ".-",
            label="measured",
            animated=blit,
        )

        ax2.set_ylim(
            [
                np.min(resp_arr) - 0.05 * (np.max(resp_arr) - np.min(resp_arr)),
                np.max(resp_arr) + 0.05 * (np.max(resp_arr) - np.min(resp_arr)),
            ]
        )

        if _do_fit:
            (line_fit_a,) = ax2.plot(
                1e-9 * self.control_freq_arr,
                _fit_n_gauss(self.control_freq_arr, M, N, *popt),
                ls="--",
                label="fit",
                animated=blit,
            )
            ax3 = fig1.add_subplot(4, 1, 4)
            # ax3.set_ylim([np.min(self.memory_amp_arr), np.max(self.memory_amp_arr)])
            fitted_alpha = []
            presto_amp = []
            for i in range(len(resp_arr)):
                try:
                    popt, perr, M, N = _fit_simple(
                        self.control_freq_arr, resp_arr[i], self.control_freq
                    )
                    presto_amp.append(self.memory_amp_arr[i])
                    fitted_alpha.append(np.sqrt(popt[4]))
                except:
                    pass
            ax3.plot(presto_amp, fitted_alpha, ".")
            try:
                ind = presto_amp.index(0.0)
            except ValueError:
                ind = -1
            if ind >= 0:
                p = np.polyfit(
                    presto_amp[:ind] + presto_amp[ind + 1 :],
                    fitted_alpha[:ind] + fitted_alpha[ind + 1 :],
                    1,
                )  # skip the fitting for 0 amplitude that usually doesn't fit well
            else:
                p = np.polyfit(presto_amp, fitted_alpha, 1)
            print(r"Fitted dispacement conversion factor: alpha = %.4f*x[FS]+%.4f" % (p[0], p[1]))
            ax3.plot(presto_amp, np.polyval(p, presto_amp), "--")
            ax3.set_xlabel("Memory drive amplitude [FS]")
            ax3.set_ylabel(r"Displacement amplitude $\alpha$")

        def onbuttonpress(event):
            if event.inaxes == ax1:
                self._AMP_IDX = np.argmin(np.abs(self.memory_amp_arr - event.ydata))
                update()

        def onkeypress(event):
            if event.inaxes == ax1:
                if event.key == "up":
                    self._AMP_IDX += 1
                    if self._AMP_IDX >= len(self.memory_amp_arr):
                        self._AMP_IDX = len(self.memory_amp_arr) - 1
                    update()
                elif event.key == "down":
                    self._AMP_IDX -= 1
                    if self._AMP_IDX < 0:
                        self._AMP_IDX = 0
                    update()

        def update():
            line_sel.set_ydata(
                [self.memory_amp_arr[self._AMP_IDX], self.memory_amp_arr[self._AMP_IDX]]
            )
            # ax1.set_title(f"amp = {amp_arr[AMP_IDX]:.2e}")
            line_a.set_ydata(resp_arr[self._AMP_IDX])
            if _do_fit:
                popt, perr, M, N = _fit_simple(
                    self.control_freq_arr, resp_arr[self._AMP_IDX], self.control_freq
                )
                line_fit_a.set_ydata(_fit_n_gauss(self.control_freq_arr, M, N, *popt))
            # ax2.set_title("")
            if blit:
                fig1.canvas.restore_region(self._bg)
                ax1.draw_artist(line_sel)
                ax2.draw_artist(line_a)
                fig1.canvas.blit(fig1.bbox)
                fig1.canvas.flush_events()
            else:
                fig1.canvas.draw()

        fig1.canvas.mpl_connect("button_press_event", onbuttonpress)
        fig1.canvas.mpl_connect("key_press_event", onkeypress)
        fig1.show()
        if blit:
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            self._bg = fig1.canvas.copy_from_bbox(fig1.bbox)
            ax1.draw_artist(line_sel)
            ax2.draw_artist(line_a)
            fig1.canvas.blit(fig1.bbox)
        ret_fig.append(fig1)
        return ret_fig


def _fit_gauss(f, mu, sigma, off, A):
    return off + np.array(A * np.exp(-1 / 2 * ((f - mu) / sigma) ** 2))


def _fit_n_gauss(f, M, N, f0, sigma, off, k, beta2, *df):
    res = np.zeros(len(f))
    for kk in range(M, N):
        res += _fit_gauss(
            f,
            f0 - df[kk - M],
            sigma,
            off,
            k * beta2**kk / np.math.factorial(kk) * np.exp(-beta2),
        )
    return res


def _fit_simple(f, x, control_freq):
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks

    def my_fit_n_gauss(f, *p):
        return _fit_n_gauss(f, M, N, *p)

    peaks, _ = find_peaks(x, distance=10, prominence=(np.max(x) - np.min(x)) / 10)
    N = len(peaks)
    if N < 2:
        chi_t = 1.5e6
        M = 0
    else:
        chi_t = f[peaks[-1]] - f[peaks[-2]]
        M = int((control_freq - f[peaks[-1]]) / chi_t)
    # chi_p_t = chi_t / 100
    back_t = np.min(x) / (N - M)
    if back_t < 0:
        back_t_min = 3 * back_t
        back_t_max = 0.01 * back_t
    else:
        back_t_min = 0.01 * back_t
        back_t_max = 3 * back_t
    if N == 1:
        b2_t = 0.0
    elif N < 3:
        b2_t = 0.5
    else:
        b2_t = (N - M) / 2 + M
    df = control_freq - f[peaks[::-1]]
    p0 = (control_freq, chi_t / 10, back_t, max(x) - min(x), b2_t, *df)
    bounds = (
        [control_freq - chi_t / 4, -np.Inf, -np.Inf, -np.Inf, 0, *df - chi_t / 4],
        [control_freq + chi_t / 4, np.Inf, np.Inf, np.Inf, np.Inf, *df + chi_t / 4],
    )
    popt, pcov = curve_fit(my_fit_n_gauss, f, x, p0=p0)  # , bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, M, N
