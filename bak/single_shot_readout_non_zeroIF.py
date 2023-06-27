# -*- coding: utf-8 -*-
"""
Measure Rabi oscillation by changing the amplitude of the control pulse.

The control pulse has a sin^2 envelope, while the readout pulse is square.
"""
import ast
import math
from typing import List, Tuple

import h5py
import numpy as np

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.utils import format_precision, rotate_opt, sin2

from _base import Base

DAC_CURRENT = 32_000  # uA
CONVERTER_CONFIGURATION = {
    "adc_mode": AdcMode.Mixed,
    "adc_fsample": AdcFSample.G2,
    "dac_mode": [DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
    "dac_fsample": [DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
}
IDX_LOW = 1_500
IDX_HIGH = 2_000


class SingleShotReadout(Base):
    def __init__(
        self,
        readout_freq: float,
        control_freq: float,
        readout_amp: float,
        control_amp: float,
        readout_duration: float,
        control_duration: float,
        sample_duration: float,
        readout_port: int,
        control_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
        template_match_duration=None,
        template_match_phase=0.0,
        drag: float = 0.0,
    ) -> None:
        self.readout_freq = readout_freq
        self.control_freq = control_freq
        self.readout_amp = readout_amp
        self.control_amp = control_amp
        self.readout_duration = readout_duration
        self.control_duration = control_duration
        self.sample_duration = sample_duration
        self.readout_port = readout_port
        self.control_port = control_port
        self.sample_port = sample_port
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages
        if template_match_duration == None:
            self.template_match_duration = sample_duration
        else:
            self.template_match_duration = template_match_duration
        self.template_match_phase = template_match_phase
        self.drag = drag

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run
        self.match_arr = None  # replaced by run

    def run(
        self,
        presto_address: str,
        presto_port: int = None,
        ext_ref_clk: bool = False,
    ) -> str:
        # Instantiate interface class
        with pulsed.Pulsed(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            **CONVERTER_CONFIGURATION,
        ) as pls:
            assert pls.hardware is not None

            pls.hardware.set_adc_attenuation(self.sample_port, 0.0)
            pls.hardware.set_dac_current(self.readout_port, DAC_CURRENT)
            pls.hardware.set_dac_current(self.control_port, DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.readout_port, 0)
            pls.hardware.set_inv_sinc(self.control_port, 0)

            readout_freq_IF = 100e6

            pls.hardware.configure_mixer(
                self.readout_freq + readout_freq_IF,
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
            # Setup lookup tables for frequencies
            pls.setup_freq_lut(
                self.readout_port,
                group=0,
                frequencies=readout_freq_IF,
                phases=0,
                phases_q=np.pi / 2,
            )

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(self.readout_port, group=0, scales=self.readout_amp)
            pls.setup_scale_lut(self.control_port, group=0, scales=[0, self.control_amp])

            # Setup readout and control pulses
            # use setup_long_drive to create a pulse with square envelope
            # setup_long_drive supports smooth rise and fall transitions for the pulse,
            # but we keep it simple here
            readout_pulse = pls.setup_long_drive(
                self.readout_port,
                group=0,
                duration=self.readout_duration,
                amplitude=1.0 + 1j,
                envelope=True,
            )

            # For the control pulse we create a sine-squared envelope,
            # and use setup_template to use the user-defined envelope
            # number of samples in the control template
            control_ns = int(round(self.control_duration * pls.get_fs("dac")))
            control_envelope = sin2(control_ns, drag=self.drag)
            control_pulse = pls.setup_template(
                self.control_port,
                group=0,
                template=control_envelope + 1j * control_envelope,
                envelope=False,
            )

            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # Setup template matching
            t_arr = np.linspace(
                0,
                self.template_match_duration,
                int(round(self.template_match_duration * pls.get_fs("dac"))),
                False,
            )
            carrier_II = np.cos(
                2 * np.pi * self.readout_freq_IF * t_arr + self.template_match_phase
            )
            carrier_IQ = np.sin(
                2 * np.pi * self.readout_freq_IF * t_arr + self.template_match_phase
            )
            carrier_QI = -np.sin(
                2 * np.pi * self.readout_freq_IF * t_arr + self.template_match_phase
            )
            carrier_QQ = np.cos(
                2 * np.pi * self.readout_freq_IF * t_arr + self.template_match_phase
            )
            match_events_I = pls.setup_template_matching_pair(
                input_port=self.sample_port,
                template1=carrier_II,
                template2=1j * carrier_IQ,
            )
            match_events_Q = pls.setup_template_matching_pair(
                input_port=self.sample_port,
                template1=carrier_QI,
                template2=1j * carrier_QQ,
            )

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            for ii in range(2):
                pls.select_scale(T, ii, self.control_port)
                pls.output_pulse(T, [control_pulse])
                T += self.control_length

                pls.reset_phase(T, self.readout_port)
                pls.output_pulse(T, [readout_pulse])
                pls.store(T + self.readout_sample_delay)
                pls.match(T + self.readout_sample_delay, [match_events_I, match_events_Q])
                T += self.readout_length + self.wait_delay

            # **************************
            # *** Run the experiment ***
            # **************************

            pls.run(period=T, repeat_count=1, num_averages=self.num_averages)
            self.t_arr, self.store_arr = pls.get_store_data()
            self.match_arr = pls.get_template_matching_data([match_events_I, match_events_Q])

        return self.save()

    def save(self, save_filename: str = None) -> str:
        return super().save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "SingleShotReadout":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = h5f.attrs["readout_freq"]
            control_freq = h5f.attrs["control_freq"]
            readout_amp = h5f.attrs["readout_amp"]
            control_amp = h5f["control_amp_arr"]
            readout_duration = h5f.attrs["readout_duration"]
            control_duration = h5f["control_duration"]
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
            control_amp=control_amp,
            readout_duration=readout_duration,
            control_duration=control_duration,
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

        idx = np.arange(IDX_LOW, IDX_HIGH)
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
        resp_arr = np.mean(self.store_arr[:, 0, idx], axis=-1)
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
