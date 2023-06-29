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


class RabiAmp(Base):
    def __init__(
        self,
        readout_freq: float,
        control_freq: float,
        readout_amp: float,
        control_amp_arr: List[float],
        readout_duration: float,
        control_duration: float,
        sample_duration: float,
        readout_port: int,
        control_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
        num_pulses: int = 1,
        jpa_params: dict = None,
        drag: float = 0.0,
    ) -> None:
        self.readout_freq = readout_freq
        self.control_freq = control_freq
        self.readout_amp = readout_amp
        self.control_amp_arr = np.atleast_1d(control_amp_arr).astype(np.float64)
        self.readout_duration = readout_duration
        self.control_duration = control_duration
        self.sample_duration = sample_duration
        self.readout_port = readout_port
        self.control_port = control_port
        self.sample_port = sample_port
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages
        self.num_pulses = num_pulses
        self.drag = drag

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run

        self.jpa_params = jpa_params

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

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            for _ in range(self.num_pulses):
                pls.output_pulse(T, control_pulse)  # Control pulse
                T += self.control_duration
            pls.output_pulse(T, readout_pulse)  # Readout
            pls.store(T + self.readout_sample_delay)
            T += self.readout_duration
            pls.next_scale(T, self.control_port)  # Move to next Rabi amplitude
            T += self.wait_delay  # Wait for decay

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
    def load(cls, load_filename: str) -> "RabiAmp":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = h5f.attrs["readout_freq"]
            control_freq = h5f.attrs["control_freq"]
            readout_amp = h5f.attrs["readout_amp"]
            control_amp_arr = h5f["control_amp_arr"][()]
            readout_duration = h5f.attrs["readout_duration"]
            control_duration = h5f.attrs["control_duration"]
            sample_duration = h5f.attrs["sample_duration"]
            readout_port = h5f.attrs["readout_port"]
            control_port = h5f.attrs["control_port"]
            sample_port = h5f.attrs["sample_port"]
            wait_delay = h5f.attrs["wait_delay"]
            readout_sample_delay = h5f.attrs["readout_sample_delay"]
            num_averages = h5f.attrs["num_averages"]
            num_pulses = h5f.attrs["num_pulses"]

            jpa_params = ast.literal_eval(h5f.attrs["jpa_params"])

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
            control_duration=control_duration,
            sample_duration=sample_duration,
            readout_port=readout_port,
            control_port=control_port,
            sample_port=sample_port,
            wait_delay=wait_delay,
            readout_sample_delay=readout_sample_delay,
            num_averages=num_averages,
            num_pulses=num_pulses,
            jpa_params=jpa_params,
            drag=drag,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr

        return self

    def analyze(self, all_plots: bool = False):
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
        data = rotate_opt(resp_arr)

        # Fit data
        popt_x, perr_x = _fit_period(self.control_amp_arr, np.real(data))
        period = popt_x[3] * self.num_pulses
        period_err = perr_x[3] * self.num_pulses
        pi_amp = period / 2
        pi_2_amp = period / 4
        if self.num_pulses > 1:
            print(f"{self.num_pulses} pulses")

        print(f"Tau pulse amplitude: {format_precision(period, period_err)} FS")
        print(f"Pi pulse amplitude: {format_precision(pi_amp, period_err / 2)} FS")
        print(f"Pi/2 pulse amplitude: {format_precision(pi_2_amp, period_err / 4)} FS")

        print(f"control_amp_180 = {pi_amp:.5f}")
        print(f"control_amp_90 = {pi_2_amp:.5f}")

        if all_plots:
            fig2, ax2 = plt.subplots(4, 1, sharex=True, figsize=(6.4, 6.4), tight_layout=True)
            ax21, ax22, ax23, ax24 = ax2
            ax21.plot(self.control_amp_arr, np.abs(data))
            ax22.plot(self.control_amp_arr, np.angle(data))
            ax23.plot(self.control_amp_arr, np.real(data))
            ax23.plot(self.control_amp_arr, _func(self.control_amp_arr, *popt_x), "--")
            ax24.plot(self.control_amp_arr, np.imag(data))

            ax21.set_ylabel("Amplitude [FS]")
            ax22.set_ylabel("Phase [rad]")
            ax23.set_ylabel("I [FS]")
            ax24.set_ylabel("Q [FS]")
            ax2[-1].set_xlabel("Pulse amplitude [FS]")
            fig2.show()
            ret_fig.append(fig2)

        data_max = np.abs(data).max()
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

        fig3, ax3 = plt.subplots(tight_layout=True)
        ax3.plot(self.control_amp_arr, mult * np.real(data), ".")
        ax3.plot(self.control_amp_arr, mult * _func(self.control_amp_arr, *popt_x), "--")
        ax3.set_ylabel(f"I quadrature [{unit:s}FS]")
        ax3.set_xlabel("Pulse amplitude [FS]")
        if self.num_pulses > 1:
            ax3.set_title(f"{self.num_pulses} pulses")
        # fig3.show()
        ret_fig.append(fig3)

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
