# -*- coding: utf-8 -*-
"""Measure the energy-relaxation time T1."""
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


class Sweep_cavity(Base):
    def __init__(
        self,
        readout_freq: float,
        control_freq: float,
        cavity_freq_center: float,
        cavity_freq_span: float,
        cavity_freq_nr: int,
        readout_amp: float,
        control_amp: float,
        cavity_amp: float,
        readout_duration: float,
        control_duration: float,
        cavity_duration: float,
        sample_duration: float,
        readout_port: int,
        control_port: int,
        cavity_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
    ) -> None:
        self.readout_freq = readout_freq
        self.control_freq = control_freq
        self.cavity_freq_center = cavity_freq_center
        self.cavity_freq_span = cavity_freq_span
        self.cavity_freq_nr = cavity_freq_nr
        self.readout_amp = readout_amp
        self.control_amp = control_amp
        self.cavity_amp = cavity_amp
        self.readout_duration = readout_duration
        self.control_duration = control_duration
        self.cavity_duration = cavity_duration
        self.sample_duration = sample_duration
        self.readout_port = readout_port
        self.control_port = control_port
        self.cavity_port = cavity_port
        self.sample_port = sample_port
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run
        self.cavity_freq_arr = None  # replaced by run

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
            pls.hardware.set_inv_sinc(self.readout_port, 0)
            pls.hardware.set_inv_sinc(self.control_port, 0)

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
                self.cavity_freq_center, out_ports=self.cavity_port, sync=True
            )  # sync here

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(self.readout_port, group=0, scales=self.readout_amp)
            pls.setup_scale_lut(self.control_port, group=0, scales=self.control_amp)
            pls.setup_scale_lut(self.cavity_port, group=0, scales=self.cavity_amp)

            # Setup lookup tables for cavity frequency
            self.cavity_freq_arr = self.cavity_freq_center + np.linspace(
                -self.cavity_freq_span / 2, self.cavity_freq_span, self.cavity_freq_nr
            )
            cavity_if_arr = self.cavity_freq_arr - self.cavity_freq_center
            mask = np.ma.masked_less(cavity_if_arr, 0).mask
            cavity_if_arr = np.abs(cavity_if_arr)
            ph_i = np.zeros_like(cavity_if_arr)
            ph_q = ph_i - np.pi / 2 + mask * np.pi  # +-np.pi/2 for low/high sideband
            pls.setup_freq_lut(
                self.cavity_port, group=0, frequencies=cavity_if_arr, phases=ph_i, phases_q=ph_q
            )

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
                envelope=False,
            )

            cavity_ns = int(round(self.cavity_duration * pls.get_fs("dac")))
            cavity_envelope = sin2(cavity_ns)
            cavity_pulse = pls.setup_template(
                self.cavity_port,
                group=0,
                template=cavity_envelope + 1j * cavity_envelope,
                envelope=True,
            )

            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            pls.reset_phase(T, self.cavity_port, group=0)
            pls.output_pulse(T, cavity_pulse)  # displace cavity
            T += self.cavity_duration
            pls.output_pulse(T, control_pulse)  # pi pulse conditioned on cavity in |0>
            T += self.control_duration
            pls.output_pulse(T, readout_pulse)  # Readout
            pls.store(T + self.readout_sample_delay)
            T += self.readout_duration
            pls.next_frequency(T, self.cavity_port, group=0)
            T += self.wait_delay  # Wait for decay

            # **************************
            # *** Run the experiment ***
            # **************************
            pls.run(period=T, repeat_count=self.cavity_freq_nr, num_averages=self.num_averages)
            self.t_arr, self.store_arr = pls.get_store_data()

        return self.save()

    def save(self, save_filename: str = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "Sweep_cavity":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = h5f.attrs["readout_freq"]
            control_freq = h5f.attrs["control_freq"]
            cavity_freq_center = h5f.attrs["cavity_freq_center"]
            cavity_freq_span = h5f.attrs["cavity_freq_span"]
            cavity_freq_nr = h5f.attrs["cavity_freq_nr"]
            readout_amp = h5f.attrs["readout_amp"]
            control_amp = h5f.attrs["control_amp"]
            cavity_amp = h5f.attrs["cavity_amp"]
            readout_duration = h5f.attrs["readout_duration"]
            control_duration = h5f.attrs["control_duration"]
            cavity_duration = h5f.attrs["cavity_duration"]
            sample_duration = h5f.attrs["sample_duration"]
            readout_port = h5f.attrs["readout_port"]
            control_port = h5f.attrs["control_port"]
            cavity_port = h5f.attrs["cavity_port"]
            sample_port = h5f.attrs["sample_port"]
            wait_delay = h5f.attrs["wait_delay"]
            readout_sample_delay = h5f.attrs["readout_sample_delay"]
            num_averages = h5f.attrs["num_averages"]

            t_arr = h5f["t_arr"][()]
            store_arr = h5f["store_arr"][()]
            cavity_freq_arr = h5f["cavity_freq_arr"][()]

        self = cls(
            readout_freq=readout_freq,
            control_freq=control_freq,
            cavity_freq_center=cavity_freq_center,
            cavity_freq_span=cavity_freq_span,
            cavity_freq_nr=cavity_freq_nr,
            readout_amp=readout_amp,
            control_amp=control_amp,
            cavity_amp=cavity_amp,
            readout_duration=readout_duration,
            control_duration=control_duration,
            cavity_duration=cavity_duration,
            sample_duration=sample_duration,
            readout_port=readout_port,
            control_port=control_port,
            cavity_port=cavity_port,
            sample_port=sample_port,
            wait_delay=wait_delay,
            readout_sample_delay=readout_sample_delay,
            num_averages=num_averages,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr
        self.cavity_freq_arr = cavity_freq_arr
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

    def analyze(self, all_plots: bool = False):
        assert self.t_arr is not None
        assert self.store_arr is not None

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
        data = rotate_opt(resp_arr)

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

        fig2, ax2 = plt.subplots(4, 1, sharex=True, figsize=(6.4, 6.4), tight_layout=True)
        ax21, ax22, ax23, ax24 = ax2
        ax21.plot(1e-9 * self.cavity_freq_arr, mult * np.abs(data))
        ax22.plot(1e-9 * self.cavity_freq_arr, np.angle(data))
        ax23.plot(1e-9 * self.cavity_freq_arr, mult * np.real(data))
        try:
            data_min = data.real.min()
            data_max = data.real.max()
            data_rng = data_max - data_min
            p0 = [self.cavity_freq_center, self.cavity_freq_span / 4, data_rng, data_min]
            popt, _ = curve_fit(_gaussian, self.cavity_freq_arr, data.real, p0)
            ax23.plot(
                1e-9 * self.cavity_freq_arr, mult * _gaussian(self.cavity_freq_arr, *popt), "--"
            )
            print(f"f0 = {popt[0]} Hz")
            print(f"sigma = {abs(popt[1])} Hz")
        except Exception:
            print("fit failed")
        ax24.plot(1e-9 * self.cavity_freq_arr, mult * np.imag(data))

        ax21.set_ylabel(f"Amplitude [{unit:s}FS]")
        ax22.set_ylabel("Phase [rad]")
        ax23.set_ylabel(f"I [{unit:s}FS]")
        ax24.set_ylabel(f"Q [{unit:s}FS]")
        ax2[-1].set_xlabel("Control frequency [GHz]")
        fig2.show()
        ret_fig.append(fig2)

        return ret_fig


def _gaussian(x, x0, s, a, o):
    return a * np.exp(-0.5 * ((x - x0) / s) ** 2) + o
