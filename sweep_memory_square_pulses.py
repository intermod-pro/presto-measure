# -*- coding: utf-8 -*-
"""Measure the frequency of the memory."""

from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt

from presto import pulsed
from presto.utils import rotate_opt

from _base import Base

IDX_LOW = 0
IDX_HIGH = -1


class Sweep_memory(Base):
    def __init__(
        self,
        readout_freq: float,
        control_freq: float,
        memory_freq_center: float,
        memory_freq_span: float,
        memory_freq_nr: int,
        readout_amp: float,
        control_amp: float,
        memory_amp: float,
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
        dt: float = 0,  # delay between start of memory and start of control pulse
    ) -> None:
        self.readout_freq = readout_freq
        self.control_freq = control_freq
        self.memory_freq_center = memory_freq_center
        self.memory_freq_span = memory_freq_span
        self.memory_freq_nr = memory_freq_nr
        self.readout_amp = readout_amp
        self.control_amp = control_amp
        self.memory_amp = memory_amp
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
        self.dt = dt

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run
        self.memory_freq_arr = None  # replaced by run

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
            dry_run=False,
            **self.DC_PARAMS,
        ) as pls:
            pls.hardware.set_adc_attenuation(self.sample_port, self.ADC_ATTENUATION)
            pls.hardware.set_dac_current(self.readout_port, self.DAC_CURRENT)
            pls.hardware.set_dac_current(self.control_port, self.DAC_CURRENT)
            pls.hardware.set_dac_current(self.memory_port, self.DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.readout_port, 0)
            pls.hardware.set_inv_sinc(self.control_port, 0)
            pls.hardware.set_inv_sinc(self.memory_port, 0)

            pls.hardware.configure_mixer(
                self.readout_freq,
                in_ports=self.sample_port,
                out_ports=self.readout_port,
            )
            pls.hardware.configure_mixer(self.control_freq, out_ports=self.control_port)
            pls.hardware.configure_mixer(self.memory_freq_center, out_ports=self.memory_port)

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(self.readout_port, group=0, scales=self.readout_amp)
            pls.setup_scale_lut(self.control_port, group=0, scales=self.control_amp)
            pls.setup_scale_lut(self.memory_port, group=0, scales=self.memory_amp)

            # Setup lookup tables for memory frequency
            self.memory_freq_arr = self.memory_freq_center + np.linspace(
                -self.memory_freq_span / 2, self.memory_freq_span / 2, self.memory_freq_nr
            )
            memory_if_arr = self.memory_freq_arr - self.memory_freq_center
            mask = memory_if_arr < 0
            memory_if_arr = np.abs(memory_if_arr)
            ph_i = np.zeros_like(memory_if_arr)
            ph_q = ph_i - np.pi / 2 + mask * np.pi  # +-np.pi/2 for low/high sideband
            pls.setup_freq_lut(
                self.memory_port, group=0, frequencies=memory_if_arr, phases=ph_i, phases_q=ph_q
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
            control_pulse = pls.setup_long_drive(
                self.control_port,
                group=0,
                duration=self.control_duration,
                amplitude=1.0 + 1j,
                envelope=False,
            )

            memory_pulse = pls.setup_long_drive(
                self.memory_port,
                group=0,
                duration=self.memory_duration,
                amplitude=1.0 + 1j,
                envelope=True,
            )

            # Setup sampling window
            pls.setup_store(self.sample_port, self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            pls.reset_phase(T, self.memory_port, group=0)
            pls.output_pulse(T, memory_pulse)  # displace memory
            T += self.dt
            pls.output_pulse(T, control_pulse)  # control conditional on memory being |0>
            T += control_pulse.get_duration()
            pls.output_pulse(T, readout_pulse)  # Readout
            pls.store(T + self.readout_sample_delay)
            T += readout_pulse.get_duration()
            pls.next_frequency(T, self.memory_port, group=0)
            T += self.wait_delay  # Wait for decay

            # **************************
            # *** Run the experiment ***
            # **************************
            pls.run(period=T, repeat_count=self.memory_freq_nr, num_averages=self.num_averages)
            self.t_arr, self.store_arr = pls.get_store_data()

        return self.save()

    def save(self, save_filename: Optional[str] = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "Sweep_memory":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = float(h5f.attrs["readout_freq"])  # type: ignore
            control_freq = float(h5f.attrs["control_freq"])  # type: ignore
            memory_freq_center = float(h5f.attrs["memory_freq_center"])  # type: ignore
            memory_freq_span = float(h5f.attrs["memory_freq_span"])  # type: ignore
            memory_freq_nr = int(h5f.attrs["memory_freq_nr"])  # type: ignore
            readout_amp = float(h5f.attrs["readout_amp"])  # type: ignore
            control_amp = float(h5f.attrs["control_amp"])  # type: ignore
            memory_amp = float(h5f.attrs["memory_amp"])  # type: ignore
            readout_duration = float(h5f.attrs["readout_duration"])  # type: ignore
            control_duration = float(h5f.attrs["control_duration"])  # type: ignore
            memory_duration = float(h5f.attrs["memory_duration"])  # type: ignore
            sample_duration = float(h5f.attrs["sample_duration"])  # type: ignore
            readout_port = int(h5f.attrs["readout_port"])  # type: ignore
            control_port = int(h5f.attrs["control_port"])  # type: ignore
            memory_port = int(h5f.attrs["memory_port"])  # type: ignore
            sample_port = int(h5f.attrs["sample_port"])  # type: ignore
            wait_delay = float(h5f.attrs["wait_delay"])  # type: ignore
            readout_sample_delay = float(h5f.attrs["readout_sample_delay"])  # type: ignore
            num_averages = int(h5f.attrs["num_averages"])  # type: ignore
            dt = float(h5f.attrs["dt"])  # type: ignore

            t_arr: npt.NDArray[np.float64] = h5f["t_arr"][()]  # type:ignore
            store_arr: npt.NDArray[np.complex128] = h5f["store_arr"][()]  # type:ignore
            memory_freq_arr: npt.NDArray[np.float64] = h5f["memory_freq_arr"][()]  # type:ignore

        self = cls(
            readout_freq=readout_freq,
            control_freq=control_freq,
            memory_freq_center=memory_freq_center,
            memory_freq_span=memory_freq_span,
            memory_freq_nr=memory_freq_nr,
            readout_amp=readout_amp,
            control_amp=control_amp,
            memory_amp=memory_amp,
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
            dt=dt,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr
        self.memory_freq_arr = memory_freq_arr
        return self

    def analyze(self, all_plots: bool = False):
        assert self.t_arr is not None
        assert self.store_arr is not None
        assert self.memory_freq_arr is not None

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
            fig1.show()  # type: ignore
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
        ax21.plot(1e-9 * self.memory_freq_arr, mult * np.abs(data))
        ax22.plot(1e-9 * self.memory_freq_arr, np.angle(data))
        ax23.plot(1e-9 * self.memory_freq_arr, mult * np.real(data))
        try:
            data_min = data.real.min()
            data_max = data.real.max()
            data_rng = data_max - data_min
            p0 = [self.memory_freq_center, self.memory_freq_span / 4, data_rng, data_min]
            popt, _ = curve_fit(_gaussian, self.memory_freq_arr, data.real, p0)
            ax23.plot(
                1e-9 * self.memory_freq_arr, mult * _gaussian(self.memory_freq_arr, *popt), "--"
            )
            print(f"f0 = {popt[0]} Hz")
            print(f"sigma = {abs(popt[1])} Hz")
        except Exception:
            print("fit failed")
        ax24.plot(1e-9 * self.memory_freq_arr, mult * np.imag(data))

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
