# -*- coding: utf-8 -*-
"""
Two-tone spectroscopy with Pulsed mode: sweep of pump frequency, with fixed pump power and fixed probe.
"""
import ast
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt

from presto.hardware import AdcMode, DacMode
from presto import pulsed
from presto.utils import rotate_opt, sin2

from _base import Base

DAC_CURRENT = 32_000  # uA
IDX_LOW = 0
IDX_HIGH = -1


class TwoTonePulsed(Base):
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
        self.readout_port = readout_port
        self.control_port = control_port
        self.sample_port = sample_port
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages
        self.drag = drag

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run
        self.control_freq_arr = None  # replaced by run

        self.jpa_params = jpa_params

    def run(
        self,
        presto_address: str,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
    ) -> str:
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
            # Setup lookup tables for frequencies that we sweep
            pls.setup_freq_lut(
                self.control_port,
                group=0,
                frequencies=control_if_arr,
                phases=np.full_like(control_if_arr, 0.0),
                phases_q=np.full_like(control_if_arr, -np.pi / 2),
            )  # HSB

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(self.readout_port, group=0, scales=self.readout_amp)
            pls.setup_scale_lut(self.control_port, group=0, scales=self.control_amp)

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
            control_envelope = sin2(control_ns)
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
            pls.reset_phase(T, self.control_port)
            pls.output_pulse(T, control_pulse)  # Control pulse
            T += self.control_duration
            pls.output_pulse(T, readout_pulse)  # Readout pulse
            pls.store(T + self.readout_sample_delay)  # Sampling window
            T += self.readout_duration
            pls.next_frequency(T, self.control_port)  # Move to next control frequency
            T += self.wait_delay  # Wait for decay

            # **************************
            # *** Run the experiment ***
            # **************************
            # repeat the whole sequence `control_freq_nr` times
            # then average `num_averages` times
            pls.run(period=T, repeat_count=self.control_freq_nr, num_averages=self.num_averages)
            self.t_arr, self.store_arr = pls.get_store_data()

        return self.save()

    def save(self, save_filename: Optional[str] = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "TwoTonePulsed":
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
            readout_port = int(h5f.attrs["readout_port"])  # type: ignore
            control_port = int(h5f.attrs["control_port"])  # type: ignore
            sample_port = int(h5f.attrs["sample_port"])  # type: ignore
            wait_delay = float(h5f.attrs["wait_delay"])  # type: ignore
            readout_sample_delay = float(h5f.attrs["readout_sample_delay"])  # type: ignore
            num_averages = int(h5f.attrs["num_averages"])  # type: ignore

            jpa_params: dict = ast.literal_eval(h5f.attrs["jpa_params"])  # type: ignore

            t_arr: npt.NDArray[np.float64] = h5f["t_arr"][()]  # type: ignore
            store_arr: npt.NDArray[np.complex128] = h5f["store_arr"][()]  # type: ignore
            control_freq_arr: npt.NDArray[np.float64] = h5f["control_freq_arr"][()]  # type: ignore

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
            readout_port=readout_port,
            control_port=control_port,
            sample_port=sample_port,
            wait_delay=wait_delay,
            readout_sample_delay=readout_sample_delay,
            num_averages=num_averages,
            jpa_params=jpa_params,
            drag=drag,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr
        self.control_freq_arr = control_freq_arr

        return self

    def analyze(self, all_plots: bool = False):
        assert self.t_arr is not None
        assert self.store_arr is not None
        assert self.control_freq_arr is not None

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
        ax21.plot(1e-9 * self.control_freq_arr, mult * np.abs(data))
        ax22.plot(1e-9 * self.control_freq_arr, np.angle(data))
        ax23.plot(1e-9 * self.control_freq_arr, mult * np.real(data))
        try:
            data_min = data.real.min()
            data_max = data.real.max()
            data_rng = data_max - data_min
            p0 = [self.control_freq_center, self.control_freq_span / 4, data_rng, data_min]
            popt, _ = curve_fit(_gaussian, self.control_freq_arr, data.real, p0)
            ax23.plot(
                1e-9 * self.control_freq_arr, mult * _gaussian(self.control_freq_arr, *popt), "--"
            )
            print(f"f0 = {popt[0]} Hz")
            print(f"sigma = {abs(popt[1])} Hz")
        except Exception:
            print("fit failed")
        ax24.plot(1e-9 * self.control_freq_arr, mult * np.imag(data))

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
