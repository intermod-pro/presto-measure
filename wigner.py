# -*- coding: utf-8 -*-
"""Measure the Wigner function of a bosonic mode."""
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


class Wigner(Base):
    def __init__(
        self,
        readout_freq: float,
        control_freq: float,
        cavity_freq: float,
        readout_amp: float,
        control_amp: float,
        cavity_amp_arr_x: List[float],
        cavity_amp_arr_y: List[float],
        dt_wigner: float,
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
        self.cavity_freq = cavity_freq
        self.readout_amp = readout_amp
        self.control_amp = control_amp
        self.cavity_amp_arr_x = np.atleast_1d(cavity_amp_arr_x).astype(np.float64)
        self.cavity_amp_arr_y = np.atleast_1d(cavity_amp_arr_y).astype(np.float64)
        self.dt_wigner = dt_wigner
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
            pls.hardware.set_dac_current(self.cavity_port, DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.readout_port, 0)
            pls.hardware.set_inv_sinc(self.control_port, 0)
            pls.hardware.set_inv_sinc(self.cavity_port, 0)

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
                self.cavity_freq, out_ports=self.cavity_port, sync=True
            )  # sync here

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(self.readout_port, group=0, scales=self.readout_amp)
            pls.setup_scale_lut(self.control_port, group=0, scales=self.control_amp)
            pls.setup_scale_lut(
                self.cavity_port, group=0, scales=self.cavity_amp_arr_x * np.sqrt(2), axis=1
            )
            pls.setup_scale_lut(
                self.cavity_port, group=1, scales=self.cavity_amp_arr_y * np.sqrt(2), axis=0
            )
            # Setup lookup tables for frequencies
            pls.setup_freq_lut(
                self.cavity_port, group=0, frequencies=0, phases=0, phases_q=-np.pi / 2
            )  # x displacement
            pls.setup_freq_lut(
                self.cavity_port, group=1, frequencies=0, phases=np.pi / 2, phases_q=0
            )  # y displacement

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
            cavity_pulse_x = pls.setup_template(
                self.cavity_port,
                group=0,
                template=cavity_envelope + 1j * cavity_envelope,
                envelope=True,
            )
            cavity_pulse_y = pls.setup_template(
                self.cavity_port,
                group=1,
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
            pls.reset_phase(T, output_ports=self.cavity_port, group=[0, 1])
            pls.output_pulse(T, [cavity_pulse_x, cavity_pulse_y])  # displace cavity
            T += self.cavity_duration
            pls.output_pulse(T, control_pulse)  # pi/2 pulse
            T += self.control_duration + self.dt_wigner
            pls.output_pulse(T, control_pulse)  # pi/2 pulse
            T += self.control_duration
            pls.output_pulse(T, readout_pulse)  # readout
            pls.store(T + self.readout_sample_delay)
            T += self.readout_duration
            T += self.wait_delay  # Wait for decay
            pls.next_scale(T, self.cavity_port, group=[0, 1])
            T += self.wait_delay

            # **************************
            # *** Run the experiment ***
            # **************************
            pls.run(
                period=T,
                repeat_count=(len(self.cavity_amp_arr_y), len(self.cavity_amp_arr_x)),
                num_averages=self.num_averages,
            )
            self.t_arr, self.store_arr = pls.get_store_data()

        return self.save()

    def save(self, save_filename: str = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "Wigner":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = h5f.attrs["readout_freq"]
            control_freq = h5f.attrs["control_freq"]
            cavity_freq = h5f.attrs["cavity_freq"]
            readout_amp = h5f.attrs["readout_amp"]
            control_amp = h5f.attrs["control_amp"]
            cavity_amp_arr_x = h5f["cavity_amp_arr_x"][()]
            cavity_amp_arr_y = h5f["cavity_amp_arr_y"][()]
            dt_wigner = h5f.attrs["dt_wigner"]
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

        self = cls(
            readout_freq=readout_freq,
            control_freq=control_freq,
            cavity_freq=cavity_freq,
            readout_amp=readout_amp,
            control_amp=control_amp,
            cavity_amp_arr_x=cavity_amp_arr_x,
            cavity_amp_arr_y=cavity_amp_arr_y,
            dt_wigner=dt_wigner,
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
        resp_arr.shape = (len(self.cavity_amp_arr_y), len(self.cavity_amp_arr_x))
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

        # choose limits for colorbar
        cutoff = 1.0  # %
        lowlim = np.percentile(resp_arr, cutoff)
        highlim = np.percentile(resp_arr, 100.0 - cutoff)

        # extent
        x_min = self.cavity_amp_arr_x[0]
        x_max = self.cavity_amp_arr_x[-1]
        dx = self.cavity_amp_arr_x[1] - self.cavity_amp_arr_x[0]
        y_min = self.cavity_amp_arr_y[0]
        y_max = self.cavity_amp_arr_y[-1]
        dy = self.cavity_amp_arr_y[1] - self.cavity_amp_arr_y[0]

        fig1 = plt.figure(tight_layout=True)
        ax1 = fig1.add_subplot(1, 1, 1)
        im = ax1.imshow(
            resp_arr,
            origin="lower",
            aspect="auto",
            interpolation="none",
            extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
            vmin=lowlim,
            vmax=highlim,
        )
        # ax1.set_title(f"amp = {amp_arr[AMP_IDX]:.2e}")
        ax1.set_xlabel("Displacement I [FS]")
        ax1.set_ylabel("Displacement Q [FS]")
        ax1.set_aspect("equal")
        cb = fig1.colorbar(im)
        cb.set_label(f"I quadrature [{unit:s}FS]")
        return ret_fig
