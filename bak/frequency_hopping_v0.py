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
    "adc_fsample": AdcFSample.G4,
    "dac_mode": [DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
    "dac_fsample": [DacFSample.G6, DacFSample.G6, DacFSample.G6, DacFSample.G6],
}
IDX_LOW = 0
IDX_HIGH = 2_000


class FreqHopp(Base):
    def __init__(
        self,
        hopp_freq_arr: List[float],
        hopp_amp_arr: List[float],
        hopp_center_freq: float,
        hopp_duration: float,
        sample_duration: float,
        const_amp_port: int,
        var_amp_port: int,
        sample_port: int,
        num_averages: int,
        num_rep: int,
    ) -> None:
        self.hopp_freq_arr = np.atleast_1d(hopp_freq_arr).astype(np.float64)
        self.hopp_amp_arr = np.atleast_1d(hopp_amp_arr).astype(np.float64)
        self.hopp_center_freq = hopp_center_freq
        self.hopp_duration = hopp_duration
        self.sample_duration = sample_duration
        self.const_amp_port = const_amp_port
        self.var_amp_port = var_amp_port
        self.sample_port = sample_port
        self.num_averages = num_averages
        self.num_rep = num_rep

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run

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
            pls.hardware.set_dac_current(self.const_amp_port, DAC_CURRENT)
            pls.hardware.set_dac_current(self.var_amp_port, DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.const_amp_port, 0)
            pls.hardware.set_inv_sinc(self.var_amp_port, 0)

            pls.hardware.configure_mixer(
                self.hopp_center_freq,
                in_ports=self.sample_port,
                out_ports=[self.const_amp_port, self.var_amp_port],
                sync=True,
            )  # sync in next call

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(self.const_amp_port, group=0, scales=1)
            pls.setup_scale_lut(self.var_amp_port, group=0, scales=1)

            # For the hopping pulse we encode both the amplitude and
            # the frequency information directly in the templates

            hopp_samp = int(round(self.hopp_duration * pls.get_fs("dac")))
            hopp_duration_total = self.hopp_duration * len(self.hopp_freq_arr)

            temp_envelope = []
            temp_envelope_Q = []
            temp_envelope_var = []
            temp_envelope_var_Q = []
            # encode frequency information
            t = np.linspace(0, self.hopp_duration, hopp_samp, endpoint=False)
            for i in range(len(self.hopp_freq_arr)):
                if self.hopp_freq_arr < 0:  # lower sideband
                    a = -1
                    phi = np.pi / 2
                else:  # upper sideband
                    a = 1
                    phi = -np.pi / 2
                temp_envelope.append(np.cos(a * 2 * np.pi * self.hopp_freq_arr[i] * t))
                temp_envelope_Q.append(np.cos(a * 2 * np.pi * self.hopp_freq_arr[i] * t + phi))
                temp_envelope_var.append(
                    self.hopp_amp_arr[i] * np.cos(a * 2 * np.pi * self.hopp_freq_arr[i] * t)
                )
                temp_envelope_var_Q.append(
                    self.hopp_amp_arr[i] * np.cos(a * 2 * np.pi * self.hopp_freq_arr[i] * t + phi)
                )

            const_amp_pulse = pls.setup_template(
                self.const_amp_port,
                group=0,
                template=temp_envelope + 1j * temp_envelope_Q,
                envelope=False,
            )
            var_amp_pulse = pls.setup_template(
                self.var_amp_port,
                group=0,
                template=temp_envelope_var + 1j * temp_envelope_var_Q,
                envelope=False,
            )

            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            pls.output_pulse(T, [const_amp_pulse, var_amp_pulse])  # output both pulses
            pls.store(T)
            T += hopp_duration_total

            # **************************
            # *** Run the experiment ***
            # **************************
            # repeat the whole sequence `nr_amps` times
            # then average `num_averages` times

            pls.run(period=T, repeat_count=self.num_rep, num_averages=self.num_averages)
            self.t_arr, self.store_arr = pls.get_store_data()

        return self.save()

    def save(self, save_filename: str = None) -> str:
        return super().save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "FreqHopp":
        with h5py.File(load_filename, "r") as h5f:
            hopp_freq_arr = h5f["hopp_freq_arr"][()]
            hopp_amp_arr = h5f["hopp_amp_arr"][()]
            hopp_center_freq = h5f.attrs["hopp_center_freq"]
            hopp_duration = h5f.attrs["hopp_duration"]
            sample_duration = h5f.attrs["sample_duration"]
            const_amp_port = h5f.attrs["const_amp_port"]
            var_amp_port = h5f.attrs["var_amp_port"]
            sample_port = h5f.attrs["sample_port"]
            num_averages = h5f.attrs["num_averages"]
            num_rep = h5f.attrs["num_rep"]

            t_arr = h5f["t_arr"][()]
            store_arr = h5f["store_arr"][()]

        self = cls(
            hopp_freq_arr=hopp_freq_arr,
            hopp_amp_arr=hopp_amp_arr,
            hopp_center_freq=hopp_center_freq,
            hopp_duration=hopp_duration,
            sample_duration=sample_duration,
            const_amp_port=const_amp_port,
            var_amp_port=var_amp_port,
            sample_port=sample_port,
            num_averages=num_averages,
            num_rep=num_rep,
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

        idx = np.arange(IDX_LOW, IDX_HIGH)
        t_low = self.t_arr[IDX_LOW]
        t_high = self.t_arr[IDX_HIGH]

        fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
        ax11, ax12 = ax1
        ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
        ax12.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
        ax11.plot(1e9 * self.t_arr, np.abs(self.store_arr[0, 0, :]))
        ax12.plot(1e9 * self.t_arr, np.angle(self.store_arr[0, 0, :]))
        ax12.set_xlabel("Time [ns]")
        fig1.show()
        ret_fig.append(fig1)

        return ret_fig
