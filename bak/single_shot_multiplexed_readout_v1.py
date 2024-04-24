# -*- coding: utf-8 -*-
"""
Measure Rabi oscillation by changing the amplitude of the control pulse.

The control pulse has a sin^2 envelope, while the readout pulse is square.
"""

import ast
import math
from typing import List, Tuple, Optional, Union

import h5py
import numpy as np
import numpy.typing as npt

from presto.hardware import AdcMode, DacMode, MultibandMode
from presto import pulsed
from presto.utils import format_precision, rotate_opt, sin2

from _base import Base

DAC_CURRENT = 40_500  # uA
CONVERTER_CONFIGURATION = {
    "adc_mode": AdcMode.Mixed,
    "dac_mode": DacMode.Mixed,
}


class SingleShotMultiplexedReadout(Base):
    def __init__(
        self,
        readout_freq: Union[List[float], npt.NDArray[np.float64]],
        control_freq: Union[List[float], npt.NDArray[np.float64]],
        readout_amp: Union[List[float], npt.NDArray[np.float64]],
        control_amp: Union[List[float], npt.NDArray[np.float64]],
        readout_duration: float,
        control_duration: float,
        sample_duration: float,
        readout_port: int,
        control_port: Union[List[float], npt.NDArray[np.float64]],
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
        template_match_delay: float,
        template_match_duration: Optional[float] = None,
        template_match_phase: float = 0.0,
        drag: float = 0.0,
    ) -> None:
        self.readout_freq = np.atleast_1d(readout_freq).astype(np.float64)
        self.control_freq = np.atleast_1d(control_freq).astype(np.float64)
        self.readout_amp = np.atleast_1d(readout_amp).astype(np.float64)
        self.control_amp = np.atleast_1d(control_amp).astype(np.float64)
        self.readout_duration = readout_duration
        self.control_duration = control_duration
        self.sample_duration = sample_duration
        self.readout_port = readout_port
        self.control_port = np.atleast_1d(control_port).astype(np.int64)
        self.sample_port = sample_port
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages
        self.template_match_delay = template_match_delay
        if template_match_duration == None:
            self.template_match_duration = sample_duration
        else:
            self.template_match_duration = template_match_duration
        self.template_match_phase = template_match_phase
        self.drag = drag

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run
        self.match_arr = None  # replaced by run
        self.readout_freq_IF = None  # replaced by run

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

            nr_readout_pulses = len(self.readout_freq)
            if max(self.readout_freq) - min(self.readout_freq) > 1e9:
                raise ValueError(
                    f"Max badwidth is 1 GHz. Readout frequencies are {(max(self.readout_freq)-min(self.readout_freq))*1e-9} GHz appart."
                )

            if nr_readout_pulses == 1:
                readout_freq_LO = self.readout_freq[0]
            elif nr_readout_pulses == 2:
                readout_freq_LO = np.mean(self.readout_freq)
            elif nr_readout_pulses == 3:
                readout_freq_LO = np.sort(self.readout_freq)[1]
            else:
                raise ValueError("Max number of readout pulses is 3 in this implementation")

            readout_freq_IF = self.readout_freq - readout_freq_LO
            self.readout_freq_IF = readout_freq_IF

            pls.hardware.configure_mixer(
                readout_freq_LO,
                in_ports=[self.sample_port],
                out_ports=self.readout_port,
            )

            pls.setup_scale_lut(self.readout_port, group=0, scales=1)
            pls.setup_scale_lut(self.readout_port, group=1, scales=1)

            readout_pulse_list = []
            control_pulse_list = []
            match_events_list = []

            control_ns = int(round(self.control_duration * pls.get_fs("dac")))
            control_envelope = sin2(control_ns, drag=self.drag)

            if nr_readout_pulses in [1, 3]:
                ind = int(np.where(self.readout_freq_IF == 0.0)[0][0])
                readout_pulse_list.append(
                    pls.setup_long_drive(
                        self.readout_port,
                        group=0,
                        duration=self.readout_duration,
                        amplitude=self.readout_amp[ind],  # type: ignore
                        envelope=False,
                    )
                )

                pls.hardware.configure_mixer(
                    self.control_freq[ind], out_ports=self.control_port[ind]
                )
                pls.setup_scale_lut(
                    self.control_port[ind], group=0, scales=[0, self.control_amp[ind]]
                )
                control_pulse_list.append(
                    pls.setup_template(
                        self.control_port[ind],
                        group=0,
                        template=control_envelope + 1j * control_envelope,
                        envelope=False,
                    )
                )

                shape = np.ones(int(round(self.template_match_duration * pls.get_fs("dac"))))
                match_events_list.extend(
                    pls.setup_template_matching_pair(
                        input_port=self.sample_port,
                        template1=shape * np.exp(self.template_match_phase),
                        template2=1j * shape * np.exp(self.template_match_phase),
                    )
                )

            if nr_readout_pulses in [2, 3]:
                #####################################################
                # IF<0

                ind = int(np.where(self.readout_freq_IF < 0.0)[0][0])
                pls.hardware.configure_mixer(
                    self.control_freq[ind], out_ports=self.control_port[ind]
                )
                pls.setup_scale_lut(
                    self.control_port[ind], group=0, scales=[0, self.control_amp[ind]]
                )
                control_pulse_list.append(
                    pls.setup_template(
                        self.control_port[ind],
                        group=0,
                        template=control_envelope + 1j * control_envelope,
                        envelope=False,
                    )
                )
                IF_freq = readout_freq_IF[ind]
                pls.setup_freq_lut(
                    self.readout_port,
                    group=0,
                    frequencies=np.abs(IF_freq),
                    phases=0,
                    phases_q=-np.pi / 2 if IF_freq > 0 else np.pi / 2,
                )

                readout_pulse_list.append(
                    pls.setup_long_drive(
                        self.readout_port,
                        group=0,
                        duration=self.readout_duration,
                        amplitude=self.readout_amp[ind] * (1.0 + 1j),
                        envelope=True,
                    )
                )

                t_arr = np.linspace(
                    0,
                    self.template_match_duration,
                    int(round(self.template_match_duration * pls.get_fs("dac"))),
                    False,
                )

                carrier_II = np.cos(2 * np.pi * IF_freq * t_arr + self.template_match_phase)
                carrier_IQ = np.sin(2 * np.pi * IF_freq * t_arr + self.template_match_phase)
                carrier_QI = -np.sin(2 * np.pi * IF_freq * t_arr + self.template_match_phase)
                carrier_QQ = np.cos(2 * np.pi * IF_freq * t_arr + self.template_match_phase)
                match_events_list.append(
                    pls.setup_template_matching_pair(
                        input_port=self.sample_port,
                        template1=carrier_II,
                        template2=1j * carrier_IQ,
                    )
                )
                match_events_list.append(
                    pls.setup_template_matching_pair(
                        input_port=self.sample_port,
                        template1=carrier_QI,
                        template2=1j * carrier_QQ,
                    )
                )

                #####################################################
                # IF>0

                ind = int(np.where(self.readout_freq_IF > 0.0)[0][0])
                pls.hardware.configure_mixer(
                    self.control_freq[ind], out_ports=self.control_port[ind]
                )
                pls.setup_scale_lut(
                    self.control_port[ind], group=0, scales=[0, self.control_amp[ind]]
                )
                control_pulse_list.append(
                    pls.setup_template(
                        self.control_port[ind],
                        group=0,
                        template=control_envelope + 1j * control_envelope,
                        envelope=False,
                    )
                )
                IF_freq = readout_freq_IF[ind]
                pls.setup_freq_lut(
                    self.readout_port,
                    group=1,
                    frequencies=np.abs(IF_freq),
                    phases=0,
                    phases_q=-np.pi / 2 if IF_freq > 0 else np.pi / 2,
                )

                readout_pulse_list.append(
                    pls.setup_long_drive(
                        self.readout_port,
                        group=1,
                        duration=self.readout_duration,
                        amplitude=self.readout_amp[ind] * (1.0 + 1j),
                        envelope=True,
                    )
                )

                t_arr = np.linspace(
                    0,
                    self.template_match_duration,
                    int(round(self.template_match_duration * pls.get_fs("dac"))),
                    False,
                )
                carrier_II = np.cos(2 * np.pi * IF_freq * t_arr + self.template_match_phase)
                carrier_IQ = np.sin(2 * np.pi * IF_freq * t_arr + self.template_match_phase)
                carrier_QI = -np.sin(2 * np.pi * IF_freq * t_arr + self.template_match_phase)
                carrier_QQ = np.cos(2 * np.pi * IF_freq * t_arr + self.template_match_phase)
                match_events_list.append(
                    pls.setup_template_matching_pair(
                        input_port=self.sample_port,
                        template1=carrier_II,
                        template2=1j * carrier_IQ,
                    )
                )
                match_events_list.append(
                    pls.setup_template_matching_pair(
                        input_port=self.sample_port,
                        template1=carrier_QI,
                        template2=1j * carrier_QQ,
                    )
                )

            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            for i in range(2):
                pls.select_scale(T, i, self.control_port, group=0)
                pls.output_pulse(T, control_pulse_list)
                T += self.control_duration

                if nr_readout_pulses > 1:
                    pls.reset_phase(T, self.readout_port)
                pls.output_pulse(T, readout_pulse_list)
                pls.store(T + self.readout_sample_delay)
                pls.match(T + self.template_match_delay, match_events_list)
                T += self.readout_duration + self.wait_delay

            # **************************
            # *** Run the experiment ***
            # **************************

            pls.run(period=T, repeat_count=1, num_averages=self.num_averages)
            self.t_arr, self.store_arr = pls.get_store_data()
            match_arr_temp = pls.get_template_matching_data(match_events_list)
            if nr_readout_pulses == 1:
                self.match_arr = match_arr_temp
            if nr_readout_pulses == 2:
                self.match_arr = []
                for i in range(0, len(match_arr_temp) - 1, 2):
                    self.match_arr.append(match_arr_temp[i] + match_arr_temp[i + 1])
            if nr_readout_pulses == 3:
                self.match_arr = []
                i = 2
                self.match_arr.append(match_arr_temp[i] + match_arr_temp[i + 1])
                i = 4
                self.match_arr.append(match_arr_temp[i] + match_arr_temp[i + 1])
                self.match_arr.append(match_arr_temp[0])
                self.match_arr.append(match_arr_temp[1])
                i = 6
                self.match_arr.append(match_arr_temp[i] + match_arr_temp[i + 1])
                i = 8
                self.match_arr.append(match_arr_temp[i] + match_arr_temp[i + 1])

        return self.save()

    def save(self, save_filename: str = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "SingleShotMultiplexedReadout":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = npt.NDArray[np.float64] = h5f["readout_freq"][()]  # type: ignore
            readout_freq_IF = npt.NDArray[np.float64] = h5f["readout_freq_IF"][()]  # type: ignore
            control_freq = npt.NDArray[np.float64] = h5f["control_freq"][()]  # type: ignore
            readout_amp = npt.NDArray[np.float64] = h5f["readout_amp"][()]  # type: ignore
            control_amp = npt.NDArray[np.float64] = h5f["control_amp"][()]  # type: ignore
            readout_duration = h5f.attrs["readout_duration"]
            control_duration = h5f["control_duration"]
            sample_duration = h5f.attrs["sample_duration"]
            readout_port = h5f.attrs["readout_port"]
            control_port = npt.NDArray[np.int64] = h5f["control_port"][()]  # type: ignore
            sample_port = h5f.attrs["sample_port"]
            wait_delay = h5f.attrs["wait_delay"]
            readout_sample_delay = h5f.attrs["readout_sample_delay"]
            num_averages = h5f.attrs["num_averages"]

            t_arr: npt.NDArray[np.float64] = h5f["t_arr"][()]  # type: ignore
            store_arr: npt.NDArray[np.complex128] = h5f["store_arr"][()]  # type: ignore
            match_arr: npt.NDArray[np.float64] = h5f["match_arr"][()]  # type: ignore

            try:
                drag = float(h5f.attrs["drag"])  # type: ignore
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
            template_match_delay=template_match_delay,
            template_match_duration=template_match_duration,
            num_averages=num_averages,
            drag=drag,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr
        self.match_arr = match_arr
        self.readout_freq_IF = readout_freq_IF

        return self

    def analyze(self, rotate_optimally: bool = True, all_plots: bool = False):
        if self.t_arr is None:
            raise RuntimeError
        if self.store_arr is None:
            raise RuntimeError
        if self.match_arr is None:
            raise RuntimeError
        import matplotlib.pyplot as plt

        ret_fig = []
        """
        fs = 1 / (self.t_arr[1] - self.t_arr[0])
        IDX_LOW = int(round(self.template_match_delay * fs))
        IDX_HIGH = int(round((self.template_match_delay + self.template_match_duration) * fs))
        t_low = self.t_arr[IDX_LOW]
        t_high = self.t_arr[IDX_HIGH]
        """
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

        fig2, ax = plt.subplots(1, len(self.readout_freq), tight_layout=True)
        if len(self.readout_freq) == 1:
            ax = [ax]
        ax[0].set_ylabel("Quadrature [FS]")
        complex_match_data_list = []
        for i in range(0, len(self.match_arr), 2):
            complex_match_data_list.append(self.match_arr[i] + 1j * self.match_arr[i + 1])

        """    
        avg_data = np.array(
            [
                np.sum(self.store_arr[0, 0, IDX_LOW:IDX_HIGH]),
                np.sum(self.store_arr[1, 0, IDX_LOW:IDX_HIGH]),
            ]
        )
        """

        for i in range(len(complex_match_data_list)):
            ax[i].set_title(str(np.sort(self.readout_freq)[i] * 1e-9) + " GHz")
            complex_match_data = complex_match_data_list[i]
            if rotate_optimally:
                _, angle = rotate_opt(complex_match_data, True)
            else:
                angle = 0
            print("Angle of rotationg the data in post-processing: ", angle)

            ground_data = complex_match_data[::2] * np.exp(1j * angle)
            excited_data = complex_match_data[1::2] * np.exp(1j * angle)
            ax[i].plot(ground_data.real, ground_data.imag, ".", alpha=0.2, label="ground")
            ax[i].plot(excited_data.real, excited_data.imag, ".", alpha=0.2, label="excited")
            ax[i].legend()
            ax[i].set_xlabel("In phase [FS]")
            ax[i].set_aspect("equal")

        ret_fig.append(fig2)

        return ret_fig
