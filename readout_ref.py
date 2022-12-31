# -*- coding: utf-8 -*-
"""Pulsed readout starting from ground and excited state.

Acquire reference templates for template matching.
"""
import ast
from typing import Optional

import h5py
import numpy as np

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.pulsed import MAX_TEMPLATE_LEN
from presto.utils import sin2, to_pm_pi

from _base import Base

DAC_CURRENT = 32_000  # uA
CONVERTER_CONFIGURATION = {
    "adc_mode": AdcMode.Mixed,
    "adc_fsample": AdcFSample.G4,
    "dac_mode": [DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
    "dac_fsample": [DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
}
IDX_LOW = 1_500
IDX_HIGH = 2_000


class ReadoutRef(Base):
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
        jpa_params: dict = None,
        drag: float = 0.0,
        clear: dict = None,
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
        self.drag = drag
        self.clear = clear

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run

        self.jpa_params = jpa_params

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
            pls.hardware.set_adc_attenuation(self.sample_port, 0.0)
            pls.hardware.set_dac_current(self.readout_port, DAC_CURRENT)
            pls.hardware.set_dac_current(self.control_port, DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.readout_port, 0)
            pls.hardware.set_inv_sinc(self.control_port, 0)
            pls.hardware.configure_mixer(
                freq=self.readout_freq,
                in_ports=self.sample_port,
                out_ports=self.readout_port,
                sync=False,  # sync in next call
            )
            pls.hardware.configure_mixer(
                freq=self.control_freq,
                out_ports=self.control_port,
                sync=True,  # sync here
            )
            if self.jpa_params is not None:
                pls.hardware.set_lmx(
                    self.jpa_params["pump_freq"],
                    self.jpa_params["pump_pwr"],
                    self.jpa_params["pump_port"],
                )
                pls.hardware.set_dc_bias(self.jpa_params["bias"], self.jpa_params["bias_port"])
                pls.hardware.sleep(1.0, False)

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for frequencies
            pls.setup_freq_lut(
                output_ports=self.readout_port,
                group=0,
                frequencies=0.0,
                phases=0.0,
                phases_q=0.0,
            )
            pls.setup_freq_lut(
                output_ports=self.control_port,
                group=0,
                frequencies=0.0,
                phases=0.0,
                phases_q=0.0,
            )

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(
                output_ports=self.readout_port,
                group=0,
                scales=1.0,  # set it in pulse
            )
            pls.setup_scale_lut(
                output_ports=self.control_port,
                group=0,
                scales=self.control_amp,
            )

            # Setup readout and control pulses
            # use setup_long_drive to create a pulse with square envelope
            # setup_long_drive supports smooth rise and fall transitions for the pulse,
            # but we keep it simple here
            if self.clear is None:
                readout_pulse = pls.setup_long_drive(
                    output_port=self.readout_port,
                    group=0,
                    duration=self.readout_duration,
                    amplitude=self.readout_amp,
                    amplitude_q=self.readout_amp,
                    rise_time=0e-9,
                    fall_time=0e-9,
                )
            else:
                from presto._clear import clear

                lens, amps = clear(self.readout_duration * 1e9, **self.clear)
                lens = [int(round(l * pls.get_fs("dac"))) for l in lens]

                readout_ns = int(
                    round(self.readout_duration * pls.get_fs("dac"))
                )  # number of samples in the control template
                readout_envelope = np.zeros(readout_ns)
                start = 0
                for l, a in zip(lens, amps):
                    stop = start + l
                    readout_envelope[start:stop] = a
                    start += l
                readout_envelope *= self.readout_amp

                readout_pulse = pls.setup_template(
                    output_port=self.readout_port,
                    group=0,
                    template=readout_envelope,
                    template_q=readout_envelope,
                    envelope=True,
                )

            # For the control pulse we create a sine-squared envelope,
            # and use setup_template to use the user-defined envelope
            control_ns = int(
                round(self.control_duration * pls.get_fs("dac"))
            )  # number of samples in the control template
            control_envelope = sin2(control_ns, drag=self.drag)
            control_pulse = pls.setup_template(
                output_port=self.control_port,
                group=0,
                template=control_envelope,
                template_q=control_envelope if self.drag == 0.0 else None,
                envelope=True,
            )

            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            for ii in range(2):
                pls.reset_phase(T, self.control_port)
                if ii > 0:
                    # pi pulse
                    pls.output_pulse(T, control_pulse)
                # Readout pulse starts after control pulse
                T += self.control_duration
                pls.reset_phase(T, self.readout_port)
                pls.output_pulse(T, readout_pulse)
                # Sampling window
                pls.store(T + self.readout_sample_delay)
                # Move to next iteration
                T += self.readout_duration
                T += self.wait_delay

            if self.jpa_params is not None:
                # adjust period to minimize effect of JPA idler
                idler_freq = self.jpa_params["pump_freq"] - self.readout_freq
                idler_if = abs(idler_freq - self.readout_freq)  # NCO at readout_freq
                idler_period = 1 / idler_if
                T_clk = int(round(T * pls.get_clk_f()))
                idler_period_clk = int(round(idler_period * pls.get_clk_f()))
                # first make T a multiple of idler period
                if T_clk % idler_period_clk > 0:
                    T_clk += idler_period_clk - (T_clk % idler_period_clk)
                # then make it off by one clock cycle
                T_clk += 1
                T = T_clk * pls.get_clk_T()

            # **************************
            # *** Run the experiment ***
            # **************************
            pls.run(
                period=T,
                repeat_count=1,
                num_averages=self.num_averages,
                print_time=True,
            )
            self.t_arr, self.store_arr = pls.get_store_data()

            if self.jpa_params is not None:
                pls.hardware.set_lmx(0.0, 0.0, self.jpa_params["pump_port"])
                pls.hardware.set_dc_bias(0.0, self.jpa_params["bias_port"])

        return self.save()

    def save(self, save_filename: str = None) -> str:
        return super().save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "ReadoutRef":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = h5f.attrs["readout_freq"]
            control_freq = h5f.attrs["control_freq"]
            readout_amp = h5f.attrs["readout_amp"]
            control_amp = h5f.attrs["control_amp"]
            readout_duration = h5f.attrs["readout_duration"]
            control_duration = h5f.attrs["control_duration"]
            sample_duration = h5f.attrs["sample_duration"]
            readout_port = h5f.attrs["readout_port"]
            control_port = h5f.attrs["control_port"]
            sample_port = h5f.attrs["sample_port"]
            wait_delay = h5f.attrs["wait_delay"]
            readout_sample_delay = h5f.attrs["readout_sample_delay"]
            num_averages = h5f.attrs["num_averages"]
            drag = h5f.attrs["drag"]

            jpa_params = ast.literal_eval(h5f.attrs["jpa_params"])
            clear = ast.literal_eval(h5f.attrs["clear"])

            t_arr = h5f["t_arr"][()]
            store_arr = h5f["store_arr"][()]

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
            jpa_params=jpa_params,
            drag=drag,
            clear=clear,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr

        return self

    def analyze(self, plot: bool = True, rotate: bool = False, match_len: Optional[int] = None):
        assert self.t_arr is not None
        assert self.store_arr is not None

        ret_fig = []

        nr_samples = len(self.t_arr)

        trace_g = self.store_arr[0, 0, :]
        trace_e = self.store_arr[1, 0, :]
        if rotate:
            trace_g, trace_e = _rotate_opt(trace_g, trace_e)

        distance = np.abs(trace_e - trace_g)

        max_match_len = MAX_TEMPLATE_LEN // 2  # I and Q
        if match_len is None:
            match_len = max_match_len
        else:
            match_len = int(match_len)
            if match_len > max_match_len:  # I and Q
                raise ValueError("maximum match length is {max_match_len}, got {match_len}")

        max_idx = 0
        max_dist = 0.0
        for idx in range(0, nr_samples - match_len, 2):
            dist = np.sum(distance[idx : idx + match_len])
            if dist > max_dist:
                max_dist = dist
                max_idx = idx

        ref_g = trace_g[max_idx : max_idx + match_len]
        ref_e = trace_e[max_idx : max_idx + match_len]
        match_t_in_store = self.t_arr[max_idx]
        readout_match_delay = self.readout_sample_delay + match_t_in_store
        print(f"Match starts at {1e9 * match_t_in_store:.0f} ns in store")
        print(f"Readout-match delay: {1e9 * readout_match_delay:.0f} ns")
        ret_dict = {
            "trace_g": trace_g,
            "trace_e": trace_e,
            "ref_g": ref_g,
            "ref_e": ref_e,
            "match_t_in_store": match_t_in_store,
            "readout_match_delay": readout_match_delay,
        }

        if plot:
            import matplotlib.pyplot as plt

            fig1, ax1 = plt.subplots(4, 1, sharex=True, tight_layout=True)
            for ax_ in ax1:
                ax_.axvspan(
                    1e9 * self.t_arr[max_idx],
                    1e9 * self.t_arr[max_idx + match_len],
                    facecolor="#dfdfdf",
                )
            ax1[0].plot(1e9 * self.t_arr, np.abs(trace_g), label="|g>")
            ax1[0].plot(1e9 * self.t_arr, np.abs(trace_e), label="|e>")
            ax1[1].plot(1e9 * self.t_arr, np.angle(trace_g))
            ax1[1].plot(1e9 * self.t_arr, np.angle(trace_e))
            ax1[2].plot(1e9 * self.t_arr, np.real(trace_g))
            ax1[2].plot(1e9 * self.t_arr, np.real(trace_e))
            ax1[3].plot(1e9 * self.t_arr, np.imag(trace_g))
            ax1[3].plot(1e9 * self.t_arr, np.imag(trace_e))
            ax1[-1].set_xlabel("Time [ns]")
            ax1[0].set_ylabel("A [FS]")
            ax1[1].set_ylabel("φ [rad]")
            ax1[2].set_ylabel("I [FS]")
            ax1[3].set_ylabel("Q [FS]")
            ax1[0].legend()
            fig1.show()
            ret_fig.append(fig1)

            data_max = distance.max()
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

            fig2, ax2 = plt.subplots(tight_layout=True)
            ax2.axvspan(
                1e9 * self.t_arr[max_idx],
                1e9 * self.t_arr[max_idx + match_len],
                facecolor="#dfdfdf",
            )
            ax2.plot(1e9 * self.t_arr, mult * distance)
            ax2.set_xlabel("Time [ns]")
            ax2.set_ylabel(f"Distance [{unit}FS]")
            ax2.set_title(r"$d = \left\Vert\left|e\right> - \left|g\right>\right\Vert$")
            fig2.show()
            ret_fig.append(fig2)

            return ret_dict, ret_fig
        else:
            return ret_dict


def _rotate_opt(trace_g, trace_e):
    data = trace_e - trace_g  # complex distance

    # calculate the mean of data.imag**2 in steps of 1 deg
    N = 360
    _mean = np.zeros(N)
    for ii in range(N):
        _data = data * np.exp(1j * 2 * np.pi / N * ii)
        _mean[ii] = np.mean(_data.imag**2)

    # the mean goes like cos(x)**2
    # FFT and find the phase at frequency "2"
    fft = np.fft.rfft(_mean) / N
    # first solution
    x_fft1 = -np.angle(fft[2])  # compensate for measured phase
    x_fft1 -= np.pi  # we want to be at the zero of cos(2x)
    x_fft1 /= 2  # move from frequency "2" to "1"
    # there's a second solution np.pi away (a minus sign)
    x_fft2 = x_fft1 + np.pi

    # convert to +/- interval
    x_fft1 = to_pm_pi(x_fft1)
    x_fft2 = to_pm_pi(x_fft2)
    # choose the closest to zero
    if np.abs(x_fft1) < np.abs(x_fft2):
        x_fft = x_fft1
    else:
        x_fft = x_fft2

    # rotate the data and return a copy
    trace_g = trace_g * np.exp(1j * x_fft)
    trace_e = trace_e * np.exp(1j * x_fft)

    return trace_g, trace_e
