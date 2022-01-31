# -*- coding: utf-8 -*-
"""
Measure Rabi oscillation by changing the amplitude of the control pulse.

The control pulse has a sin^2 envelope, while the readout pulse is square.
"""
import ast
import math
import os
import time

import h5py
import numpy as np
from numpy.typing import ArrayLike

from mla_server import set_dc_bias
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.utils import get_sourcecode, sin2


class RabiAmp:
    def __init__(
        self,
        readout_freq: float,
        control_freq: float,
        readout_port: int,
        control_port: int,
        readout_amp: float,
        readout_duration: float,
        control_duration: float,
        sample_duration: float,
        sample_port: int,
        control_amp_arr: ArrayLike,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
        jpa_params=None,
    ):
        self.readout_freq = readout_freq
        self.control_freq = control_freq
        self.readout_port = readout_port
        self.control_port = control_port
        self.readout_amp = readout_amp
        self.readout_duration = readout_duration
        self.control_duration = control_duration
        self.sample_duration = sample_duration
        self.sample_port = sample_port
        self.control_amp_arr = control_amp_arr
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages

        self.rabi_n = len(control_amp_arr)
        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run

        self.jpa_params = jpa_params

    def run(
        self,
        presto_address,
        presto_port=None,
        ext_ref_clk=False,
    ):
        # Instantiate interface class
        with pulsed.Pulsed(
                address=presto_address,
                port=presto_port,
                ext_ref_clk=ext_ref_clk,
                adc_mode=AdcMode.Mixed,
                adc_fsample=AdcFSample.G2,
                dac_mode=[DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
                dac_fsample=[DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
        ) as pls:
            pls.hardware.set_adc_attenuation(self.sample_port, 0.0)
            pls.hardware.set_dac_current(self.readout_port, 32_000)
            pls.hardware.set_dac_current(self.control_port, 32_000)
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
                pls.hardware.set_lmx(self.jpa_params['jpa_pump_freq'], self.jpa_params['jpa_pump_pwr'])
                set_dc_bias(self.jpa_params['jpa_bias_port'], self.jpa_params['jpa_bias'])
                time.sleep(1.0)

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
                scales=self.readout_amp,
            )
            pls.setup_scale_lut(
                output_ports=self.control_port,
                group=0,
                scales=self.control_amp_arr,
            )

            # Setup readout and control pulses
            # use setup_long_drive to create a pulse with square envelope
            # setup_long_drive supports smooth rise and fall transitions for the pulse,
            # but we keep it simple here
            readout_pulse = pls.setup_long_drive(
                output_port=self.readout_port,
                group=0,
                duration=self.readout_duration,
                amplitude=1.0,
                amplitude_q=1.0,
                rise_time=0e-9,
                fall_time=0e-9,
            )
            # For the control pulse we create a sine-squared envelope,
            # and use setup_template to use the user-defined envelope
            control_ns = int(round(self.control_duration *
                                   pls.get_fs("dac")))  # number of samples in the control template
            control_envelope = sin2(control_ns)
            control_pulse = pls.setup_template(
                output_port=self.control_port,
                group=0,
                template=control_envelope,
                template_q=control_envelope,
                envelope=True,
            )

            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            # Control pulse
            pls.reset_phase(T, self.control_port)
            pls.output_pulse(T, control_pulse)
            # Readout pulse starts right after control pulse
            T += self.control_duration
            pls.reset_phase(T, self.readout_port)
            pls.output_pulse(T, readout_pulse)
            # Sampling window
            pls.store(T + self.readout_sample_delay)
            # Move to next Rabi amplitude
            T += self.readout_duration
            pls.next_scale(T, self.control_port)  # every iteration will have a different amplitude
            # Wait for decay
            T += self.wait_delay

            # **************************
            # *** Run the experiment ***
            # **************************
            # repeat the whole sequence `rabi_n` times
            # then average `num_averages` times
            pls.run(
                period=T,
                repeat_count=self.rabi_n,
                num_averages=self.num_averages,
                print_time=True,
            )
            t_arr, (data_I, data_Q) = pls.get_store_data()

            if self.jpa_params is not None:
                pls.hardware.set_lmx(0.0, 0.0)
                set_dc_bias(self.jpa_params['jpa_bias_port'], 0.0)

        self.t_arr = t_arr
        self.store_arr = data_I + 1j * data_Q

        return self.save()

    def save(self, save_filename=None):
        # *************************
        # *** Save data to HDF5 ***
        # *************************
        if save_filename is None:
            script_path = os.path.realpath(__file__)  # full path of current script
            current_dir, script_basename = os.path.split(script_path)
            script_filename = os.path.splitext(script_basename)[0]  # name of current script
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # current date and time
            save_basename = f"{script_filename:s}_{timestamp:s}.h5"  # name of save file
            save_path = os.path.join(current_dir, "data", save_basename)  # full path of save file
        else:
            save_path = os.path.realpath(save_filename)

        source_code = get_sourcecode(__file__)  # save also the sourcecode of the script for future reference
        with h5py.File(save_path, "w") as h5f:
            dt = h5py.string_dtype(encoding='utf-8')
            ds = h5f.create_dataset("source_code", (len(source_code), ), dt)
            for ii, line in enumerate(source_code):
                ds[ii] = line

            for attribute in self.__dict__:
                print(f"{attribute}: {self.__dict__[attribute]}")
                if attribute.startswith("_"):
                    # don't save private attributes
                    continue
                if attribute == "jpa_params":
                    h5f.attrs[attribute] = str(self.__dict__[attribute])
                elif np.isscalar(self.__dict__[attribute]):
                    h5f.attrs[attribute] = self.__dict__[attribute]
                else:
                    h5f.create_dataset(attribute, data=self.__dict__[attribute])
        print(f"Data saved to: {save_path}")
        return save_path

    @classmethod
    def load(cls, load_filename):
        with h5py.File(load_filename, "r") as h5f:
            num_averages = h5f.attrs["num_averages"]
            control_freq = h5f.attrs["control_freq"]
            readout_freq = h5f.attrs["readout_freq"]
            readout_duration = h5f.attrs["readout_duration"]
            control_duration = h5f.attrs["control_duration"]
            readout_amp = h5f.attrs["readout_amp"]
            sample_duration = h5f.attrs["sample_duration"]
            # rabi_n = h5f.attrs["rabi_n"]
            wait_delay = h5f.attrs["wait_delay"]
            readout_sample_delay = h5f.attrs["readout_sample_delay"]

            control_amp_arr = h5f["control_amp_arr"][()]
            t_arr = h5f["t_arr"][()]
            store_arr = h5f["store_arr"][()]
            # source_code = h5f["source_code"][()]

            # these were added later
            try:
                readout_port = h5f.attrs["readout_port"]
            except KeyError:
                readout_port = 0
            try:
                control_port = h5f.attrs["control_port"]
            except KeyError:
                control_port = 0
            try:
                sample_port = h5f.attrs["sample_port"]
            except KeyError:
                sample_port = 0
            try:
                jpa_params = ast.literal_eval(h5f.attrs["jpa_params"])
            except KeyError:
                jpa_params = None

        self = cls(
            readout_freq,
            control_freq,
            readout_port,
            control_port,
            readout_amp,
            readout_duration,
            control_duration,
            sample_duration,
            sample_port,
            control_amp_arr,
            wait_delay,
            readout_sample_delay,
            num_averages,
            jpa_params,
        )
        self.control_amp_arr = control_amp_arr
        self.t_arr = t_arr
        self.store_arr = store_arr

        return self

    def analyze(self, all_plots=False):
        if self.t_arr is None:
            raise RuntimeError
        if self.store_arr is None:
            raise RuntimeError

        import matplotlib.pyplot as plt
        from presto.utils import rotate_opt

        ret_fig = []

        t_low = 1500 * 1e-9
        t_high = 2000 * 1e-9
        # t_span = t_high - t_low
        idx_low = np.argmin(np.abs(self.t_arr - t_low))
        idx_high = np.argmin(np.abs(self.t_arr - t_high))
        idx = np.arange(idx_low, idx_high)
        # nr_samples = len(idx)

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
        data = rotate_opt(resp_arr)

        # Fit data
        popt_x, perr_x = _fit_period(self.control_amp_arr, np.real(data))
        period = popt_x[3]
        period_err = perr_x[3]
        pi_amp = period / 2
        pi_2_amp = period / 4
        print("Tau pulse amplitude: {} +- {} FS".format(period, period_err))
        print("Pi pulse amplitude: {} +- {} FS".format(pi_amp, period_err / 2))
        print("Pi/2 pulse amplitude: {} +- {} FS".format(pi_2_amp, period_err / 4))

        if all_plots:
            fig2, ax2 = plt.subplots(4, 1, sharex=True, figsize=(6.4, 6.4), tight_layout=True)
            ax21, ax22, ax23, ax24 = ax2
            ax21.plot(self.control_amp_arr, np.abs(data))
            ax22.plot(self.control_amp_arr, np.angle(data))
            ax23.plot(self.control_amp_arr, np.real(data))
            ax23.plot(self.control_amp_arr, _func(self.control_amp_arr, *popt_x), '--')
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
        ax3.plot(self.control_amp_arr, mult * np.real(data), '.')
        ax3.plot(self.control_amp_arr, mult * _func(self.control_amp_arr, *popt_x), '--')
        ax3.set_ylabel(f"I quadrature [{unit:s}FS]")
        ax3.set_xlabel("Pulse amplitude [FS]")
        fig3.show()
        ret_fig.append(fig3)

        return ret_fig


def _func(t, offset, amplitude, T2, period, phase):
    frequency = 1 / period
    return offset + amplitude * np.exp(-t / T2) * np.cos(math.tau * frequency * t + phase)




def _fit_period(x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    from scipy.optimize import curve_fit
    # from scipy.optimize import least_squares

    pkpk = np.max(y) - np.min(y)
    offset = np.min(y) + pkpk / 2
    amplitude = 0.5 * pkpk
    T2 = 0.5 * (np.max(x) - np.min(x))
    freqs = np.fft.rfftfreq(len(x), x[1] - x[0])
    fft = np.fft.rfft(y)
    frequency = freqs[1 + np.argmax(np.abs(fft[1:]))]
    period = 1 / frequency
    first = (y[0] - offset) / amplitude
    if first > 1.:
        first = 1.
    elif first < -1.:
        first = -1.
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
    # def _residuals(p, x, y):
    #     offset, amplitude, T2, period, phase = p
    #     return _func(x, offset, amplitude, T2, period, phase) - y
    # res = least_squares(_residuals, p0, args=(x, y))
    # # offset, amplitude, T2, period, phase = res.x
    # return res.x, np.zeros_like(res.x)


if __name__ == "__main__":
    WHICH_QUBIT = 2  # 1 (higher resonator) or 2 (lower resonator)
    USE_JPA = True
    WITH_COUPLER = False

    # Presto's IP address or hostname
    ADDRESS = "130.237.35.90"
    PORT = 42874
    # ADDRESS = "127.0.0.1"
    # PORT = 7878
    EXT_REF_CLK = False  # set to True to lock to an external reference clock
    jpa_bias_port = 1

    if WHICH_QUBIT == 1:
        if WITH_COUPLER:
            readout_freq = 6.167_009 * 1e9  # Hz, frequency for resonator readout
            control_freq = 3.556_520 * 1e9  # Hz
        else:
            readout_freq = 6.166_600 * 1e9  # Hz, frequency for resonator readout
            control_freq = 3.557_866 * 1e9  # Hz
        control_port = 3
        jpa_pump_freq = 2 * 6.169e9  # Hz
        jpa_pump_pwr = 11  # lmx units
        jpa_bias = +0.437  # V
    elif WHICH_QUBIT == 2:
        if WITH_COUPLER:
            readout_freq = 6.029_130 * 1e9  # Hz, frequency for resonator readout
            control_freq = 4.093_042 * 1e9  # Hz
        else:
            readout_freq = 6.028_400 * 1e9  # Hz, frequency for resonator readout
            control_freq = 4.093_372 * 1e9  # Hz
        control_port = 4
        jpa_pump_freq = 2 * 6.031e9  # Hz
        jpa_pump_pwr = 9  # lmx units
        jpa_bias = +0.449  # V
    else:
        raise ValueError

    # cavity drive: readout
    readout_amp = 0.4  # FS
    readout_duration = 2e-6  # s, duration of the readout pulse
    readout_port = 1

    # qubit drive: control
    control_duration = 20e-9  # s, duration of the control pulse

    # cavity readout: sample
    sample_duration = 4 * 1e-6  # s, duration of the sampling window
    sample_port = 1

    # Rabi experiment
    num_averages = 1_000
    rabi_n = 128  # number of steps when changing duration of control pulse
    control_amp_arr = np.linspace(0.0, 1.0, rabi_n)  # FS, amplitudes for control pulse
    wait_delay = 200e-6  # s, delay between repetitions to allow the qubit to decay
    readout_sample_delay = 290 * 1e-9  # s, delay between readout pulse and sample window to account for latency

    jpa_params = {
        'jpa_bias': jpa_bias,
        'jpa_bias_port': jpa_bias_port,
        'jpa_pump_freq': jpa_pump_freq,
        'jpa_pump_pwr': jpa_pump_pwr,
    } if USE_JPA else None

    rabi = RabiAmp(
        readout_freq,
        control_freq,
        readout_port,
        control_port,
        readout_amp,
        readout_duration,
        control_duration,
        sample_duration,
        sample_port,
        control_amp_arr,
        wait_delay,
        readout_sample_delay,
        num_averages,
        jpa_params,
    )
    rabi.run(ADDRESS, PORT, EXT_REF_CLK)
    fig = rabi.analyze()
