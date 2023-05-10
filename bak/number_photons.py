# -*- coding: utf-8 -*-
import ast
import math
import os
import time

import h5py
import numpy as np

from mla_server import set_dc_bias
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.utils import get_sourcecode, sin2

"""
# import load_ramsey_single

WHICH_QUBIT = 2  # 1 (higher resonator) or 2 (lower resonator)
USE_JPA = True
WITH_COUPLER = False

# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42874
EXT_REF_CLK = False  # set to True to lock to an external reference clock
jpa_bias_port = 1

if WHICH_QUBIT == 1:
    if WITH_COUPLER:
        readout_freq = 6.167_009 * 1e9  # Hz, frequency for resonator readout
        control_amp = 0.267  # FS <-- pi/2 pulse
        control_freq = 3.556_520 * 1e9  # Hz
    else:
        readout_freq = 6.166_600 * 1e9  # Hz, frequency for resonator readout
        control_amp = 0.05129  # FS <-- pi/2 pulse
        control_freq = 3.557_866 * 1e9  # Hz
    control_port = 3
    jpa_pump_freq = 2 * 6.169e9  # Hz
    jpa_pump_pwr = 11  # lmx units
    jpa_bias = +0.437  # V
elif WHICH_QUBIT == 2:
    if WITH_COUPLER:
        readout_freq = 6.029_130 * 1e9  # Hz, frequency for resonator readout
        control_amp = 0.380  # FS <-- pi/2 pulse
        control_freq = 4.093_042 * 1e9  # Hz
    else:
        readout_freq = 6.028_450 * 1e9  # Hz, frequency for resonator readout
        control_amp = 0.3808  # FS <-- pi/2 pulse
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
# control_freq += 500e3  # Hz, detune from qubit frequency
control_freq += 10 * 1e6  # Hz, detune from qubit frequency
control_if = 0 * 1e6  # Hz
control_duration = 20 * 1e-9  # s, duration of the control pulse

# cavity readout: sample
sample_duration = 4 * 1e-6  # s, duration of the sampling window
sample_port = 1

# Ramsey experiment
num_averages = 1_000
# nr_delays = 128  # number of steps when changing delay between control and readout pulses
nr_delays = 256  # number of steps when changing delay between control and readout pulses
# dt_delays = 0.1 * 1e-6  # s, step size when changing delay between control and readout pulses
dt_delays = 2 * 1e-9  # s, step size when changing delay between control and readout pulses
wait_delay = 200e-6  # s, delay between repetitions to allow the qubit to decay
readout_sample_delay = 290 * 1e-9  # s, delay between readout pulse and sample window to account for latency

first_pulse_amp_arr = np.sqrt(np.linspace(0.0, 1.0, 256))
buffer_time = 1.5 * 1e-6  # time between first and second readout pulses
"""


class NumberPhotons:
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
        control_amp: float,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
        ramsey_delay_arr,
        first_pulse_amp_arr,
        buffer_time: float,
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
        self.control_amp = control_amp
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages
        self.ramsey_delay_arr = ramsey_delay_arr
        self.first_pulse_amp_arr = first_pulse_amp_arr
        self.buffer_time = buffer_time

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run

        self.jpa_params = jpa_params

        # for plotting
        self._amp_idx = 0

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
                pls.hardware.set_lmx(
                    self.jpa_params["jpa_pump_freq"], self.jpa_params["jpa_pump_pwr"]
                )
                set_dc_bias(self.jpa_params["jpa_bias_port"], self.jpa_params["jpa_bias"])
                time.sleep(1.0)

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for frequencies
            # we only need to use carrier 1
            pls.setup_freq_lut(
                output_ports=self.readout_port,
                group=1,
                frequencies=0.0,
                phases=0.0,
                phases_q=0.0,
            )
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
                group=1,
                scales=self.first_pulse_amp_arr,
            )
            pls.setup_scale_lut(
                output_ports=self.readout_port,
                group=0,
                scales=self.readout_amp,
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
            first_pulse = pls.setup_long_drive(
                output_port=self.readout_port,
                group=1,
                duration=self.readout_duration,
                amplitude=1.0,
                amplitude_q=1.0,
                rise_time=0e-9,
                fall_time=0e-9,
            )
            readout_pulse = pls.setup_long_drive(
                output_port=self.readout_port,
                group=0,
                duration=self.readout_duration,
                amplitude=1.0,
                amplitude_q=1.0,
                rise_time=0e-9,
                fall_time=0e-9,
            )
            control_ns = int(
                round(self.control_duration * pls.get_fs("dac"))
            )  # number of samples in the control template
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
            for ramsey_delay in self.ramsey_delay_arr:
                # first "readout" pulse
                pls.reset_phase(T, self.readout_port)
                pls.output_pulse(T, first_pulse)
                T += self.readout_duration
                T_end_first = T
                # first pi/2 pulse
                pls.reset_phase(T, self.control_port)
                pls.output_pulse(T, control_pulse)
                T += self.control_duration
                # Ramsey delay
                T += ramsey_delay
                # second pi/2 pulse
                pls.output_pulse(T, control_pulse)
                T += self.control_duration
                # Readout pulse starts after control pulse,
                # `buffer_time` away from the first pulse
                assert T <= T_end_first + self.buffer_time
                T = T_end_first + self.buffer_time
                pls.reset_phase(T, self.readout_port)
                pls.output_pulse(T, readout_pulse)
                # Sampling window
                pls.store(T + self.readout_sample_delay)
                # Move to next iteration
                T += self.readout_duration
                T += self.wait_delay
            # next scale on first pulse
            pls.next_scale(T, self.readout_port, 1)
            T += self.wait_delay

            # **************************
            # *** Run the experiment ***
            # **************************
            pls.run(
                period=T,
                repeat_count=len(self.first_pulse_amp_arr),
                num_averages=self.num_averages,
                print_time=True,
            )
            t_arr, (data_I, data_Q) = pls.get_store_data()

            if self.jpa_params is not None:
                pls.hardware.set_lmx(0.0, 0.0)
                set_dc_bias(self.jpa_params["jpa_bias_port"], 0.0)

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

        source_code = get_sourcecode(
            __file__
        )  # save also the sourcecode of the script for future reference
        with h5py.File(save_path, "w") as h5f:
            dt = h5py.string_dtype(encoding="utf-8")
            ds = h5f.create_dataset("source_code", (len(source_code),), dt)
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
            readout_freq = h5f.attrs["readout_freq"]
            control_freq = h5f.attrs["control_freq"]
            readout_port = h5f.attrs["readout_port"]
            control_port = h5f.attrs["control_port"]
            readout_amp = h5f.attrs["readout_amp"]
            readout_duration = h5f.attrs["readout_duration"]
            control_duration = h5f.attrs["control_duration"]
            sample_duration = h5f.attrs["sample_duration"]
            sample_port = h5f.attrs["sample_port"]
            control_amp = h5f.attrs["control_amp"]
            wait_delay = h5f.attrs["wait_delay"]
            readout_sample_delay = h5f.attrs["readout_sample_delay"]
            num_averages = h5f.attrs["num_averages"]
            ramsey_delay_arr = h5f["ramsey_delay_arr"][()]
            first_pulse_amp_arr = h5f["first_pulse_amp_arr"][()]
            buffer_time = h5f.attrs["buffer_time"]
            jpa_params = ast.literal_eval(h5f.attrs["jpa_params"])
            t_arr = h5f["t_arr"][()]
            store_arr = h5f["store_arr"][()]

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
            control_amp,
            wait_delay,
            readout_sample_delay,
            num_averages,
            ramsey_delay_arr,
            first_pulse_amp_arr,
            buffer_time,
            jpa_params,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr

        return self

    def analyze(self, crazy=False, blit=True):
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

        # Analyze
        resp_arr = np.mean(self.store_arr[:, 0, idx], axis=-1)
        data = rotate_opt(resp_arr).real

        nr_delays = len(self.ramsey_delay_arr)
        nr_amps = len(self.first_pulse_amp_arr)
        data.shape = (nr_amps, nr_delays)
        self._amp_idx = 0

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

        if np.mean(data) < 0:
            data *= -mult
        else:
            data *= mult
        data_max = np.max(data)
        data_min = np.min(data)
        data_rng = data_max - data_min

        # choose limits for colorbar
        cutoff = 0.0  # %
        lowlim = np.percentile(data, cutoff)
        highlim = np.percentile(data, 100.0 - cutoff)

        x_data = 1e9 * self.ramsey_delay_arr
        y_data = self.first_pulse_amp_arr**2 * 100

        # extent
        x_min = x_data[0]
        x_max = x_data[-1]
        x_rng = x_max - x_min
        dx = x_data[1] - x_data[0]
        y_min = y_data[0]
        y_max = y_data[-1]
        dy = y_data[1] - y_data[0]

        fig1 = plt.figure(tight_layout=True, figsize=(12.8, 4.8))
        ax11 = fig1.add_subplot(1, 2, 1)
        im = ax11.imshow(
            data,
            origin="lower",
            aspect="auto",
            interpolation="none",
            extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
            vmin=lowlim,
            vmax=highlim,
        )
        self._line_sel = ax11.axhline(y_data[self._amp_idx], ls="--", c="k", lw=3, animated=blit)
        ax11.set_xlabel("Ramsey delay [ns]")
        ax11.set_ylabel("First pulse power [%]")
        cb = fig1.colorbar(im)
        # cb.set_label(f"Response I quadrature [{unit:s}FS]")
        ax11.set_title(f"Response I quadrature [{unit:s}FS]")

        ax12 = fig1.add_subplot(1, 2, 2)
        ax12.yaxis.set_label_position("right")
        ax12.yaxis.tick_right()
        (self._line_slice,) = ax12.plot(
            x_data, data[self._amp_idx], ".", label="measured", animated=blit
        )
        try:
            if crazy:
                popt, _ = _fit_simple(x_data, data[self._amp_idx])
                fit_data = _func(x_data, *popt)
            else:
                popt, _ = _fit_simple(x_data, data[self._amp_idx])
                fit_data = _func(x_data, *popt)
            (self._line_fit,) = ax12.plot(x_data, fit_data, "--", label="fit", animated=blit)
        except Exception:
            (self._line_fit,) = ax12.plot(
                x_data, np.full_like(x_data, np.nan), "--", label="fit", animated=blit
            )
        ax12.set_xlim(x_min - 0.05 * x_rng, x_max + 0.05 * x_rng)
        ax12.set_ylim(data_min - 0.05 * data_rng, data_max + 0.05 * data_rng)
        ax12.set_xlabel("Ramsey delay [ns]")
        ax12.set_title(f"Response I quadrature [{unit:s}FS]")
        ax12.legend(loc="lower right", ncol=2)

        def onbuttonpress(event):
            if event.inaxes == ax11:
                self._amp_idx = np.argmin(np.abs(y_data - event.ydata))
                update()

        def onkeypress(event):
            if event.inaxes == ax11:
                if event.key == "up":
                    self._amp_idx += 1
                    if self._amp_idx >= nr_amps:
                        self._amp_idx = nr_amps - 1
                    update()
                elif event.key == "down":
                    self._amp_idx -= 1
                    if self._amp_idx < 0:
                        self._amp_idx = 0
                    update()

        def update():
            self._line_sel.set_ydata([y_data[self._amp_idx], y_data[self._amp_idx]])
            # ax1.set_title(f"amp = {amp_arr[AMP_IDX]:.2e}")
            # print(f"drive amp {self._amp_idx:d}: {qubit_amp_arr[self._amp_idx]:.2e} FS = {amp_dBFS[self._amp_idx]:.1f} dBFS")
            self._line_slice.set_ydata(data[self._amp_idx])
            try:
                if crazy:
                    popt, _ = _fit_simple(x_data, data[self._amp_idx])
                    fit_data = _func(x_data, *popt)
                else:
                    popt, _ = _fit_simple(x_data, data[self._amp_idx])
                    fit_data = _func(x_data, *popt)
                self._line_fit.set_ydata(fit_data)
            except Exception:
                self._line_fit.set_ydata(np.full_like(x_data, np.nan))
            # ax2.set_title("")
            if blit:
                fig1.canvas.restore_region(self._bg)
                ax11.draw_artist(self._line_sel)
                ax12.draw_artist(self._line_slice)
                ax12.draw_artist(self._line_fit)
                fig1.canvas.blit(fig1.bbox)
                # fig1.canvas.flush_events()
            else:
                fig1.canvas.draw()

        fig1.canvas.mpl_connect("button_press_event", onbuttonpress)
        fig1.canvas.mpl_connect("key_press_event", onkeypress)

        fig1.show()
        if blit:
            fig1.canvas.draw()
            # fig1.canvas.flush_events()
            self._bg = fig1.canvas.copy_from_bbox(fig1.bbox)
            ax11.draw_artist(self._line_sel)
            ax12.draw_artist(self._line_slice)
            ax12.draw_artist(self._line_fit)
            fig1.canvas.blit(fig1.bbox)

        ret_fig.append(fig1)

        # Fit the lowest powers
        idx_fit = np.s_[0:75]
        pwr_fit = y_data[idx_fit]
        n_fit = np.zeros_like(pwr_fit)
        for ii, fit_data in enumerate(data[idx_fit]):
            try:
                popt, _ = _fit_simple(x_data, fit_data)
                n_fit[ii] = popt[0]
            except Exception:
                n_fit[ii] = np.nan
        fig2, ax2 = plt.subplots(tight_layout=True)
        ax2.plot(pwr_fit, n_fit, ".")
        ax2.grid()
        ax2.set_xlabel("First pulse power [%]")
        ax2.set_ylabel("Fitted number of photons")
        fig2.show()
        ret_fig.append(fig2)

        return ret_fig


def _func(t, n0, phase0, amplitude, offset, slope):
    kappa = math.tau * 463e3 * 1e-9  # Grad/s
    chi = -math.tau * 278e3 * 1e-9  # Grad/s
    T2 = 18e-6 * 1e9  # ns
    gamma2 = 1 / T2
    delta = math.tau * 10e6 * 1e-9  # Grad/s

    # number of photons decays exponentially during measurement
    # n = n0 * np.exp(-kappa * t)
    # phase is integral of instantaneous frequency w = delta - 2 * chi * n
    phase = phase0 + delta * t + 2 * chi / kappa * n0 * (np.exp(-kappa * t) - 1.0)
    # decay is solution to dx/dt = -gamma * x, where gamma = gamma2 + gm * exp(-kappa * t)
    # decay = np.exp((gm * (np.exp(-kappa * t) - 1.0)) / kappa - gamma2 * t)
    # T = 512e-9 * 1e9  # ns
    # n_avg = n0 / (kappa * T) * (1.0 - np.exp(-kappa * T))
    n = n0 * np.exp(-kappa * t)
    decay = np.exp(
        -gamma2 * t
        - 16.0 * chi**2 / kappa**2 * n * (kappa / 2 * t - 1.0 + np.exp(-kappa / 2 * t))
    )
    return amplitude * decay * np.cos(phase) + offset + slope * t


def _fit_simple(x, y):
    from scipy.optimize import curve_fit

    kappa = math.tau * 463e3 * 1e-9  # Grad/s
    chi = -math.tau * 278e3 * 1e-9  # Grad/s
    T2 = 18e-6 * 1e9  # ns
    gamma2 = 1 / T2
    delta = math.tau * 10e6 * 1e-9  # Grad/s

    slope, offset = np.polyfit(x, y, 1)
    y_flat = y - np.polyval([slope, offset], x)
    # T2 = 0.5 * (np.max(x) - np.min(x))
    freqs = np.fft.rfftfreq(len(x), x[1] - x[0])
    fft = np.fft.rfft(y_flat) / len(y_flat)
    fft[0] = 0
    idx_max = np.argmax(np.abs(fft))
    frequency = freqs[idx_max]
    w = math.tau * frequency
    phase = np.angle(fft[idx_max])
    amplitude = 2 * np.abs(fft[idx_max])
    n = (delta - w) / (2 * chi)

    p0 = (
        n,
        # 0.0,
        phase,
        amplitude,
        offset,
        slope,
    )
    popt, pcov = curve_fit(
        _func,
        x,
        y,
        p0=p0,
    )
    perr = np.sqrt(np.diag(pcov))
    print(popt[0])
    return popt, perr


"""
# *************************
# *** Save data to HDF5 ***
# *************************
script_path = os.path.realpath(__file__)  # full path of current script
current_dir, script_basename = os.path.split(script_path)
script_filename = os.path.splitext(script_basename)[0]  # name of current script
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # current date and time
save_basename = f"{script_filename:s}_{timestamp:s}.h5"  # name of save file
save_path = os.path.join(current_dir, "data", save_basename)  # full path of save file
source_code = get_sourcecode(__file__)  # save also the sourcecode of the script for future reference
with h5py.File(save_path, "w") as h5f:
    dt = h5py.string_dtype(encoding='utf-8')
    ds = h5f.create_dataset("source_code", (len(source_code), ), dt)
    for ii, line in enumerate(source_code):
        ds[ii] = line
    h5f.attrs["num_averages"] = num_averages
    h5f.attrs["control_freq"] = control_freq
    h5f.attrs["control_if"] = control_if
    h5f.attrs["readout_freq"] = readout_freq
    h5f.attrs["readout_duration"] = readout_duration
    h5f.attrs["control_duration"] = control_duration
    h5f.attrs["readout_amp"] = readout_amp
    h5f.attrs["control_amp"] = control_amp
    h5f.attrs["sample_duration"] = sample_duration
    h5f.attrs["nr_delays"] = nr_delays
    h5f.attrs["dt_delays"] = dt_delays
    h5f.attrs["wait_delay"] = wait_delay
    h5f.attrs["readout_sample_delay"] = readout_sample_delay
    h5f.attrs["buffer_time"] = buffer_time
    h5f.create_dataset("first_pulse_amp_arr", data=first_pulse_amp_arr)
    h5f.create_dataset("t_arr", data=t_arr)
    h5f.create_dataset("store_arr", data=store_arr)
print(f"Data saved to: {save_path}")

# *****************
# *** Plot data ***
# *****************
# load_ramsey_single.load(save_path)
"""
if __name__ == "__main__":
    numph = NumberPhotons.load("data/number_photons_20211126_012621.h5")
    figures = numph.analyze(True)
