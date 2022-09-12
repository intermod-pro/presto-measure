# -*- coding: utf-8 -*-
import os

# https://github.com/ContinuumIO/anaconda-issues/issues/905#issuecomment-232498034
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
import signal
import time

import h5py
from matplotlib import _pylab_helpers
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.utils import get_sourcecode, rotate_opt, sin2

# Presto's IP address or hostname
ADDRESS = "192.0.2.53"
EXT_REF_CLK = False  # set to True to lock to an external reference clock

# cavity drive: readout
readout_freq_1 = 6.213095 * 1e9  # Hz, resonator 1
readout_freq_2 = 6.376650 * 1e9  # Hz, resonator 2
readout_amp = 10 ** (-10.0 / 20)  # FS
readout_duration = 2e-6  # s, duration of the readout pulse
readout_port = 1

# qubit drive: control
control_freq_1 = 4.146391 * 1e9  # Hz <-- from data/ramsey_20210428_041929.h5, qubit 1
control_freq_2 = 4.776805 * 1e9  # Hz <-- from data/ramsey_20210428_160241.h5, qubit 2
control_amp_pi_1 = 0.1594  # FS <-- pi pulse from data/rabi_amp_20210428_101944.h5, qubit 1
control_amp_pi_2 = 0.1188  # FS <-- pi pulse from data/rabi_amp_20210428_164251.h5, qubit 2
control_amp_pidiv2_1 = 0.07968  # FS <-- pi/2 pulse from data/rabi_amp_20210428_101944.h5, qubit 1
control_amp_pidiv2_2 = 0.05941  # FS <-- pi/2 pulse from data/rabi_amp_20210428_164251.h5, qubit 2
control_duration = 100 * 1e-9  # s, duration of the control pulse
control_port_1 = 5  # qubit 1
control_port_2 = 7  # qubit 2

# cavity readout: sample
sample_duration = 4 * 1e-6  # s, duration of the sampling window
sample_port = 1

# other
num_averages = 1_000
wait_delay = 500e-6  # s, delay between repetitions to allow the qubit to decay
readout_sample_delay = (
    300 * 1e-9
)  # s, delay between readout pulse and sample window to account for latency
t_low = 1500 * 1e-9
t_high = 2000 * 1e-9


def measure_t1(which_qubit):
    if which_qubit == 1:
        readout_freq = readout_freq_1
        control_freq = control_freq_1
        control_amp = control_amp_pi_1
        control_port = control_port_1
    else:
        readout_freq = readout_freq_2
        control_freq = control_freq_2
        control_amp = control_amp_pi_2
        control_port = control_port_2

    nr_delays = 128  # number of steps when changing delay between control and readout pulses
    dt_delays = 4 * 1e-6  # s, step size when changing delay between control and readout pulses

    with pulsed.Pulsed(
        address=ADDRESS,
        ext_ref_clk=EXT_REF_CLK,
        adc_mode=AdcMode.Mixed,
        adc_fsample=AdcFSample.G2,
        dac_mode=[DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample=[DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
    ) as pls:
        pls.hardware.set_adc_attenuation(sample_port, 0.0)
        pls.hardware.set_dac_current(readout_port, 32_000)
        pls.hardware.set_dac_current(control_port, 32_000)
        pls.hardware.set_inv_sinc(readout_port, 0)
        pls.hardware.set_inv_sinc(control_port, 0)
        pls.hardware.configure_mixer(
            freq=readout_freq,
            in_ports=sample_port,
            out_ports=readout_port,
            sync=False,  # sync in next call
        )
        pls.hardware.configure_mixer(
            freq=control_freq,
            out_ports=control_port,
            sync=True,  # sync here
        )

        # ************************************
        # *** Setup measurement parameters ***
        # ************************************

        # Setup lookup tables for frequencies
        # we only need to use carrier 1
        pls.setup_freq_lut(
            output_ports=readout_port,
            group=0,
            frequencies=0.0,
            phases=0.0,
            phases_q=0.0,
        )
        pls.setup_freq_lut(
            output_ports=control_port,
            group=0,
            frequencies=0.0,
            phases=0.0,
            phases_q=0.0,
        )

        # Setup lookup tables for amplitudes
        pls.setup_scale_lut(
            output_ports=readout_port,
            group=0,
            scales=readout_amp,
        )
        pls.setup_scale_lut(
            output_ports=control_port,
            group=0,
            scales=control_amp,
        )

        # Setup readout and control pulses
        # use setup_long_drive to create a pulse with square envelope
        # setup_long_drive supports smooth rise and fall transitions for the pulse,
        # but we keep it simple here
        readout_pulse = pls.setup_long_drive(
            output_port=readout_port,
            group=0,
            duration=readout_duration,
            amplitude=1.0,
            amplitude_q=1.0,
            rise_time=0e-9,
            fall_time=0e-9,
        )
        control_ns = int(
            round(control_duration * pls.get_fs("dac"))
        )  # number of samples in the control template
        control_envelope = sin2(control_ns)
        control_pulse = pls.setup_template(
            output_port=control_port,
            group=0,
            template=control_envelope,
            template_q=control_envelope,
            envelope=True,
        )

        # Setup sampling window
        pls.set_store_ports(sample_port)
        pls.set_store_duration(sample_duration)

        # ******************************
        # *** Program pulse sequence ***
        # ******************************
        T = 0.0  # s, start at time zero ...
        for ii in range(nr_delays):
            # pi pulse
            pls.reset_phase(T, control_port)
            pls.output_pulse(T, control_pulse)
            # Readout pulse starts after control pulse,
            # with an increasing delay
            T += control_duration + ii * dt_delays
            pls.reset_phase(T, readout_port)
            pls.output_pulse(T, readout_pulse)
            # Sampling window
            pls.store(T + readout_sample_delay)
            # Move to next iteration
            T += readout_duration
            T += wait_delay

        # **************************
        # *** Run the experiment ***
        # **************************
        pls.run(
            period=T,
            repeat_count=1,
            num_averages=num_averages,
            print_time=True,
        )
        t_arr, (data_I, data_Q) = pls.get_store_data()

    store_arr = data_I + 1j * data_Q

    idx_low = np.argmin(np.abs(t_arr - t_low))
    idx_high = np.argmin(np.abs(t_arr - t_high))
    idx = np.arange(idx_low, idx_high)

    # Analyze T1
    resp_arr = np.mean(store_arr[:, 0, idx], axis=-1)
    resp_arr = rotate_opt(resp_arr)
    delay_arr = dt_delays * np.arange(nr_delays)

    # Fit data
    popt_x, perr_x = t1_fit_simple(delay_arr, np.real(resp_arr))

    T1 = popt_x[0]
    T1_err = perr_x[0]

    return T1, T1_err


def measure_t2(which_qubit):
    if which_qubit == 1:
        readout_freq = readout_freq_1
        control_freq = control_freq_1
        control_amp = control_amp_pidiv2_1
        control_port = control_port_1
    else:
        readout_freq = readout_freq_2
        control_freq = control_freq_2
        control_amp = control_amp_pidiv2_2
        control_port = control_port_2

    control_if = 0 * 1e6  # Hz
    nr_delays = 256  # number of steps when changing delay between control and readout pulses
    dt_delays = 1 * 1e-6  # s, step size when changing delay between control and readout pulses

    # Instantiate interface class
    with pulsed.Pulsed(
        address=ADDRESS,
        ext_ref_clk=EXT_REF_CLK,
        adc_mode=AdcMode.Mixed,
        adc_fsample=AdcFSample.G2,
        dac_mode=[DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample=[DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
    ) as pls:
        pls.hardware.set_adc_attenuation(sample_port, 0.0)
        pls.hardware.set_dac_current(readout_port, 32_000)
        pls.hardware.set_dac_current(control_port, 32_000)
        pls.hardware.set_inv_sinc(readout_port, 0)
        pls.hardware.set_inv_sinc(control_port, 0)
        pls.hardware.configure_mixer(
            freq=readout_freq,
            in_ports=sample_port,
            out_ports=readout_port,
            sync=False,  # sync in next call
        )
        pls.hardware.configure_mixer(
            freq=control_freq - control_if,
            out_ports=control_port,
            sync=True,  # sync here
        )

        # ************************************
        # *** Setup measurement parameters ***
        # ************************************

        # Setup lookup tables for frequencies
        # we only need to use carrier 1
        pls.setup_freq_lut(
            output_ports=readout_port,
            group=0,
            frequencies=0.0,
            phases=0.0,
            phases_q=0.0,
        )
        pls.setup_freq_lut(
            output_ports=control_port,
            group=0,
            frequencies=control_if,
            phases=0.0,
            phases_q=0.0 if control_if == 0.0 else -np.pi / 2,
        )

        # Setup lookup tables for amplitudes
        pls.setup_scale_lut(
            output_ports=readout_port,
            group=0,
            scales=readout_amp,
        )
        pls.setup_scale_lut(
            output_ports=control_port,
            group=0,
            scales=control_amp,
        )

        # Setup readout and control pulses
        # use setup_long_drive to create a pulse with square envelope
        # setup_long_drive supports smooth rise and fall transitions for the pulse,
        # but we keep it simple here
        readout_pulse = pls.setup_long_drive(
            output_port=readout_port,
            group=0,
            duration=readout_duration,
            amplitude=1.0,
            amplitude_q=1.0,
            rise_time=0e-9,
            fall_time=0e-9,
        )
        control_ns = int(
            round(control_duration * pls.get_fs("dac"))
        )  # number of samples in the control template
        control_envelope = sin2(control_ns)
        control_pulse = pls.setup_template(
            output_port=control_port,
            group=0,
            template=control_envelope,
            template_q=control_envelope,
            envelope=True,
        )

        # Setup sampling window
        pls.set_store_ports(sample_port)
        pls.set_store_duration(sample_duration)

        # ******************************
        # *** Program pulse sequence ***
        # ******************************
        T = 0.0  # s, start at time zero ...
        for ii in range(nr_delays):
            # first pi/2 pulse
            pls.reset_phase(T, control_port)
            pls.output_pulse(T, control_pulse)
            # second pi/2 pulse
            T += control_duration + ii * dt_delays
            pls.output_pulse(T, control_pulse)
            # Readout pulse starts after control pulse,
            # with an increasing delay
            T += control_duration
            pls.reset_phase(T, readout_port)
            pls.output_pulse(T, readout_pulse)
            # Sampling window
            pls.store(T + readout_sample_delay)
            # Move to next iteration
            T += readout_duration
            T += wait_delay

        # **************************
        # *** Run the experiment ***
        # **************************
        pls.run(
            period=T,
            repeat_count=1,
            num_averages=num_averages,
            print_time=True,
        )
        t_arr, (data_I, data_Q) = pls.get_store_data()

    store_arr = data_I + 1j * data_Q

    idx_low = np.argmin(np.abs(t_arr - t_low))
    idx_high = np.argmin(np.abs(t_arr - t_high))
    idx = np.arange(idx_low, idx_high)

    # Analyze T2
    resp_arr = np.mean(store_arr[:, 0, idx], axis=-1)
    data = rotate_opt(resp_arr)
    delay_arr = dt_delays * np.arange(nr_delays)

    # Fit data
    popt_x, perr_x = t2_fit_simple(delay_arr, np.real(data))

    T2 = popt_x[2]
    T2_err = perr_x[2]
    # det = popt_x[3]
    # det_err = perr_x[3]

    return T2, T2_err


def t1_decay(t, *p):
    T1, xe, xg = p
    return xg + (xe - xg) * np.exp(-t / T1)


def t1_fit_simple(t, x):
    T1 = 0.5 * (t[-1] - t[0])
    xe, xg = x[0], x[-1]
    p0 = (T1, xe, xg)
    popt, pcov = curve_fit(t1_decay, t, x, p0, maxfev=12_000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def t2_func(t, offset, amplitude, T2, frequency, phase):
    return offset + amplitude * np.exp(-t / T2) * np.cos(2.0 * np.pi * frequency * t + phase)


def t2_fit_simple(x, y):
    pkpk = np.max(y) - np.min(y)
    offset = np.min(y) + pkpk / 2
    amplitude = 0.5 * pkpk
    T2 = 0.5 * (np.max(x) - np.min(x))
    freqs = np.fft.rfftfreq(len(x), x[1] - x[0])
    fft = np.fft.rfft(y)
    frequency = freqs[1 + np.argmax(np.abs(fft[1:]))]
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
        frequency,
        phase,
    )
    popt, pcov = curve_fit(
        t2_func,
        x,
        y,
        p0=p0,
        maxfev=12_000,
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def my_pause(interval=0.1):
    manager = _pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
        # plt.show(block=False)
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)


def handler(signum, frame):
    global KEEP_GOING
    if KEEP_GOING:
        print("\n\n")
        print("Ctrl-C pressed!")
        print("Will finish this run and then stop.")
        print("Press Ctrl-C again to abort.")
        print("\n\n")
        KEEP_GOING = False
    else:
        raise KeyboardInterrupt


if __name__ == "__main__":
    fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
    ax1, ax2 = ax

    time_arr = np.array([])
    rel_time_arr = np.array([])
    t1_arr_q1 = np.array([])
    t1_arr_q2 = np.array([])
    (line_t1_q1,) = ax1.plot(rel_time_arr, t1_arr_q1, ".", c="tab:blue")
    (line_t1_q2,) = ax1.plot(rel_time_arr, t1_arr_q2, ".", c="tab:orange")
    ax1.set_ylabel("T1 [us]")

    t2_arr_q1 = np.array([])
    t2_arr_q2 = np.array([])
    (line_t2_q1,) = ax2.plot(rel_time_arr, t2_arr_q1, ".", c="tab:blue")
    (line_t2_q2,) = ax2.plot(rel_time_arr, t2_arr_q2, ".", c="tab:orange")
    ax2.set_ylabel("T2* [us]")

    ax2.set_xlabel("Time since start [s]")

    fig.show()
    my_pause()

    KEEP_GOING = True
    signal.signal(signal.SIGINT, handler)
    count = 0
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # current date and time
    source_code = get_sourcecode(
        __file__
    )  # save also the sourcecode of the script for future reference
    while KEEP_GOING:
        print("\n\n\n")
        print(f"******* Run number {count+1:d} *******")
        count += 1

        time_arr = np.r_[time_arr, time.time()]
        rel_time_arr = time_arr - time_arr[0]

        for which_qubit in range(1, 3):
            print("\n")
            print(f"------- measure T1 on qubit {which_qubit:d} -------")
            try:
                t1, _ = measure_t1(which_qubit)
            except:
                t1 = np.nan
            if which_qubit == 1:
                t1_arr_q1 = np.r_[t1_arr_q1, t1]
                line_t1_q1.set_data(rel_time_arr, 1e6 * t1_arr_q1)
            else:
                t1_arr_q2 = np.r_[t1_arr_q2, t1]
                line_t1_q2.set_data(rel_time_arr, 1e6 * t1_arr_q2)
            ax1.relim()
            ax1.autoscale()
            my_pause()

            print("\n")
            print(f"------- measure T2 on qubit {which_qubit:d} -------")
            try:
                t2, _ = measure_t2(which_qubit)
            except:
                t2 = np.nan
            if which_qubit == 1:
                t2_arr_q1 = np.r_[t2_arr_q1, t2]
                line_t2_q1.set_data(rel_time_arr, 1e6 * t2_arr_q1)
            else:
                t2_arr_q2 = np.r_[t2_arr_q2, t2]
                line_t2_q2.set_data(rel_time_arr, 1e6 * t2_arr_q2)
            ax2.relim()
            ax2.autoscale()
            my_pause()

        # *** Save ***
        script_path = os.path.realpath(__file__)  # full path of current script
        current_dir, script_basename = os.path.split(script_path)
        script_filename = os.path.splitext(script_basename)[0]  # name of current script
        save_basename = f"{script_filename:s}_{timestamp:s}.h5"  # name of save file
        save_path = os.path.join(current_dir, "data", save_basename)  # full path of save file
        with h5py.File(save_path, "w") as h5f:
            dt = h5py.string_dtype(encoding="utf-8")
            ds = h5f.create_dataset("source_code", (len(source_code),), dt)
            for ii, line in enumerate(source_code):
                ds[ii] = line
            h5f.create_dataset("rel_time_arr", data=rel_time_arr)
            h5f.create_dataset("time_arr", data=time_arr)
            h5f.create_dataset("t1_arr_q1", data=t1_arr_q1)
            h5f.create_dataset("t1_arr_q2", data=t1_arr_q2)
            h5f.create_dataset("t2_arr_q1", data=t2_arr_q1)
            h5f.create_dataset("t2_arr_q2", data=t2_arr_q2)
        print(f"Data saved to: {save_path}")

    print("\n\n\n")
    print("Done")
    input("___ Press Enter to close ___")
