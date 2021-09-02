# -*- coding: utf-8 -*-
"""
Simple frequency sweep using the Test mode.
"""
import os
import time

import h5py
import numpy as np

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import test
from presto.utils import format_sec, get_sourcecode

import load_sweep_nco_time

# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42874
EXT_CLK_REF = False

output_port = 1
input_port = 1

amp = 1.0  # FS
phase = 0.0
dither = True

extra = 500
f_center = 6.0284 * 1e9  # resonator 1
# f_center = 6.1666 * 1e9  # resonator 2
f_span = 5 * 1e6
f_start = f_center - f_span / 2
f_stop = f_center + f_span / 2
df = 1e4  # Hz
Navg = 100

with test.Test(
        address=ADDRESS,
        port=PORT,
        ext_ref_clk=EXT_CLK_REF,
        adc_mode=AdcMode.Mixed,
        adc_fsample=AdcFSample.G2,
        dac_mode=DacMode.Mixed42,
        dac_fsample=DacFSample.G10,
) as lck:
    lck.hardware.set_adc_attenuation(input_port, 0.0)
    # lck.hardware.set_dac_current(output_port, 6_425)
    lck.hardware.set_dac_current(output_port, 32_000)
    lck.hardware.set_inv_sinc(output_port, 0)

    fs = lck.get_fs()
    nr_samples = int(round(fs / df))
    df = fs / nr_samples

    n_start = int(round(f_start / df))
    n_stop = int(round(f_stop / df))
    n_arr = np.arange(n_start, n_stop + 1)
    nr_freq = len(n_arr)
    freq_arr = df * n_arr
    resp_arr = np.zeros(nr_freq, np.complex128)

    lck.hardware.set_run(False)
    lck.hardware.configure_mixer(
        freq=freq_arr[0],
        in_ports=input_port,
        out_ports=output_port,
    )
    lck.set_frequency(output_port, 0.0)
    lck.set_scale(output_port, amp, amp)
    lck.set_phase(output_port, phase, phase)
    lck.set_dither(output_port, dither)
    lck.set_dma_source(input_port)
    lck.hardware.set_run(True)

    t_start = time.time()
    t_last = t_start
    prev_print_len = 0
    print()
    for ii in range(len(n_arr)):
        # lck.hardware.sleep(1e-1, False)
        f = freq_arr[ii]

        lck.hardware.set_run(False)
        lck.hardware.configure_mixer(
            freq=f,
            in_ports=input_port,
            out_ports=output_port,
        )
        lck.hardware.sleep(1e-3, False)
        lck.start_dma(Navg * nr_samples + extra)
        lck.hardware.set_run(True)
        lck.wait_for_dma()
        lck.stop_dma()

        _data = lck.get_dma_data(Navg * nr_samples + extra)
        data_i = _data[0::2][-Navg * nr_samples:] / 32767
        data_q = _data[1::2][-Navg * nr_samples:] / 32767

        data_i.shape = (Navg, nr_samples)
        data_q.shape = (Navg, nr_samples)
        data_i = np.mean(data_i, axis=0)
        data_q = np.mean(data_q, axis=0)

        avg_i = np.mean(data_i)
        avg_q = np.mean(data_q)
        resp_arr[ii] = avg_i + 1j * avg_q

        # Calculate and print remaining time
        t_now = time.time()
        if t_now - t_last > np.pi / 3 / 5:
            t_last = t_now
            t_sofar = t_now - t_start
            nr_sofar = ii + 1
            nr_left = nr_freq - ii - 1
            t_avg = t_sofar / nr_sofar
            t_left = t_avg * nr_left
            str_left = format_sec(t_left)
            msg = "Time remaining: {:s}".format(str_left)
            print_len = len(msg)
            if print_len < prev_print_len:
                msg += " " * (prev_print_len - print_len)
            print(msg, end="\r", flush=True)
            prev_print_len = print_len

    # Mute outputs at the end of the sweep
    lck.hardware.set_run(False)
    lck.set_scale(output_port, 0.0, 0.0)


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
    h5f.attrs["df"] = df
    h5f.attrs["amp"] = amp
    h5f.attrs["phase"] = phase
    h5f.attrs["dither"] = dither
    h5f.attrs["input_port"] = input_port
    h5f.attrs["output_port"] = output_port
    h5f.create_dataset("freq_arr", data=freq_arr)
    h5f.create_dataset("resp_arr", data=resp_arr)
print(f"Data saved to: {save_path}")

# *****************
# *** Plot data ***
# *****************
fig1, span_a, span_p = load_sweep_nco_time.load(save_path)
