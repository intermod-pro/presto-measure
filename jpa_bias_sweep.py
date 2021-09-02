# -*- coding: utf-8 -*-
"""
2D sweep of DC bias and frequency of probe to find the modulation curve of the JPA.
"""
import os
import time

import h5py
import numpy as np

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import test
from presto.utils import format_sec, get_sourcecode

from mla_server import set_dc_bias
import load_jpa_bias_sweep


# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42874
EXT_CLK_REF = False

f_start = 5.6e9
f_stop = 7.0e9
df = 1e6
Navg = 1_000

bias_min = -1.25
bias_max = +1.25
nr_bias = 64
bias_arr = np.linspace(bias_min, bias_max, nr_bias)

input_port = 1
bias_port = 1
output_port = 1

amp = 0.1
dither = True
extra = 2_000
current = 32_000  # uA

with test.Test(
        address=ADDRESS,
        port=PORT,
        ext_ref_clk=EXT_CLK_REF,
        adc_mode=AdcMode.Mixed,
        adc_fsample=AdcFSample.G2,
        dac_mode=DacMode.Mixed42,
        dac_fsample=DacFSample.G10,
) as lck:
    lck.hardware.mts()
    lck.hardware.set_adc_attenuation(input_port, 0.0)
    lck.hardware.set_dac_current(output_port, current)
    lck.hardware.set_inv_sinc(output_port, 0)

    fs = lck.get_fs()
    nr_samples = int(round(fs / df))
    df = fs / nr_samples
    n_start = int(round(f_start / df))
    n_stop = int(round(f_stop / df))
    n_arr = np.arange(n_start, n_stop + 1)
    nr_freq = len(n_arr)
    freq_arr = df * n_arr

    resp_arr = np.zeros((nr_bias, nr_freq), np.complex128)

    set_dc_bias(bias_port, bias_arr[0])
    time.sleep(1.0)

    lck.hardware.set_run(False)
    lck.hardware.configure_mixer(
        freq=freq_arr[0],
        in_ports=input_port,
        out_ports=output_port,
    )
    lck.set_frequency(output_port, 0.0)
    lck.set_scale(output_port, amp, amp)
    lck.set_phase(output_port, 0.0, 0.0)
    lck.set_dither(output_port, dither)
    lck.set_dma_source(input_port)
    lck.hardware.set_run(True)

    t_start = time.time()
    prev_print_len = 0
    count = 0
    print()
    for jj, bias in enumerate(bias_arr):
        set_dc_bias(bias_port, bias)
        time.sleep(1.0)

        for ii, freq in enumerate(freq_arr):
            lck.hardware.set_run(False)
            lck.hardware.configure_mixer(
                freq=freq,
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
            resp_arr[jj, ii] = avg_i + 1j * avg_q

            count += 1
            if count % 10 == 0:
                # print estimated time left
                t_now = time.time()
                t_sofar = t_now - t_start
                nr_sofar = jj * nr_freq + ii + 1
                nr_left = (nr_bias - jj - 1) * nr_freq + (nr_freq - ii - 1)
                t_avg = t_sofar / nr_sofar
                t_left = t_avg * nr_left
                str_left = format_sec(t_left)
                msg = "Time remaining: {:s}".format(str_left)
                print_len = len(msg)
                if print_len < prev_print_len:
                    msg += " " * (prev_print_len - print_len)
                print(msg, end="\r", flush=True)
                prev_print_len = print_len

    print(f"Measurement completed in: {format_sec(time.time()-t_start):s}")
    # Mute outputs at the end of the sweep
    lck.hardware.set_run(False)
    lck.set_scale(output_port, 0.0, 0.0)
    set_dc_bias(bias_port, 0.0)

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
    h5f.attrs["Navg"] = Navg
    h5f.attrs["amp"] = amp
    h5f.attrs["dither"] = dither
    h5f.attrs["input_port"] = input_port
    h5f.attrs["output_port"] = output_port
    h5f.attrs["bias_port"] = bias_port
    h5f.create_dataset("freq_arr", data=freq_arr)
    h5f.create_dataset("bias_arr", data=bias_arr)
    h5f.create_dataset("resp_arr", data=resp_arr)
print(f"Data saved to: {save_path}")

# ********************
# *** Plot results ***
# ********************
fig1 = load_jpa_bias_sweep.load(save_path)
