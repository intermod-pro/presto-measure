# -*- coding: utf-8 -*-
"""
2D sweep of DC bias and frequency of probe to find the modulation curve of the JPA.
Copyright (C) 2021  Intermodulation Products AB.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
<https://www.gnu.org/licenses/>.
"""
import os
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from presto import commands as cmd
from presto import test
from presto.utils import format_sec, get_sourcecode

if '/home/riccardo/IntermodulatorSuite' not in sys.path:
    sys.path.append('/home/riccardo/IntermodulatorSuite')
from mlaapi import mla_api, mla_globals

settings = mla_globals.read_config()
mla = mla_api.MLA(settings)

# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42878
EXT_CLK_REF = False

f_start = 5.6e9
f_stop = 7.0e9
df = 1e6
Navg = 100

bias_min = -1.25
bias_max = +1.25
nr_bias = 64
bias_arr = np.linspace(bias_min, bias_max, nr_bias)

input_port = 1
bias_port = 1
output_port = [1, 9]

amp = 0.1
dither = True
extra = 2_000

mla.connect()
with test.Test(
        address=ADDRESS,
        port=PORT,
        ext_ref_clk=EXT_CLK_REF,
        adc_mode=cmd.AdcMixed,
        adc_fsample=cmd.AdcG2,
        dac_mode=cmd.DacMixed42,
        dac_fsample=cmd.DacG10,
) as lck:
    lck.hardware.mts()

    fs = lck.get_fs()
    nr_samples = int(round(fs / df))
    df = fs / nr_samples
    n_start = int(round(f_start / df))
    n_stop = int(round(f_stop / df))
    n_arr = np.arange(n_start, n_stop + 1)
    nr_freq = len(n_arr)
    freq_arr = df * n_arr

    resp_arr = np.zeros((nr_bias, nr_freq), np.complex128)

    mla.lockin.set_dc_offset(bias_port, bias_arr[0])
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
        mla.lockin.set_dc_offset(bias_port, bias)
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
            data_i = _data[0::2][-Navg * nr_samples:]
            data_q = _data[1::2][-Navg * nr_samples:]

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
    mla.lockin.set_dc_offset(bias_port, 0.0)
mla.disconnect()

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

# *****************
# *** Plot data ***
# *****************
resp_db = 20 * np.log10(np.abs(resp_arr))
resp_phase = np.unwrap(np.angle(resp_arr), axis=0)
dw = 2 * np.pi * df
resp_grpdly = np.diff(resp_phase, axis=0) / dw
data_plot = resp_grpdly * 1e9

vmin = np.percentile(data_plot, 5)
vmax = np.percentile(data_plot, 95)

fig, ax = plt.subplots(tight_layout=True)
im = ax.imshow(
    data_plot,
    origin='lower',
    aspect='auto',
    extent=(1e-9 * freq_arr[0], 1e-9 * freq_arr[-1], bias_arr[0], bias_arr[-1]),
    vmin=vmin,
    vmax=vmax,
)
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Bias [FS]')
fig.colorbar(im)
fig.show()
