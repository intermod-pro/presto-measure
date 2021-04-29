# -*- coding: utf-8 -*-
"""
2D sweep of drive power and frequency with the Test mode.
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
import time

import h5py
import numpy as np
from presto import commands as cmd
from presto import test
from presto.utils import format_sec, get_sourcecode

import load_sweep_power

# Presto's IP address or hostname
ADDRESS = "192.0.2.53"
EXT_REF_CLK = False  # set to True to lock to an external reference clock

# center_freq = 6.213 * 1e9  # Hz, center frequency for sweep, resonator 1
center_freq = 6.376 * 1e9  # Hz, center frequency for sweep, resonator 2

span = 20e6  # Hz, span for frequency sweep
df = 1e4  # Hz, measurement bandwidth for each point in sweep
Navg = 3

output_port = 1
input_port = 1
dither = False
extra = 500
current = 32_000  # uA
# current = 20_000  # uA
# current = 6_425  # uA

amp_arr = np.logspace(-10, 0, 64, base=2, endpoint=False)
nr_amps = len(amp_arr)

with test.Test(
        address=ADDRESS,
        ext_ref_clk=EXT_REF_CLK,
        adc_mode=cmd.AdcMixed,
        adc_fsample=cmd.AdcG2,
        dac_mode=cmd.DacMixed42,
        dac_fsample=cmd.DacG10,
) as lck:
    lck.hardware.mts()
    lck.hardware.set_adc_attenuation(input_port, 0.0)
    lck.hardware.set_dac_current(output_port, current)
    lck.hardware.set_inv_sinc(output_port, 0)

    fs = lck.get_fs()
    nr_samples = int(round(fs / df))
    df = fs / nr_samples

    n_start = int(round((center_freq - span / 2) / df))
    n_stop = int(round((center_freq + span / 2) / df))
    n_arr = np.arange(n_start, n_stop + 1)
    nr_freq = len(n_arr)
    freq_arr = df * n_arr
    resp_arr = np.zeros((nr_amps, nr_freq), np.complex128)

    lck.hardware.set_run(False)
    lck.set_frequency(output_port, 0.0)
    lck.set_scale(output_port, amp_arr[0], amp_arr[0])
    lck.set_phase(output_port, 0.0, 0.0)
    lck.set_dither(output_port, dither)
    lck.set_dma_source(input_port)
    lck.hardware.set_run(True)

    t_start = time.time()
    prev_print_len = 0
    count = 0
    print()
    for jj, amp in enumerate(amp_arr):
        for ii, freq in enumerate(freq_arr):
            lck.hardware.set_run(False)
            lck.hardware.configure_mixer(
                freq=freq,
                in_ports=input_port,
                out_ports=output_port,
            )
            lck.set_scale(output_port, amp, amp)
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

            # Calculate and print remaining time
            count += 1
            if count % 10 == 0:
                # print estimated time left
                t_now = time.time()
                t_sofar = t_now - t_start
                nr_sofar = jj * nr_freq + ii + 1
                nr_left = (nr_amps - jj - 1) * nr_freq + (nr_freq - ii - 1)
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
    h5f.attrs["dither"] = dither
    h5f.attrs["input_port"] = input_port
    h5f.attrs["output_port"] = output_port
    h5f.create_dataset("freq_arr", data=freq_arr)
    h5f.create_dataset("amp_arr", data=amp_arr)
    h5f.create_dataset("resp_arr", data=resp_arr)
print(f"Data saved to: {save_path}")

# ********************
# *** Plot results ***
# ********************
fig1 = load_sweep_power.load(save_path)
