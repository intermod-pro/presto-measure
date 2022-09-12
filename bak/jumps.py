# -*- coding: utf-8 -*-
"""
Monitor the amplitude of a driven cavity and plot histogram, to see quantum jumps using the Test mode.
"""
import os
import sys
import time

import h5py
import numpy as np
from presto import commands as cmd
from presto import test
from presto.utils import get_sourcecode

import load_jumps

JPA = True
if JPA:
    if "/home/riccardo/IntermodulatorSuite" not in sys.path:
        sys.path.append("/home/riccardo/IntermodulatorSuite")
    from mlaapi import mla_api, mla_globals

    settings = mla_globals.read_config()
    mla = mla_api.MLA(settings)

# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42878
EXT_REF_CLK = False  # set to True to lock to an external reference clock

output_port = [1, 9]
input_port = 1

amp = 0.003
phase = 0.0
dither = True
# current = 6_425
# current = 20_000
current = 32_000

extra = 500
freq = 6.029035e9  # Hz
df = 1e5  # Hz
Nruns = 10_000 * 5

if JPA:
    jpa_pump_freq = 2 * 6.031e9  # Hz
    jpa_pump_pwr = 7  # lmx units
    jpa_bias = +0.432  # V
    bias_port = 1
    mla.connect()

with test.Test(
    address=ADDRESS,
    port=PORT,
    ext_ref_clk=EXT_REF_CLK,
    adc_mode=cmd.AdcMixed,
    adc_fsample=cmd.AdcG2,
    dac_mode=cmd.DacMixed42,
    dac_fsample=cmd.DacG10,
) as lck:
    lck.hardware.set_adc_attenuation(input_port, 0.0)
    lck.hardware.set_dac_current(output_port, current)
    lck.hardware.set_inv_sinc(output_port, 0)

    if JPA:
        lck.hardware.set_lmx(jpa_pump_freq, jpa_pump_pwr)
        mla.lockin.set_dc_offset(bias_port, jpa_bias)
    else:
        lck.hardware.set_lmx(0.0, 0)
    time.sleep(1.0)

    fs = lck.get_fs()
    nr_samples = int(round(fs / df))
    df = fs / nr_samples
    t_arr = np.arange(Nruns) / df
    resp_arr = np.zeros(Nruns, np.complex128)

    lck.hardware.set_run(False)
    lck.hardware.configure_mixer(
        freq=freq,
        in_ports=input_port,
        out_ports=output_port,
    )
    lck.set_frequency(output_port, 0.0)
    lck.set_scale(output_port, amp, amp)
    lck.set_phase(output_port, phase, phase)
    lck.set_dither(output_port, dither)
    lck.set_dma_source(input_port)
    lck.hardware.sleep(1e-3, False)

    lck.start_dma(Nruns * nr_samples + extra)
    lck.hardware.set_run(True)
    lck.wait_for_dma()
    lck.stop_dma()

    all_data = lck.get_dma_data(Nruns * nr_samples + extra)
    lck.hardware.set_run(False)

    for ii in range(Nruns):
        print(f"{ii} / {Nruns}")
        _data = all_data[ii * nr_samples * 2 + 2 * extra : (ii + 1) * nr_samples * 2 + 2 * extra]
        data_i = _data[0::2][-nr_samples:] / 32767
        data_q = _data[1::2][-nr_samples:] / 32767

        avg_i = np.mean(data_i)
        avg_q = np.mean(data_q)
        resp_arr[ii] = avg_i + 1j * avg_q

    # Mute outputs at the end of the sweep
    lck.set_scale(output_port, 0.0, 0.0)
    lck.hardware.set_lmx(0.0, 0)
if JPA:
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
source_code = get_sourcecode(
    __file__
)  # save also the sourcecode of the script for future reference
with h5py.File(save_path, "w") as h5f:
    dt = h5py.string_dtype(encoding="utf-8")
    ds = h5f.create_dataset("source_code", (len(source_code),), dt)
    for ii, line in enumerate(source_code):
        ds[ii] = line
    h5f.attrs["df"] = df
    h5f.attrs["dither"] = dither
    h5f.attrs["input_port"] = input_port
    h5f.attrs["output_port"] = output_port
    h5f.attrs["amp"] = amp
    h5f.attrs["freq"] = freq
    h5f.create_dataset("t_arr", data=t_arr)
    h5f.create_dataset("resp_arr", data=resp_arr)
print(f"Data saved to: {save_path}")

# ********************
# *** Plot results ***
# ********************
fig = load_jumps.load(save_path)
