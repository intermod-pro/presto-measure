"""Power-dependent single-frequency 2D sweep using the test firmware (measure in time domain).
"""
import os
import sys
import time

import h5py
import numpy as np

import commands as cmd
import rflockin
from utils import get_sourcecode, format_sec
import load_sweep_power

JPA = False
if JPA:
    if '/home/riccardo/IntermodulatorSuite' not in sys.path:
        sys.path.append('/home/riccardo/IntermodulatorSuite')
    from mlaapi import mla_api
    from mlaapi import mla_globals
    settings = mla_globals.read_config()
    mla = mla_api.MLA(settings)

# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42878
EXT_REF_CLK = False  # set to True to lock to an external reference clock

center_freq = 6.027e9  # Hz, center frequency for sweep

span = 10e6  # Hz, span for frequency sweep
df = 1e4  # Hz, measurement bandwidth for each point in sweep
Navg = 1

output_port = [1, 9]
input_port = 1
dither = False
extra = 500
current = 32_000  # uA
# current = 20_000  # uA
# current = 6_425  # uA

amp_arr = np.logspace(-10, 0, 64, base=2, endpoint=False)
nr_amps = len(amp_arr)

if JPA:
    jpa_pump_freq = 12_058e6  # Hz
    jpa_pump_pwr = 0  # lmx units
    jpa_bias = +0.455  # V
    bias_port = 1
    mla.connect()

with rflockin.Test(
        address=ADDRESS,
        port=PORT,
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

    if JPA:
        lck.hardware.set_lmx(jpa_pump_freq, jpa_pump_pwr)
        mla.lockin.set_dc_offset(bias_port, jpa_bias)
        time.sleep(1.0)
    else:
        lck.hardware.set_lmx(0.0, 0)

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
    lck.hardware.set_lmx(0.0, 0)

if JPA:
    mla.lockin.set_dc_offset(bias_port, 0.0)
    mla.disconnect()

# *************************
# *** Save data to HDF5 ***
# *************************
script_filename = os.path.splitext(os.path.basename(__file__))[0]  # name of current script
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # current date and time
save_filename = f"{script_filename:s}_{timestamp:s}.h5"  # name of save file
source_code = get_sourcecode(__file__)  # save also the sourcecode of the script for future reference
with h5py.File(save_filename, "w") as h5f:
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

# ********************
# *** Plot results ***
# ********************
fig1 = load_sweep_power.load(save_filename)
