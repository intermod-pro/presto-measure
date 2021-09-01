# -*- coding: utf-8 -*-
"""
Two-tone spectroscopy with Test mode: 2D sweep of pump power and frequency, with fixed probe.
"""
import os
import time

import h5py
import numpy as np

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import test
from presto.utils import format_sec, get_sourcecode

# import load_two_tone_power

WHICH_QUBIT = 1  # 1 or 2

# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42874
EXT_REF_CLK = False  # set to True to lock to an external reference clock
USE_JPA = True

if WHICH_QUBIT == 1:
    center_freq = 3.437 * 1e9  # Hz, center frequency for qubit sweep, qubit 1
    cavity_freq = 6.166_600 * 1e9  # Hz, frequency for cavity, resonator 1
    qubit_port = 3  # qubit 1
elif WHICH_QUBIT == 2:
    center_freq = 3.975 * 1e9  # Hz, center frequency for qubit sweep, qubit 2
    cavity_freq = 6.028_448 * 1e9  # Hz, frequency for cavity, resonator 2
    qubit_port = 4  # qubit 2
else:
    raise ValueError

span = 350 * 1e6  # Hz, span for qubit frequency sweep
df = 1e6  # Hz, measurement bandwidth for each point in sweep

cavity_amp = 10**(-20.0 / 20)  # FS

nr_amps = 61
qubit_amp_arr = np.logspace(-3, 0, nr_amps)

cavity_port = 1
input_port = 1
dither = True
extra = 500
Navg = 1_000

if USE_JPA:
    jpa_bias_port = 1
    if WHICH_QUBIT == 1:
        jpa_bias = +0.437  # V
        jpa_pump_pwr = 11
        jpa_pump_freq = 2 * 6.169 * 1e9  # 2.5 MHz away from resonator 1
    elif WHICH_QUBIT == 2:
        jpa_bias = +0.449  # V
        jpa_pump_pwr = 9
        jpa_pump_freq = 2 * 6.031 * 1e9  # 2.5 MHz away from resonator 2
    else:
        raise ValueError

    import sys
    if '/home/riccardo/IntermodulatorSuite' not in sys.path:
        sys.path.append('/home/riccardo/IntermodulatorSuite')
    from mlaapi import mla_api, mla_globals

    settings = mla_globals.read_config()
    mla = mla_api.MLA(settings)
    mla.connect()
    mla.lockin.set_dc_offset(jpa_bias_port, jpa_bias)
    time.sleep(1.0)

with test.Test(
        address=ADDRESS,
        port=PORT,
        ext_ref_clk=EXT_REF_CLK,
        reset=True,
        adc_mode=AdcMode.Mixed,
        adc_fsample=AdcFSample.G2,
        dac_mode=[
            DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02
        ],
        dac_fsample=[
            DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6
        ],
) as lck:
    lck.hardware.set_adc_attenuation(input_port, 0.0)
    lck.hardware.set_dac_current(cavity_port, 32_000)
    lck.hardware.set_dac_current(qubit_port, 32_000)
    lck.hardware.set_inv_sinc(cavity_port, 0)
    lck.hardware.set_inv_sinc(qubit_port, 0)
    if USE_JPA:
        lck.hardware.set_lmx(jpa_pump_freq, jpa_pump_pwr)

    fs = lck.get_fs()
    nr_samples = int(round(fs / df))
    df = fs / nr_samples

    n_start = int(round((center_freq - span / 2) / df))
    n_stop = int(round((center_freq + span / 2) / df))
    n_arr = np.arange(n_start, n_stop + 1)
    nr_freq = len(n_arr)
    qubit_freq_arr = df * n_arr
    resp_arr = np.zeros((nr_amps, nr_freq), np.complex128)

    lck.hardware.set_run(False)
    lck.hardware.configure_mixer(
        freq=cavity_freq,
        in_ports=input_port,
        out_ports=cavity_port,
    )
    lck.hardware.configure_mixer(
        freq=qubit_freq_arr[0],
        out_ports=qubit_port,
    )
    lck.set_frequency(cavity_port, 0.0)
    lck.set_frequency(qubit_port, 0.0)
    lck.set_scale(cavity_port, cavity_amp, cavity_amp)
    lck.set_scale(qubit_port, qubit_amp_arr[0], qubit_amp_arr[0])
    lck.set_phase(cavity_port, 0.0, 0.0)
    lck.set_phase(qubit_port, 0.0, 0.0)
    lck.set_dither(cavity_port, dither)
    lck.set_dither(qubit_port, dither)
    lck.set_dma_source(input_port)
    lck.hardware.set_run(True)

    t_start = time.time()
    prev_print_len = 0
    count = 0
    print()
    for jj, qubit_amp in enumerate(qubit_amp_arr):
        for ii, qubit_freq in enumerate(qubit_freq_arr):
            lck.hardware.set_run(False)
            lck.hardware.configure_mixer(
                freq=qubit_freq,
                out_ports=qubit_port,
            )
            lck.set_scale(qubit_port, qubit_amp, qubit_amp)
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

    print(f"Measurement completed in: {format_sec(time.time()-t_start):s}")
    # Mute outputs at the end of the sweep
    lck.hardware.set_run(False)
    lck.set_scale(cavity_port, 0.0, 0.0)
    lck.set_scale(qubit_port, 0.0, 0.0)
    if USE_JPA:
        lck.hardware.set_lmx(0.0, 0)

if USE_JPA:
    mla.lockin.set_dc_offset(jpa_bias_port, 0.0)
    mla.disconnect()

# *************************
# *** Save data to HDF5 ***
# *************************
script_path = os.path.realpath(__file__)  # full path of current script
current_dir, script_basename = os.path.split(script_path)
script_filename = os.path.splitext(script_basename)[
    0]  # name of current script
timestamp = time.strftime("%Y%m%d_%H%M%S",
                          time.localtime())  # current date and time
save_basename = f"{script_filename:s}_{timestamp:s}.h5"  # name of save file
save_path = os.path.join(current_dir, "data",
                         save_basename)  # full path of save file
source_code = get_sourcecode(
    script_path)  # save also the sourcecode of the script for future reference
with h5py.File(save_path, "w") as h5f:
    dt = h5py.string_dtype(encoding='utf-8')
    ds = h5f.create_dataset("source_code", (len(source_code), ), dt)
    for ii, line in enumerate(source_code):
        ds[ii] = line
    h5f.attrs["df"] = df
    h5f.attrs["dither"] = dither
    h5f.attrs["input_port"] = input_port
    h5f.attrs["cavity_port"] = cavity_port
    h5f.attrs["qubit_port"] = qubit_port
    h5f.attrs["cavity_amp"] = cavity_amp
    h5f.attrs["cavity_freq"] = cavity_freq
    h5f.create_dataset("qubit_freq_arr", data=qubit_freq_arr)
    h5f.create_dataset("qubit_amp_arr", data=qubit_amp_arr)
    h5f.create_dataset("resp_arr", data=resp_arr)
print(f"Data saved to: {save_path}")

# ********************
# *** Plot results ***
# ********************
# fig1 = load_two_tone_power.load(save_path)
