# -*- coding: utf-8 -*-
"""
Measure Rabi oscillation by changing the duration of the control pulse.
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
from presto import pulsed
from presto.utils import get_sourcecode

import load_rabi_time

# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42878
EXT_REF_CLK = False  # set to True to lock to an external reference clock

# Use JPA?
JPA = False
if JPA:
    if '/home/riccardo/IntermodulatorSuite' not in sys.path:
        sys.path.append('/home/riccardo/IntermodulatorSuite')
    from mlaapi import mla_api, mla_globals
    settings = mla_globals.read_config()
    mla = mla_api.MLA(settings)

freq_if = 100 * 1e6

# cavity drive: readout
readout_freq = 6.029 * 1e9  # Hz
readout_amp = 3e-3  # FS
readout_duration = 2e-6  # s, duration of the readout pulse
readout_port = 1

# qubit drive: control
control_freq = 4.0807 * 1e9  # Hz
control_amp = 0.01  # FS
control_port = 7

# cavity readout: sample
sample_duration = 2100 * 1e-9  # s, duration of the sampling window
sample_port = 1

# Rabi experiment
num_averages = 100_000
rabi_n = 512  # number of steps when changing duration of control pulse
rabi_dt = 2e-9  # s, step size when changing duration of control pulse
wait_delay = 500e-6  # s, delay between repetitions to allow the qubit to decay
readout_sample_delay = 250 * 1e-9  # s, delay between readout pulse and sample window to account for latency

# Instantiate interface class
if JPA:
    jpa_pump_freq = 2 * 6.031e9  # Hz
    jpa_pump_pwr = 7  # lmx units
    jpa_bias = +0.432  # V
    bias_port = 1
    mla.connect()
with pulsed.Pulsed(
        address=ADDRESS,
        port=PORT,
        ext_ref_clk=EXT_REF_CLK,
        adc_mode=pulsed.MODE_MIX,
        dac_mode=pulsed.MODE_LSB if freq_if > 0.0 else pulsed.MODE_MIX,
) as pls:
    pls.hardware.set_adc_attenuation(sample_port, 0.0)
    pls.hardware.set_dac_current(readout_port, 32_000)
    pls.hardware.set_dac_current(control_port, 32_000)
    pls.hardware.set_inv_sinc(readout_port, 0)
    pls.hardware.set_inv_sinc(control_port, 0)
    pls.hardware.configure_mixer(
        freq=readout_freq + freq_if,
        in_ports=None if pls.adc_mode == pulsed.MODE_DIRECT else sample_port,
        out_ports=readout_port,
        sync=False,  # sync in next call
    )
    pls.hardware.configure_mixer(
        freq=control_freq + freq_if,
        out_ports=control_port,
    )
    if JPA:
        pls.hardware.set_lmx(jpa_pump_freq, jpa_pump_pwr)
        mla.lockin.set_dc_offset(bias_port, jpa_bias)
        time.sleep(1.0)
    else:
        pls.hardware.set_lmx(0.0, 0)

    # ************************************
    # *** Setup measurement parameters ***
    # ************************************

    # Setup lookup tables for frequencies
    # we only need to use carrier 1
    pls.setup_freq_lut(
        output_ports=readout_port,
        carrier=1,
        frequencies=freq_if,
        phases=0.0,
    )
    pls.setup_freq_lut(control_port, 1, freq_if, 0.0)

    # Setup lookup tables for amplitudes
    pls.setup_scale_lut(
        output_ports=readout_port,
        scales=readout_amp,
    )
    pls.setup_scale_lut(control_port, control_amp)

    # Setup readout and control pulses
    # use setup_long_drive to create a pulse with square envelope
    # setup_long_drive supports smooth rise and fall transitions for the pulse,
    # but we keep it simple here
    readout_pulse = pls.setup_long_drive(
        output_port=readout_port,
        carrier=1,
        duration=readout_duration,
        rise_time=0e-9,
        fall_time=0e-9,
    )
    control_pulse = pls.setup_long_drive(control_port, 1, rabi_dt)  # minimum-length pulse, extend it later

    # Setup sampling window
    pls.set_store_ports(sample_port)
    pls.set_store_duration(sample_duration)

    # ******************************
    # *** Program pulse sequence ***
    # ******************************
    T = 0.0  # s, start at time zero ...
    for ii in range(rabi_n):
        pls.reset_phase(T, control_port)
        if ii > 0:  # no control pulse the first iteration
            # Rabi pulse
            rabi_duration = ii * rabi_dt
            control_pulse.set_total_duration(rabi_duration)
            pls.output_pulse(T, control_pulse)
        # Readout pulse starts right after control pulse
        T += ii * rabi_dt
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
    t_arr, store_arr = pls.run(
        period=T,
        repeat_count=1,
        num_averages=num_averages,
        print_time=True,
    )

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
    h5f.attrs["readout_freq"] = readout_freq
    h5f.attrs["readout_duration"] = readout_duration
    h5f.attrs["readout_amp"] = readout_amp
    h5f.attrs["control_amp"] = control_amp
    h5f.attrs["sample_duration"] = sample_duration
    h5f.attrs["rabi_n"] = rabi_n
    h5f.attrs["rabi_dt"] = rabi_dt
    h5f.attrs["wait_delay"] = wait_delay
    h5f.attrs["readout_sample_delay"] = readout_sample_delay
    h5f.attrs["freq_if"] = freq_if
    h5f.create_dataset("t_arr", data=t_arr)
    h5f.create_dataset("store_arr", data=store_arr)
print(f"Data saved to: {save_path}")

# *****************
# *** Plot data ***
# *****************
fig1, fig2 = load_rabi_time.load(save_path)
