# -*- coding: utf-8 -*-
"""
Measure Rabi oscillation by changing the amplitude of the control pulse.
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
from presto import pulsed
from presto.utils import get_sourcecode, sin2

import load_rabi_amp

# Presto's IP address or hostname
ADDRESS = "192.0.2.53"
EXT_REF_CLK = False  # set to True to lock to an external reference clock

# cavity drive: readout
readout_freq = 6.213095 * 1e9  # Hz, frequency for resonator readout
readout_amp = 10**(-10.0 / 20)  # FS
readout_duration = 2e-6  # s, duration of the readout pulse
readout_port = 1

# qubit drive: control
control_freq = 4.141 * 1e9  # Hz
control_duration = 100e-9  # s, duration of the control pulse
control_port = 5

# cavity readout: sample
sample_duration = 4 * 1e-6  # s, duration of the sampling window
sample_port = 1

# Rabi experiment
num_averages = 10_000
rabi_n = 128  # number of steps when changing duration of control pulse
control_amp_arr = np.linspace(0.0, 0.707, rabi_n)  # FS, amplitudes for control pulse
wait_delay = 500e-6  # s, delay between repetitions to allow the qubit to decay
readout_sample_delay = 300 * 1e-9  # s, delay between readout pulse and sample window to account for latency

# Instantiate interface class
with pulsed.Pulsed(
        address=ADDRESS,
        ext_ref_clk=EXT_REF_CLK,
        adc_mode=cmd.AdcMixed,
        adc_fsample=cmd.AdcG2,
        dac_mode=[cmd.DacMixed42, cmd.DacMixed02, cmd.DacMixed02, cmd.DacMixed02],
        dac_fsample=[cmd.DacG10, cmd.DacG6, cmd.DacG6, cmd.DacG6],
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
        scales=control_amp_arr,
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
    # For the control pulse we create a sine-squared envelope,
    # and use setup_template to use the user-defined envelope
    control_ns = int(round(control_duration * pls.get_fs("dac")))  # number of samples in the control template
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
    # Control pulse
    pls.reset_phase(T, control_port)
    pls.output_pulse(T, control_pulse)
    # Readout pulse starts right after control pulse
    T += control_duration
    pls.reset_phase(T, readout_port)
    pls.output_pulse(T, readout_pulse)
    # Sampling window
    pls.store(T + readout_sample_delay)
    # Move to next Rabi amplitude
    T += readout_duration
    pls.next_scale(T, control_port)  # every iteration will have a different amplitude
    # Wait for decay
    T += wait_delay

    # **************************
    # *** Run the experiment ***
    # **************************
    # repeat the whole sequence `rabi_n` times
    # then average `num_averages` times
    pls.run(
        period=T,
        repeat_count=rabi_n,
        num_averages=num_averages,
        print_time=True,
    )
    t_arr, (data_I, data_Q) = pls.get_store_data()

store_arr = data_I + 1j * data_Q

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
    h5f.attrs["control_duration"] = control_duration
    h5f.attrs["readout_amp"] = readout_amp
    h5f.attrs["sample_duration"] = sample_duration
    h5f.attrs["rabi_n"] = rabi_n
    h5f.attrs["wait_delay"] = wait_delay
    h5f.attrs["readout_sample_delay"] = readout_sample_delay
    h5f.create_dataset("control_amp_arr", data=control_amp_arr)
    h5f.create_dataset("t_arr", data=t_arr)
    h5f.create_dataset("store_arr", data=store_arr)
print(f"Data saved to: {save_path}")

# *****************
# *** Plot data ***
# *****************
fig1, fig2 = load_rabi_amp.load(os.path.join(save_path))
