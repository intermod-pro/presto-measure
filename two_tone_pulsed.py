# -*- coding: utf-8 -*-
"""
Two-tone spectroscopy with Pulsed mode: sweep of pump frequency, with fixed pump power and fixed probe.
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
import matplotlib.pyplot as plt
import numpy as np
from presto import commands as cmd
from presto import pulsed
from presto.utils import get_sourcecode

# Presto's IP address or hostname
ADDRESS = "192.0.2.53"
EXT_REF_CLK = False  # set to True to lock to an external reference clock

# Readout
readout_freq = 6.213095 * 1e9  # Hz, frequency for resonator readout, resonator 1
readout_if = 0.0  # Hz, intermediate frequency for digital mixer
readout_amp = 10**(-10.0 / 20)  # FS
readout_duration = 2e-6  # s
readout_port = 1

# Control
control_amp = 10**(-40.0 / 20)  # FS, qubit 1
control_port = 5  # qubit 1
control_center_freq = 4.141 * 1e9  # Hz, center frequency for control-pulse, qubit 1
control_center_freq_if = 400e6  # Hz, intermediate frequency for digital mixer
control_freq_nco = control_center_freq - control_center_freq_if  # Hz, using upper sideband
control_span = 50e6  # Hz, span of sweep
nr_freqs = 128

# Sample
sample_delay = 300e-9  # s, delay between readout pulse and sample window to account for latency
sample_duration = 4e-6  # s, duration of the sampling window
sample_port = 1
nr_averages = 1_000

# other
wait_for_decay = 500e-6  # s

df = control_span / nr_freqs
control_duration = round(1 / df / 2e-9) * 2e-9  # s, multiple of 2 ns
df = 1 / control_duration
n_center = int(round(control_center_freq_if / df))
n_arr = n_center + np.arange(nr_freqs) - nr_freqs // 2
control_freq_if_arr = df * n_arr
control_freq_arr = control_freq_nco + control_freq_if_arr

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
        freq=readout_freq - readout_if,
        in_ports=sample_port,
        out_ports=readout_port,
        sync=False,  # sync at next call
    )
    pls.hardware.configure_mixer(
        freq=control_freq_nco,
        out_ports=control_port,
        sync=True,  # sync here
    )

    pls.set_store_ports(sample_port)
    pls.set_store_duration(sample_duration)
    pls.setup_freq_lut(
        output_ports=readout_port,
        group=0,
        frequencies=readout_if,
        phases=0.0,
        phases_q=0.0,
    )
    pls.setup_freq_lut(
        output_ports=control_port,
        group=0,
        frequencies=control_freq_if_arr,
        phases=0.0 * np.ones_like(control_freq_if_arr),
        phases_q=-np.pi / 2 * np.ones_like(control_freq_if_arr),
    )
    pls.setup_scale_lut(output_ports=readout_port, group=0, scales=readout_amp)
    pls.setup_scale_lut(output_ports=control_port, group=0, scales=control_amp)

    readout_pulse = pls.setup_long_drive(readout_port, 0, readout_duration, amplitude=1.0, amplitude_q=1.0)
    control_pulse = pls.setup_long_drive(control_port, 0, control_duration, amplitude=1.0, amplitude_q=1.0)

    T = 0.0
    # for ii in range(nr_freqs):
    pls.reset_phase(T, control_port)
    pls.output_pulse(T, control_pulse)
    pls.next_frequency(T + control_duration, control_port)  # control is done, prepare for next iteration

    T += control_duration
    pls.reset_phase(T, readout_port)
    pls.output_pulse(T, readout_pulse)
    pls.store(T + sample_delay)

    T += readout_duration
    T += wait_for_decay

    pls.run(period=T, repeat_count=nr_freqs, num_averages=nr_averages)

    t_arr, (data_I, data_Q) = pls.get_store_data()
    data = data_I + 1j * data_Q  # 0 Hz intermediate frequency: just make it complex

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
    h5f.attrs["sample_port"] = sample_port
    h5f.attrs["readout_port"] = readout_port
    h5f.attrs["control_port"] = control_port
    h5f.attrs["readout_amp"] = readout_amp
    h5f.attrs["control_amp"] = control_amp
    h5f.attrs["readout_freq"] = readout_freq
    h5f.attrs["readout_duration"] = readout_duration
    h5f.attrs["sample_delay"] = sample_delay
    h5f.attrs["nr_averages"] = nr_averages
    h5f.attrs["wait_for_decay"] = wait_for_decay
    h5f.create_dataset("control_freq_arr", data=control_freq_arr)
    h5f.create_dataset("t_arr", data=t_arr)
    h5f.create_dataset("data", data=data)
print(f"Data saved to: {save_path}")

signal = np.mean(np.abs(data[:, 0, 1_500:2_000]), axis=-1)

fig1, ax1 = plt.subplots(tight_layout=True)
ax1.plot(np.abs(data[0, 0, :]))
fig1.show()

fig2, ax2 = plt.subplots(tight_layout=True)
ax2.plot(1e-9 * control_freq_arr, signal)
fig2.show()
