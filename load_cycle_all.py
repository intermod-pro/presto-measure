# -*- coding: utf-8 -*-
"""
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

import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

rcParams['figure.dpi'] = 108.8

if len(sys.argv) == 2:
    load_filename = sys.argv[1]
    print(f"Loading: {os.path.realpath(load_filename)}")
else:
    load_filename = None

with h5py.File(load_filename, "r") as h5f:
    rel_time_arr = h5f["rel_time_arr"][()]
    time_arr = h5f["time_arr"][()]
    t1_arr_q1 = h5f["t1_arr_q1"][()]
    t1_arr_q2 = h5f["t1_arr_q2"][()]
    t2_arr_q1 = h5f["t2_arr_q1"][()]
    t2_arr_q2 = h5f["t2_arr_q2"][()]
    source_code = h5f["source_code"][()]

# get rid of garbage
for data in [t1_arr_q1, t1_arr_q2, t2_arr_q1, t2_arr_q2]:
    idx = np.logical_or(data < 1e-5, data > 1e-2)
    data[idx] = np.nan

# Plot time series
fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax11, ax12 = ax1

ax11.semilogy(rel_time_arr / 3_600, 1e6 * t1_arr_q1, '.', c="tab:blue", label="qubit 1")
ax11.semilogy(rel_time_arr / 3_600, 1e6 * t1_arr_q2, '.', c="tab:orange", label="qubit 2")
ax11.set_ylabel("T1 [μs]")
ax11.legend()

ax12.semilogy(rel_time_arr / 3_600, 1e6 * t2_arr_q1, '.', c="tab:blue", label="qubit 1")
ax12.semilogy(rel_time_arr / 3_600, 1e6 * t2_arr_q2, '.', c="tab:orange", label="qubit 2")
ax12.set_ylabel("T2* [μs]")

ax12.set_xlabel("Time since start [h]")
ax12.legend()

fig1.show()

# Plot histogram
fig2, ax2 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax21, ax22 = ax2
ax21.hist(1e6 * t1_arr_q1, bins=100, range=(0, 200), alpha=0.5, label="qubit 1")
ax21.hist(1e6 * t1_arr_q2, bins=100, range=(0, 200), alpha=0.5, label="qubit 2")
ax22.hist(1e6 * t2_arr_q1, bins=100, range=(0, 200), alpha=0.5, label="qubit 1")
ax22.hist(1e6 * t2_arr_q2, bins=100, range=(0, 200), alpha=0.5, label="qubit 2")
ax21.set_ylabel("T1 counts")
ax22.set_ylabel("T2* counts")
ax22.set_xlabel("Time constant [μs]")
ax21.legend()
ax22.legend()
fig2.show()
