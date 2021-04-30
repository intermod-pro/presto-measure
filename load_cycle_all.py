import h5py
import matplotlib.pyplot as plt
import numpy as np

load_path = "data/cycle_all_20210428_192655.h5"

with h5py.File(load_path, "r") as h5f:
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

ax11.semilogy(rel_time_arr / 3_600, 1e6 * t1_arr_q1, '.', c="tab:blue")
ax11.semilogy(rel_time_arr / 3_600, 1e6 * t1_arr_q2, '.', c="tab:orange")
ax11.set_ylabel("T1 [us]")

ax12.semilogy(rel_time_arr / 3_600, 1e6 * t2_arr_q1, '.', c="tab:blue")
ax12.semilogy(rel_time_arr / 3_600, 1e6 * t2_arr_q2, '.', c="tab:orange")
ax12.set_ylabel("T2* [us]")

ax12.set_xlabel("Time since start [h]")

fig1.show()

# Plot histogram
fig2, ax2 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax21, ax22 = ax2
ax21.hist(1e6 * t1_arr_q1, alpha=0.5)
ax21.hist(1e6 * t1_arr_q2, alpha=0.5)
ax22.hist(1e6 * t2_arr_q1, alpha=0.5)
ax22.hist(1e6 * t2_arr_q2, alpha=0.5)
fig2.show()
