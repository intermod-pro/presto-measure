# -*- coding: utf-8 -*-
import h5py
import matplotlib.pyplot as plt
import numpy as np

load_filename = "jpa_sweep_pwr_bias_gain_20201226_065543.h5"
load_filename = "jpa_sweep_pwr_bias_gain_20201227_011557.h5"
load_filename = "jpa_sweep_pwr_bias_gain_20201227_132023.h5"
load_filename = "jpa_sweep_pwr_bias_gain_20201228_055817.h5"
with h5py.File(load_filename, "r") as h5f:
    df = h5f.attrs["df"]
    Navg = h5f.attrs["Navg"]
    amp = h5f.attrs["amp"]
    dither = h5f.attrs["dither"]
    input_port = h5f.attrs["input_port"]
    output_port = h5f.attrs["output_port"]
    bias_port = h5f.attrs["bias_port"]
    freq_arr = h5f["freq_arr"][()]
    bias_arr = h5f["bias_arr"][()]
    resp_arr = h5f["resp_arr"][()]
    pump_pwr_arr = h5f["pump_pwr_arr"][()]
    source_code = h5f["source_code"][()]

# extract reference
ref_arr = resp_arr[0, :, :]
resp_arr = resp_arr[1:, :, :]
pump_pwr_arr = pump_pwr_arr[1:]

nr_pump_pwr = len(pump_pwr_arr)
nr_bias = len(bias_arr)
bias_min = bias_arr.min()
bias_max = bias_arr.max()
freq_min = freq_arr.min()
freq_max = freq_arr.max()

ref_db = 20 * np.log10(np.abs(ref_arr))
# ref_grpdly = np.diff(np.unwrap(np.angle(ref_plot)))
data_db = 20 * np.log10(np.abs(resp_arr))
# data_grpdly = np.diff(np.unwrap(np.angle(data_plot)))

gain_db = np.zeros_like(data_db)
for pp in range(nr_pump_pwr):
    for bb in range(nr_bias):
        gain_db[pp, bb, :] = data_db[pp, bb, :] - ref_db[bb, :]

low = np.percentile(gain_db, 1)
high = np.percentile(gain_db, 99)
lim = max(abs(low), abs(high))

fig, ax = plt.subplots(6, 10, sharex=True, sharey=True, tight_layout=True)
for ii in range(60):
    _ax = ax[ii // 10][ii % 10]
    im = _ax.imshow(gain_db[ii, :, :],
                    origin='lower',
                    aspect='auto',
                    extent=(1e-9 * freq_min, 1e-9 * freq_max, bias_min, bias_max),
                    vmin=-lim,
                    vmax=lim,
                    cmap="RdBu_r")
    _ax.set_title(str(pump_pwr_arr[ii]))
fig.show()

bias_idx = np.argmin(np.abs(bias_arr - 0.44))
fig, ax = plt.subplots()
for pp in range(25, 35):
    pwr = pump_pwr_arr[pp]
    ax.plot(1e-9 * freq_arr, gain_db[pp, bias_idx, :], label=str(pwr))
ax.legend()
fig.show()

pwr_idx = np.argmin(np.abs(pump_pwr_arr - 7.5))
bias_start = np.argmin(np.abs(bias_arr - 0.43))
bias_stop = np.argmin(np.abs(bias_arr - 0.45))
fig, ax = plt.subplots()
for bb in range(bias_start, bias_stop):
    bias = bias_arr[bb]
    ax.plot(1e-9 * freq_arr, gain_db[pwr_idx, bb, :], label=str(bias))
ax.legend()
fig.show()
