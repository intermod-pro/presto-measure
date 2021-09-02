# -*- coding: utf-8 -*-
import os
import sys

import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from presto.pulsed import MAX_TEMPLATE_LEN
from presto.utils import rotate_opt

rcParams['figure.dpi'] = 108.8

if len(sys.argv) == 2:
    load_filename = sys.argv[1]
    print(f"Loading: {os.path.realpath(load_filename)}")
else:
    load_filename = None


def load(load_filename):
    with h5py.File(load_filename, "r") as h5f:
        num_averages = h5f.attrs["num_averages"]
        readout_freq = h5f.attrs["readout_freq"]
        control_freq = h5f.attrs["control_freq"]
        readout_duration = h5f.attrs["readout_duration"]
        control_duration = h5f.attrs["control_duration"]
        readout_amp = h5f.attrs["readout_amp"]
        control_amp = h5f.attrs["control_amp"]
        sample_duration = h5f.attrs["sample_duration"]
        wait_delay = h5f.attrs["wait_delay"]
        readout_sample_delay = h5f.attrs["readout_sample_delay"]
        t_arr = h5f["t_arr"][()]
        store_arr = h5f["store_arr"][()]
        source_code = h5f["source_code"][()]

    # t_low = 1500 * 1e-9
    # t_high = 2000 * 1e-9
    # t_span = t_high - t_low
    # idx_low = np.argmin(np.abs(t_arr - t_low))
    # idx_high = np.argmin(np.abs(t_arr - t_high))
    # idx = np.arange(idx_low, idx_high)
    # nr_samples = len(idx)
    nr_samples = len(t_arr)
    t_span = nr_samples * (t_arr[1] - t_arr[0])

    trace_g = store_arr[0, 0, :]
    trace_e = store_arr[1, 0, :]
    distance = np.abs(trace_e - trace_g)
    match_len = MAX_TEMPLATE_LEN // 2  # I and Q

    max_idx = 0
    max_dist = 0.0
    idx = -2
    while idx + 2 + match_len <= nr_samples:
        idx += 2  # next clock cycle
        dist = np.sum(distance[idx:idx + match_len])
        if dist > max_dist:
            max_dist = dist
            max_idx = idx

    template_g = trace_g[max_idx:max_idx + match_len]
    template_e = trace_e[max_idx:max_idx + match_len]
    match_t_in_store = t_arr[max_idx]
    print(f"Match starts at {1e9 * t_arr[max_idx]:.0f} ns in store")
    print(f"Saving templates back into: {load_filename}")
    with h5py.File(load_filename, "r+") as h5f:
        h5f.attrs["match_t_in_store"] = match_t_in_store
        h5f.create_dataset("template_g", data=template_g)
        h5f.create_dataset("template_e", data=template_e)

    # Plot raw store data for first iteration as a check
    fig1, ax1 = plt.subplots(4, 1, sharex=True, tight_layout=True)
    for ax_ in ax1:
        ax_.axvspan(1e9 * t_arr[max_idx], 1e9 * t_arr[max_idx + match_len], facecolor="#dfdfdf")
    ax1[0].plot(1e9 * t_arr, np.abs(trace_g))
    ax1[0].plot(1e9 * t_arr, np.abs(trace_e))
    ax1[1].plot(1e9 * t_arr, np.angle(trace_g))
    ax1[1].plot(1e9 * t_arr, np.angle(trace_e))
    ax1[2].plot(1e9 * t_arr, np.real(trace_g))
    ax1[2].plot(1e9 * t_arr, np.real(trace_e))
    ax1[3].plot(1e9 * t_arr, np.imag(trace_g))
    ax1[3].plot(1e9 * t_arr, np.imag(trace_e))
    ax1[-1].set_xlabel("Time [ns]")
    fig1.show()

    fig2, ax2 = plt.subplots(tight_layout=True)
    ax2.axvspan(1e9 * t_arr[max_idx], 1e9 * t_arr[max_idx + match_len], facecolor="#dfdfdf")
    ax2.plot(1e9 * t_arr, distance)
    fig2.show()

    return fig1, fig2


if __name__ == "__main__":
    fig1, fig2 = load(load_filename)
