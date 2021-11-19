# -*- coding: utf-8 -*-
"""
Loader for files saved by two_tone_pulsed.py
"""
import os
import sys

import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

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
        readout_duration = h5f.attrs["readout_duration"]
        control_duration = h5f.attrs["control_duration"]
        readout_amp = h5f.attrs["readout_amp"]
        control_amp = h5f.attrs["control_amp"]
        sample_duration = h5f.attrs["sample_duration"]
        nr_freqs = h5f.attrs["nr_freqs"]
        wait_delay = h5f.attrs["wait_delay"]
        readout_sample_delay = h5f.attrs["readout_sample_delay"]
        control_freq_arr = h5f["control_freq_arr"][()]
        t_arr = h5f["t_arr"][()]
        store_arr = h5f["store_arr"][()]

    t_low = 1500 * 1e-9
    t_high = 2000 * 1e-9
    t_span = t_high - t_low
    idx_low = np.argmin(np.abs(t_arr - t_low))
    idx_high = np.argmin(np.abs(t_arr - t_high))
    idx = np.arange(idx_low, idx_high)
    nr_samples = len(idx)

    # Plot raw store data for first iteration as a check
    fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
    ax11, ax12 = ax1
    ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    ax12.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    ax11.plot(1e9 * t_arr, np.abs(store_arr[0, 0, :]))
    ax12.plot(1e9 * t_arr, np.angle(store_arr[0, 0, :]))
    ax12.set_xlabel("Time [ns]")
    fig1.show()

    # Analyze
    resp_arr = np.mean(store_arr[:, 0, idx], axis=-1)
    data = rotate_opt(resp_arr)

    fig2, ax2 = plt.subplots(4, 1, sharex=True, figsize=(6.4, 6.4), tight_layout=True)
    ax21, ax22, ax23, ax24 = ax2
    ax21.plot(1e-9 * control_freq_arr, np.abs(data))
    ax22.plot(1e-9 * control_freq_arr, np.angle(data))
    ax23.plot(1e-9 * control_freq_arr, np.real(data))
    ax24.plot(1e-9 * control_freq_arr, np.imag(data))

    ax21.set_ylabel("Amplitude [FS]")
    ax22.set_ylabel("Phase [rad]")
    ax23.set_ylabel("I [FS]")
    ax24.set_ylabel("Q [FS]")
    ax2[-1].set_xlabel("Control frequency [GHz]")
    fig2.show()

    return fig1, fig2


if __name__ == "__main__":
    fig = load(load_filename)
