# -*- coding: utf-8 -*-
import os
import sys

import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from presto.utils import rotate_opt

rcParams["figure.dpi"] = 108.8

if len(sys.argv) == 2:
    load_filename = sys.argv[1]
    print(f"Loading: {os.path.realpath(load_filename)}")
else:
    load_filename = None


def load(load_filename):
    with h5py.File(load_filename, "r") as h5f:
        num_averages = h5f.attrs["num_averages"]
        control_freq_1 = h5f.attrs["control_freq_1"]
        control_freq_2 = h5f.attrs["control_freq_2"]
        control_if = h5f.attrs["control_if"]
        readout_freq_1 = h5f.attrs["readout_freq_1"]
        readout_freq_2 = h5f.attrs["readout_freq_2"]
        readout_duration = h5f.attrs["readout_duration"]
        control_duration = h5f.attrs["control_duration"]
        readout_amp = h5f.attrs["readout_amp"]
        control_amp_1 = h5f.attrs["control_amp_1"]
        control_amp_2 = h5f.attrs["control_amp_2"]
        sample_duration = h5f.attrs["sample_duration"]
        wait_delay = h5f.attrs["wait_delay"]
        readout_sample_delay = h5f.attrs["readout_sample_delay"]
        coupler_dc_bias = h5f.attrs["coupler_dc_bias"]
        nr_freqs = h5f.attrs["nr_freqs"]
        nr_amps = h5f.attrs["nr_amps"]
        coupler_ac_duration = h5f.attrs["coupler_ac_duration"]
        t_arr = h5f["t_arr"][()]
        store_arr = h5f["store_arr"][()]
        coupler_ac_freq_arr = h5f["coupler_ac_freq_arr"][()]
        coupler_ac_amp_arr = h5f["coupler_ac_amp_arr"][()]

    t_low = 1500 * 1e-9
    t_high = 2000 * 1e-9
    idx_low = np.argmin(np.abs(t_arr - t_low))
    idx_high = np.argmin(np.abs(t_arr - t_high))
    idx = np.arange(idx_low, idx_high)

    # # Plot raw store data for first iteration as a check
    # fig0, ax0 = plt.subplots(2, 1, sharex=True, tight_layout=True)
    # ax01, ax02 = ax0
    # ax01.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    # ax02.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    # ax01.plot(1e9 * t_arr, np.abs(store_arr[0, 0, :]))
    # ax02.plot(1e9 * t_arr, np.angle(store_arr[0, 0, :]))
    # ax02.set_xlabel("Time [ns]")
    # fig0.show()

    resp_arr = np.mean(store_arr[:, 0, idx], axis=-1)
    data = rotate_opt(resp_arr)
    data.shape = (nr_amps, nr_freqs)
    plot_data = data.real

    # choose limits for colorbar
    cutoff = 0.0  # %
    lowlim = np.percentile(plot_data, cutoff)
    highlim = np.percentile(plot_data, 100.0 - cutoff)

    # extent
    x_min = 1e-6 * coupler_ac_freq_arr[0]
    x_max = 1e-6 * coupler_ac_freq_arr[-1]
    dx = 1e-6 * (coupler_ac_freq_arr[1] - coupler_ac_freq_arr[0])
    y_min = coupler_ac_amp_arr[0]
    y_max = coupler_ac_amp_arr[-1]
    dy = coupler_ac_amp_arr[1] - coupler_ac_amp_arr[0]

    fig1, ax1 = plt.subplots(tight_layout=True)
    im = ax1.imshow(
        plot_data,
        origin="lower",
        aspect="auto",
        interpolation="none",
        extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
        vmin=lowlim,
        vmax=highlim,
    )
    ax1.set_xlabel("Coupler frequency [MHz]")
    ax1.set_ylabel("Coupler amplitude [FS]")
    cb = fig1.colorbar(im)
    cb.set_label("Response amplitude [FS]")
    fig1.show()

    return fig1


if __name__ == "__main__":
    fig1 = load(load_filename)
