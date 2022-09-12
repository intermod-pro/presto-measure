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
        readout_freq = h5f.attrs["readout_freq"]
        control_freq = h5f.attrs["control_freq"]
        readout_duration = h5f.attrs["readout_duration"]
        control_duration = h5f.attrs["control_duration"]
        match_duration = h5f.attrs["match_duration"]
        readout_amp = h5f.attrs["readout_amp"]
        control_amp = h5f.attrs["control_amp"]
        sample_duration = h5f.attrs["sample_duration"]
        wait_delay = h5f.attrs["wait_delay"]
        readout_sample_delay = h5f.attrs["readout_sample_delay"]
        readout_match_delay = h5f.attrs["readout_match_delay"]
        t_arr = h5f["t_arr"][()]
        store_arr = h5f["store_arr"][()]
        match_i_data = h5f["match_i_data"][()]
        match_q_data = h5f["match_q_data"][()]
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

    # Plot raw store data for first iteration as a check
    fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
    ax11, ax12 = ax1
    # ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    # ax12.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    ax11.plot(1e9 * t_arr, np.abs(store_arr[0, 0, :]))
    ax11.plot(1e9 * t_arr, np.abs(store_arr[1, 0, :]))
    ax12.plot(1e9 * t_arr, np.angle(store_arr[0, 0, :]))
    ax12.plot(1e9 * t_arr, np.angle(store_arr[1, 0, :]))
    ax12.set_xlabel("Time [ns]")
    fig1.show()

    # # Analyze
    data = match_i_data + 1j * match_q_data
    data = rotate_opt(data)
    data_g = data[0::2]
    data_e = data[1::2]
    x_g = data_g.real
    y_g = data_g.imag
    x_e = data_e.real
    y_e = data_e.imag
    std = max([x.std() for x in [x_g, y_g, x_e, y_e]])
    x_min = min(x_g.mean(), x_e.mean()) - 5 * std
    x_max = max(x_g.mean(), x_e.mean()) + 5 * std
    y_min = min(y_g.mean(), y_e.mean()) - 5 * std
    y_max = max(y_g.mean(), y_e.mean()) + 5 * std
    H_g, xedges, yedges = np.histogram2d(
        x_g, y_g, bins=100, range=[[x_min, x_max], [y_min, y_max]], density=True
    )
    H_e, xedges, yedges = np.histogram2d(
        x_e, y_e, bins=100, range=[[x_min, x_max], [y_min, y_max]], density=True
    )
    H_g = H_g.T
    H_e = H_e.T
    z_max = max(H_g.max(), H_e.max())

    # fig2, ax2 = plt.subplots(tight_layout=True)
    # ax2.plot(match_i_data[0::2], match_q_data[0::2], '.')
    # ax2.plot(match_i_data[1::2], match_q_data[1::2], '.')
    # ax2.plot(np.mean(match_i_data[0::2]), np.mean(match_q_data[0::2]), '.')
    # ax2.plot(np.mean(match_i_data[1::2]), np.mean(match_q_data[1::2]), '.')
    # ax2.axhline(0.0, c="tab:gray", alpha=0.25)
    # ax2.axvline(0.0, c="tab:gray", alpha=0.25)
    # fig2.show()

    fig3, ax3 = plt.subplots(1, 2, sharex=True, sharey=True, tight_layout=True, figsize=(9.6, 4.8))
    ax31, ax32 = ax3
    ax31.imshow(
        H_g,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="RdBu_r",
        vmin=-z_max,
        vmax=z_max,
    )
    ax32.imshow(
        H_e,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="RdBu_r",
        vmin=-z_max,
        vmax=z_max,
    )
    ax31.axhline(0.0, c="tab:gray", alpha=0.25)
    ax31.axvline(0.0, c="tab:gray", alpha=0.25)
    ax32.axhline(0.0, c="tab:gray", alpha=0.25)
    ax32.axvline(0.0, c="tab:gray", alpha=0.25)
    ax31.set_aspect("equal")
    ax32.set_aspect("equal")
    fig3.show()

    xdata = 0.5 * (xedges[1:] + xedges[:-1])
    fig4, ax4 = plt.subplots(tight_layout=True)
    ax4.plot(xdata, np.sum(H_g, axis=0))
    ax4.plot(xdata, np.sum(H_e, axis=0))
    fig4.show()

    return fig1, fig2


if __name__ == "__main__":
    fig1, fig2 = load(load_filename)
