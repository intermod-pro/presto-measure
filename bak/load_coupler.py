# -*- coding: utf-8 -*-
import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from presto.utils import rotate_opt

rcParams["figure.dpi"] = 108.8

load_filename = "data/coupler_20210429_165327.h5"


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
        coupler_ac_amp = h5f.attrs["coupler_ac_amp"]
        nr_freqs = h5f.attrs["nr_freqs"]
        coupler_ac_duration = h5f.attrs["coupler_ac_duration"]
        t_arr = h5f["t_arr"][()]
        store_arr = h5f["store_arr"][()]
        coupler_ac_freq_arr = h5f["coupler_ac_freq_arr"][()]

    t_low = 1500 * 1e-9
    t_high = 2000 * 1e-9
    idx_low = np.argmin(np.abs(t_arr - t_low))
    idx_high = np.argmin(np.abs(t_arr - t_high))
    idx = np.arange(idx_low, idx_high)

    # Plot raw store data for first iteration as a check
    fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
    ax11, ax12 = ax1
    ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    ax12.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    ax11.plot(1e9 * t_arr, np.abs(store_arr[0, 0, :]))
    ax12.plot(1e9 * t_arr, np.angle(store_arr[0, 0, :]))
    ax12.set_xlabel("Time [ns]")
    fig1.show()

    # Analyze T2
    resp_arr = np.mean(store_arr[:, 0, idx], axis=-1)
    data = rotate_opt(resp_arr)

    fig2, ax2 = plt.subplots(4, 1, sharex=True, figsize=(6.4, 6.4), tight_layout=True)
    ax21, ax22, ax23, ax24 = ax2
    ax21.plot(coupler_ac_freq_arr, np.abs(data))
    ax22.plot(coupler_ac_freq_arr, np.unwrap(np.angle(data)))
    ax23.plot(coupler_ac_freq_arr, np.real(data))
    ax24.plot(coupler_ac_freq_arr, np.imag(data))

    ax21.set_ylabel("Amplitude [FS]")
    ax22.set_ylabel("Phase [rad]")
    ax23.set_ylabel("I [FS]")
    ax24.set_ylabel("Q [FS]")
    ax2[-1].set_xlabel("Coupler AC amplitude [FS]")
    fig2.show()

    return fig1, fig2


if __name__ == "__main__":
    fig1, fig2 = load(load_filename)
