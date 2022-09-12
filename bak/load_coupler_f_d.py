# -*- coding: utf-8 -*-
import os
import sys

import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

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
        nr_steps = h5f.attrs["nr_steps"]
        dt_steps = h5f.attrs["dt_steps"]
        coupler_ac_amp = h5f.attrs["coupler_ac_amp"]
        t_arr = h5f["t_arr"][()]
        store_arr = h5f["store_arr"][()]
        coupler_ac_freq_arr = h5f["coupler_ac_freq_arr"][()]
        coupler_ac_duration_arr = h5f["coupler_ac_duration_arr"][()]

        # these were added later
        try:
            # multiplexed readout
            readout_if_1 = h5f.attrs["readout_if_1"]
            readout_if_2 = h5f.attrs["readout_if_2"]
            readout_nco = h5f.attrs["readout_nco"]
        except KeyError:
            # only readout resonator 1
            readout_if_1 = 0.0
            readout_if_2 = None
            readout_nco = readout_freq_1

    t_low = 1500 * 1e-9
    t_high = 2000 * 1e-9
    idx_low = np.argmin(np.abs(t_arr - t_low))
    idx_high = np.argmin(np.abs(t_arr - t_high))
    idx = np.arange(idx_low, idx_high)

    # Plot raw store data for first iteration as a check
    # fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
    # ax11, ax12 = ax1
    # ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    # ax12.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    # ax11.plot(1e9 * t_arr, np.abs(store_arr[0, 0, :]))
    # ax12.plot(1e9 * t_arr, np.angle(store_arr[0, 0, :]))
    # ax12.set_xlabel("Time [ns]")
    # fig1.show()

    if readout_if_2 is None:
        # only readout resonator 1
        resp_arr = np.mean(store_arr[:, 0, idx], axis=-1)
        data = rotate_opt(resp_arr)
        data.shape = (nr_freqs, nr_steps)
        plot_data = data.real

        # choose limits for colorbar
        cutoff = 0.0  # %
        lowlim = np.percentile(plot_data, cutoff)
        highlim = np.percentile(plot_data, 100.0 - cutoff)

        # extent
        x_min = 1e9 * coupler_ac_duration_arr[0]
        x_max = 1e9 * coupler_ac_duration_arr[-1]
        dx = 1e9 * (coupler_ac_duration_arr[1] - coupler_ac_duration_arr[0])
        y_min = 1e-6 * coupler_ac_freq_arr[0]
        y_max = 1e-6 * coupler_ac_freq_arr[-1]
        dy = 1e-6 * (coupler_ac_freq_arr[1] - coupler_ac_freq_arr[0])

        fig2, ax2 = plt.subplots(tight_layout=True)
        im = ax2.imshow(
            plot_data,
            origin="lower",
            aspect="auto",
            interpolation="none",
            extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
            vmin=lowlim,
            vmax=highlim,
        )
        ax2.set_xlabel("Coupler duration [ns]")
        ax2.set_ylabel("Coupler frequency [MHz]")
        cb = fig2.colorbar(im)
        cb.set_label("Response I quadrature [FS]")
        fig2.show()
    else:
        # multiplexed readout
        dt = t_arr[1] - t_arr[0]
        nr_samples = len(idx)
        freq_arr = np.fft.fftfreq(nr_samples, dt)
        # complex FFT should take care of upper/lower sideband
        resp_fft = np.fft.fft(store_arr[:, 0, idx], axis=-1) / len(idx)
        idx_1 = np.argmin(np.abs(freq_arr - readout_if_1))
        idx_2 = np.argmin(np.abs(freq_arr - readout_if_2))
        resp_arr_1 = 2 * resp_fft[:, idx_1]
        resp_arr_2 = 2 * resp_fft[:, idx_2]
        data_1 = rotate_opt(resp_arr_1)
        data_2 = rotate_opt(resp_arr_2)
        data_1.shape = (nr_freqs, nr_steps)
        data_2.shape = (nr_freqs, nr_steps)
        plot_data_1 = data_1.real
        plot_data_2 = data_2.real

        # convert to population
        # g_state = -0.000199191400873993
        # e_state = -0.001772512103413742
        # plot_data_1 = (data_1.real - g_state) / (e_state - g_state)
        # g_state = -0.0005096104775241157
        # e_state = 0.0010518469556880861
        # plot_data_2 = (data_2.real - g_state) / (e_state - g_state)
        if abs(np.min(plot_data_1)) > abs(np.max(plot_data_1)):
            plot_data_1 *= -1
        if abs(np.min(plot_data_2)) > abs(np.max(plot_data_2)):
            plot_data_2 *= -1

        # choose limits for colorbar
        cutoff = 0.1  # %
        lowlim_1 = np.percentile(plot_data_1, cutoff)
        highlim_1 = np.percentile(plot_data_1, 100.0 - cutoff)
        lowlim_2 = np.percentile(plot_data_2, cutoff)
        highlim_2 = np.percentile(plot_data_2, 100.0 - cutoff)
        # alldata = (np.r_[plot_data_1, plot_data_2]).ravel()
        # lowlim = np.percentile(alldata, cutoff)
        # highlim = np.percentile(alldata, 100. - cutoff)
        # lowlim_1, highlim_1 = lowlim, highlim
        # lowlim_2, highlim_2 = lowlim, highlim

        # extent
        x_min = 1e9 * coupler_ac_duration_arr[0]
        x_max = 1e9 * coupler_ac_duration_arr[-1]
        dx = 1e9 * (coupler_ac_duration_arr[1] - coupler_ac_duration_arr[0])
        y_min = 1e-6 * coupler_ac_freq_arr[0]
        y_max = 1e-6 * coupler_ac_freq_arr[-1]
        dy = 1e-6 * (coupler_ac_freq_arr[1] - coupler_ac_freq_arr[0])

        fig2, ax2 = plt.subplots(1, 2, figsize=(9.6, 4.8), tight_layout=True)
        ax21, ax22 = ax2
        im1 = ax21.imshow(
            plot_data_1,
            origin="lower",
            aspect="auto",
            interpolation="none",
            extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
            vmin=lowlim_1,
            vmax=highlim_1,
        )
        im2 = ax22.imshow(
            plot_data_2,
            origin="lower",
            aspect="auto",
            interpolation="none",
            extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
            vmin=lowlim_2,
            vmax=highlim_2,
        )
        ax21.set_title("Population on qubit 1")
        ax22.set_title("Population on qubit 2")
        ax21.set_xlabel("Coupler duration [ns]")
        ax22.set_xlabel("Coupler duration [ns]")
        ax21.set_ylabel("Coupler frequency [MHz]")
        # ax22.set_ylabel("Coupler frequency [MHz]")
        # ax22.yaxis.set_label_position("right")
        # ax22.yaxis.tick_right()
        for tick in ax22.get_yticklabels():
            tick.set_visible(False)
        # cb1 = fig2.colorbar(im1, ax=ax21)
        # cb1.set_ticks([0, 1])
        # cb2 = fig2.colorbar(im2, ax=ax22)
        # cb2.set_ticks([0, 1])
        # cb = fig2.colorbar(im1, ax=[ax21, ax22], use_gridspec=False)
        fig2.show()

        # Line cut at the center frequency
        idx = np.argmax(np.mean(plot_data_1, axis=-1))
        fig3, ax3 = plt.subplots(tight_layout=True)
        ax3.plot(1e9 * coupler_ac_duration_arr, plot_data_1[idx, :], label="qubit 1")
        ax3.plot(1e9 * coupler_ac_duration_arr, plot_data_2[idx, :], label="qubit 2")
        ax3.set_xlabel("Coupler duration [ns]")
        ax3.set_ylabel("Population")
        fig3.show()

    return fig2


if __name__ == "__main__":
    fig = load(load_filename)
