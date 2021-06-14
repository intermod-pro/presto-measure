# -*- coding: utf-8 -*-
"""
Loader for files saved by t1.py
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
from scipy.optimize import curve_fit

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
        control_freq = h5f.attrs["control_freq"]
        readout_duration = h5f.attrs["readout_duration"]
        control_duration = h5f.attrs["control_duration"]
        readout_amp = h5f.attrs["readout_amp"]
        control_amp = h5f.attrs["control_amp"]
        sample_duration = h5f.attrs["sample_duration"]
        nr_freqs = h5f.attrs["nr_freqs"]
        df = h5f.attrs["df"]
        readout_nco = h5f.attrs["readout_nco"]
        readout_if_center = h5f.attrs["readout_if_center"]
        wait_delay = h5f.attrs["wait_delay"]
        readout_sample_delay = h5f.attrs["readout_sample_delay"]
        t_arr = h5f["t_arr"][()]
        store_arr = h5f["store_arr"][()]
        readout_freq_arr = h5f["readout_freq_arr"][()]
        readout_if_arr = h5f["readout_if_arr"][()]
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
    ax12.plot(1e9 * t_arr, np.angle(store_arr[0, 0, :]))
    ax12.set_xlabel("Time [ns]")
    fig1.show()

    # Analyze
    store_arr.shape = (nr_freqs, 2, len(t_arr))
    resp_arr = np.zeros((2, nr_freqs), np.complex128)
    rms_arr = np.zeros((2, nr_freqs), np.float64)
    for ff in range(nr_freqs):
        f_if = readout_if_arr[ff]
        n1 = int(round(f_if * t_span))
        resp_arr[0, ff] = np.fft.fft(store_arr[ff, 0, :])[n1] / nr_samples
        resp_arr[1, ff] = np.fft.fft(store_arr[ff, 1, :])[n1] / nr_samples
        rms_arr[0, ff] = np.std(store_arr[ff, 0, :])
        rms_arr[1, ff] = np.std(store_arr[ff, 1, :])

    fig2, ax2 = plt.subplots(3, 1, sharex=True, tight_layout=True)
    ax21, ax22, ax23 = ax2
    ax21.plot(1e-9 * readout_freq_arr, np.abs(resp_arr[0, :]))
    ax21.plot(1e-9 * readout_freq_arr, np.abs(resp_arr[1, :]))
    ax22.plot(1e-9 * readout_freq_arr, np.angle(resp_arr[0, :]))
    ax22.plot(1e-9 * readout_freq_arr, np.angle(resp_arr[1, :]))
    ax23.plot(1e-9 * readout_freq_arr, rms_arr[0, :])
    ax23.plot(1e-9 * readout_freq_arr, rms_arr[1, :])

    ax21.set_ylabel("Amplitude [FS]")
    ax22.set_ylabel("Phase [rad]")
    ax2[-1].set_xlabel("Readout frequency [GHz]")
    fig2.show()

    return fig1, fig2


if __name__ == "__main__":
    fig1, fig2 = load(load_filename)
