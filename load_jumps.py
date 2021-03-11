# -*- coding: utf-8 -*-
"""
Loader for files saved by jumps.py
Copyright (C) 2021  Intermodulation Products AB.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
<https://www.gnu.org/licenses/>.
"""
import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
from sklearn.mixture import GaussianMixture

rcParams['figure.dpi'] = 108.8

load_filename = "jumps_20210302_075538.h5"


def single_gaussian(x, m, s):
    return np.exp(-(x - m)**2 / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


def multi_gaussian(x, m_arr, s_arr, w_arr):
    xx = np.tile(x, (len(m_arr), 1)).T
    res = w_arr * np.exp(-(xx - m_arr)**2 / (2 * s_arr**2)) / np.sqrt(2 * np.pi * s_arr**2)
    return np.sum(res, axis=-1)


def error_f(p, x, y):
    m = p[0::3]
    s = p[1::3]
    w = p[2::3]
    y_est = multi_gaussian(x, m, s, w)
    return y - y_est


def load(load_filename):
    with h5py.File(load_filename, "r") as h5f:
        source_code = h5f["source_code"][()]
        df = h5f.attrs["df"]
        dither = h5f.attrs["dither"]
        input_port = h5f.attrs["input_port"]
        output_port = h5f.attrs["output_port"]
        amp = h5f.attrs["amp"]
        freq = h5f.attrs["freq"]
        t_arr = h5f["t_arr"][()]
        resp_arr = h5f["resp_arr"][()]

    fig = plt.figure(figsize=(12.8, 4.8), tight_layout=True)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 3, sharex=ax1)
    ax3 = fig.add_subplot(1, 2, 2)
    ax1.plot(1e3 * t_arr, np.abs(resp_arr))
    ax2.plot(1e3 * t_arr, np.angle(resp_arr))
    ax1.set_ylabel("Amplitude [FS]")
    ax2.set_ylabel("phase [rad]")
    ax2.set_xlabel("Time [ms]")
    for tick in ax1.get_xticklabels():
        tick.set_visible(False)

    amps = np.abs(resp_arr)

    # Make and plot histogram
    hist, bins, patches = ax3.hist(amps, bins=256, density=False)
    bin_width = bins[1] - bins[0]
    scale = len(amps) * bin_width
    ax3.set_yscale("log")
    ax3.set_xlabel("Amplitude [FS]")
    ax3.set_ylabel("Counts")
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()

    # Estimate parameters
    data_min = amps.min()
    data_max = amps.max()
    data_middle = 0.5 * (data_min + data_max)
    data_low = amps[amps < data_middle]
    data_high = amps[amps >= data_middle]
    init = np.array([
        np.mean(data_low), np.std(data_low), len(data_low) / len(amps),
        np.mean(data_high), np.std(data_high), len(data_high) / len(amps),
    ])

    # Fit the histogram
    x_data = bins[:-1] + bin_width / 2  # centers of the bins
    pfit, cov = leastsq(error_f, init, args=(x_data, hist / scale))
    m_fit = np.atleast_1d(pfit[0::3])
    s_fit = np.atleast_1d(pfit[1::3])
    w_fit = np.atleast_1d(pfit[2::3])

    # Plot fit
    ax3.autoscale(False)
    ax3.plot(x_data, scale * w_fit[0] * single_gaussian(x_data, m_fit[0], s_fit[0]), label=f"{w_fit[0]:.1%}")
    ax3.plot(x_data, scale * w_fit[1] * single_gaussian(x_data, m_fit[1], s_fit[1]), label=f"{w_fit[1]:.1%}")
    ax3.legend()

    fig.show()


if __name__ == "__main__":
    fig = load(load_filename)
