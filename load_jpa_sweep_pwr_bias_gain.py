# -*- coding: utf-8 -*-
"""
Loader for files saved by jpa_sweep_pwr_bias_gain.py
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

rcParams['figure.dpi'] = 108.8

if len(sys.argv) == 2:
    load_filename = sys.argv[1]
    print(f"Loading: {os.path.realpath(load_filename)}")
else:
    load_filename = None


def load(load_filename):
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

    ref_db = 20 * np.log10(np.abs(ref_arr))
    # ref_grpdly = np.diff(np.unwrap(np.angle(ref_plot)))
    data_db = 20 * np.log10(np.abs(resp_arr))
    # data_grpdly = np.diff(np.unwrap(np.angle(data_plot)))

    gain_db = np.zeros_like(data_db)
    for pp in range(nr_pump_pwr):
        for bb in range(nr_bias):
            gain_db[pp, bb, :] = data_db[pp, bb, :] - ref_db[bb, :]

    # choose limits for colorbar
    cutoff = 1.  # %
    lowlim = np.percentile(gain_db, cutoff)
    highlim = np.percentile(gain_db, 100. - cutoff)
    abslim = max(abs(lowlim), abs(highlim))

    # extent
    x_min = 1e-9 * freq_arr[0]
    x_max = 1e-9 * freq_arr[-1]
    dx = 1e-9 * (freq_arr[1] - freq_arr[0])
    y_min = bias_arr[0]
    y_max = bias_arr[-1]
    dy = bias_arr[1] - bias_arr[0]

    fig1, ax1 = plt.subplots(3, 4, sharex=True, sharey=True, tight_layout=True)
    for ii in range(12):
        _ax = ax1[ii // 4][ii % 4]
        im = _ax.imshow(gain_db[ii, :, :],
                        origin='lower',
                        aspect='auto',
                        extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2,
                                y_max + dy / 2),
                        vmin=-abslim,
                        vmax=abslim,
                        cmap="RdBu_r")
        _ax.set_title(str(pump_pwr_arr[ii]))
    fig1.show()

    # bias_idx = np.argmin(np.abs(bias_arr - 0.44))
    # fig, ax = plt.subplots()
    # for pp in range(25, 35):
    #     pwr = pump_pwr_arr[pp]
    #     ax.plot(1e-9 * freq_arr, gain_db[pp, bias_idx, :], label=str(pwr))
    # ax.legend()
    # fig.show()

    # pwr_idx = np.argmin(np.abs(pump_pwr_arr - 7.5))
    # bias_start = np.argmin(np.abs(bias_arr - 0.43))
    # bias_stop = np.argmin(np.abs(bias_arr - 0.45))
    # fig, ax = plt.subplots()
    # for bb in range(bias_start, bias_stop):
    #     bias = bias_arr[bb]
    #     ax.plot(1e-9 * freq_arr, gain_db[pwr_idx, bb, :], label=str(bias))
    # ax.legend()
    # fig.show()

    return fig1


if __name__ == "__main__":
    fig1 = load(load_filename)
