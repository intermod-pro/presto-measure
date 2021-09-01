# -*- coding: utf-8 -*-
"""
Loader for files saved by jpa_bias_sweep.py
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
        source_code = h5f["source_code"][()]

    resp_db = 20 * np.log10(np.abs(resp_arr))
    resp_phase = np.unwrap(np.unwrap(np.angle(resp_arr), axis=1), axis=0)
    dw = 2 * np.pi * df
    resp_grpdly = -np.diff(resp_phase, axis=1) / dw
    db = bias_arr[1] - bias_arr[0]
    resp_dpdb = np.diff(resp_phase, axis=0) / db

    # data_plot = resp_grpdly * 1e9
    data_plot = resp_dpdb

    # choose limits for colorbar
    cutoff = 1.  # %
    lowlim = np.percentile(data_plot, cutoff)
    highlim = np.percentile(data_plot, 100. - cutoff)

    # extent
    x_min = 1e-9 * freq_arr[0]
    x_max = 1e-9 * freq_arr[-1]
    dx = 1e-9 * (freq_arr[1] - freq_arr[0])
    y_min = bias_arr[0]
    y_max = bias_arr[-1]
    dy = bias_arr[1] - bias_arr[0]

    fig1, ax1 = plt.subplots(tight_layout=True)
    im = ax1.imshow(
        data_plot,
        origin='lower',
        aspect='auto',
        extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
        vmin=lowlim,
        vmax=highlim,
    )
    ax1.set_xlabel('Frequency [GHz]')
    ax1.set_ylabel('Bias [V]')
    cb = fig1.colorbar(im)
    # cb.set_label("Group delay [ns]")
    cb.set_label(r"$\mathrm{d}\phi / \mathrm{d} V$ [rad / V]")
    fig1.show()

    return fig1


if __name__ == "__main__":
    fig1 = load(load_filename)
