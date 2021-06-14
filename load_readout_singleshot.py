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
from scipy.special import erf

from presto.utils import rotate_opt

rcParams['figure.dpi'] = 108.8

if len(sys.argv) == 2:
    load_filename = sys.argv[1]
    print(f"Loading: {os.path.realpath(load_filename)}")
else:
    load_filename = None


def inprod(f, g, t=None, dt=None):
    if t is not None:
        dt = t[1] - t[0]
        ns = len(t)
        T = ns * dt
    elif dt is not None:
        ns = len(f)
        T = ns * dt
    else:
        T = 1.
    return np.trapz(f * np.conj(g), x=t) / T


def norm(x, t=None, dt=None):

    return np.sqrt(np.real(inprod(x, x, t=t, dt=dt)))


def single_gaussian(x, m, s, w):
    return w * np.exp(-(x - m)**2 / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


def double_gaussian(x, m0, s0, w0, m1, s1, w1):
    return single_gaussian(x, m0, s0, w0) + single_gaussian(x, m1, s1, w1)


def hist_plot(ax, spec, bin_ar, **kwargs):
    x_quad = np.zeros((len(bin_ar) - 1) * 4)  # length 4*N
    # bin_ar[0:-1] takes all but the rightmost element of bin_ar -> length N
    x_quad[::4] = bin_ar[0:-1]  # for lower left,  (x,0)
    x_quad[1::4] = bin_ar[0:-1]  # for upper left,  (x,y)
    x_quad[2::4] = bin_ar[1:]  # for upper right, (x+1,y)
    x_quad[3::4] = bin_ar[1:]  # for lower right, (x+1,0)

    y_quad = np.zeros(len(spec) * 4)
    y_quad[1::4] = spec  # for upper left,  (x,y)
    y_quad[2::4] = spec  # for upper right, (x+1,y)

    return ax.plot(x_quad, y_quad, **kwargs)


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
        match_t_in_store = h5f.attrs["match_t_in_store"]
        t_arr = h5f["t_arr"][()]
        store_arr = h5f["store_arr"][()]
        match_g_data = h5f["match_g_data"][()]
        match_e_data = h5f["match_e_data"][()]
        template_g = h5f["template_g"][()]
        template_e = h5f["template_e"][()]
        source_code = h5f["source_code"][()]

    nr_samples = len(t_arr)
    t_span = nr_samples * (t_arr[1] - t_arr[0])
    match_idx = np.argmin(np.abs(t_arr - match_t_in_store))
    match_len = len(template_g)
    t_low = t_arr[match_idx]
    t_high = t_arr[match_idx + match_len]

    # Plot raw store data for first iteration as a check
    fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
    ax11, ax12 = ax1
    ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    ax12.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    ax11.plot(1e9 * t_arr, np.abs(store_arr[0, 0, :]))
    ax11.plot(1e9 * t_arr, np.abs(store_arr[1, 0, :]))
    ax12.plot(1e9 * t_arr, np.angle(store_arr[0, 0, :]))
    ax12.plot(1e9 * t_arr, np.angle(store_arr[1, 0, :]))
    ax12.set_xlabel("Time [ns]")
    fig1.show()

    # # Analyze
    threshold = 0.5 * (norm(template_e)**2 - norm(template_g)**2)
    match_diff = match_e_data - match_g_data - threshold  # does |e> match better than |g>?
    match_diff_g = match_diff[0::2]  # qubit was prepared in |g>
    match_diff_e = match_diff[1::2]  # qubit was prepared in |e>
    mean_g = match_diff_g.mean()
    mean_e = match_diff_e.mean()
    std_g = match_diff_g.std()
    std_e = match_diff_e.std()
    std = max(std_g, std_e)
    x_min = min(mean_g, mean_e) - 5 * std
    x_max = max(mean_g, mean_e) + 5 * std
    H_g, xedges = np.histogram(match_diff_g,
                               bins=100,
                               range=(x_min, x_max),
                               density=True)
    H_e, xedges = np.histogram(match_diff_e,
                               bins=100,
                               range=(x_min, x_max),
                               density=True)
    xdata = 0.5 * (xedges[1:] + xedges[:-1])
    # z_max = max(H_g.max(), H_e.max())

    init_g = np.array([mean_g, std_g, 0.9, mean_e, std_e, 0.1])
    init_e = np.array([mean_g, std_g, 0.1, mean_e, std_e, 0.9])
    popt_g, pcov_g = curve_fit(double_gaussian, xdata, H_g, p0=init_g)
    popt_e, pcov_e = curve_fit(double_gaussian, xdata, H_e, p0=init_e)
    fidelity_g = 0.5 * (1 + erf((0.0 - popt_g[0]) / np.sqrt(2 * popt_g[1]**2)))
    fidelity_e = 1.0 - 0.5 * (1 + erf(
        (0.0 - popt_e[3]) / np.sqrt(2 * popt_e[4]**2)))

    fig2, ax2 = plt.subplots(1,
                             2,
                             sharex=True,
                             sharey=True,
                             tight_layout=True,
                             figsize=(12.8, 4.8))
    ax21, ax22 = ax2
    for ax_ in ax2:
        ax_.axvline(0.0, c="tab:gray", alpha=0.25)
        ax_.axhline(0.0, c="tab:gray", alpha=0.25)

    hist_plot(ax21, H_g, xedges, lw=1)
    ax21.plot(xdata, double_gaussian(xdata, *popt_g), c="k")
    ax21.plot(xdata,
              single_gaussian(xdata, *popt_g[:3]),
              ls="--",
              label=f"$\\left|\\mathrm{{g}}\\right>$: {popt_g[2]:.1%}")
    ax21.plot(xdata,
              single_gaussian(xdata, *popt_g[3:]),
              ls="--",
              label=f"$\\left|\\mathrm{{e}}\\right>$: {popt_g[5]:.1%}")
    ax21.set_xlabel("Comparator result")
    ax21.set_title(
        f"Qubit prepared in $\\left|\\mathrm{{g}}\\right>$: $\\mathcal{{F}}$ = {fidelity_g:.1%}"
    )
    ax21.legend(title="Qubit measured in")

    hist_plot(ax22, H_e, xedges, lw=1)
    ax22.plot(xdata, double_gaussian(xdata, *popt_e), c="k")
    ax22.plot(xdata,
              single_gaussian(xdata, *popt_e[:3]),
              ls="--",
              label=f"$\\left|\\mathrm{{g}}\\right>$: {popt_e[2]:.1%}")
    ax22.plot(xdata,
              single_gaussian(xdata, *popt_e[3:]),
              ls="--",
              label=f"$\\left|\\mathrm{{e}}\\right>$: {popt_e[5]:.1%}")
    ax22.set_xlabel("Comparator result")
    ax22.set_title(
        f"Qubit prepared in $\\left|\\mathrm{{e}}\\right>$: $\\mathcal{{F}}$ = {fidelity_e:.1%}"
    )
    ax22.legend(title="Qubit measured in")

    fig2.show()

    print(popt_g)
    print(popt_e)

    return fig1, fig2


if __name__ == "__main__":
    fig1, fig2 = load(load_filename)
