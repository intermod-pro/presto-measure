# -*- coding: utf-8 -*-
import os
import sys

import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

from presto.utils import rotate_opt

rcParams["figure.dpi"] = 108.8

if len(sys.argv) == 2:
    load_filename = sys.argv[1]
    print(f"Loading: {os.path.realpath(load_filename)}")
else:
    load_filename = None

FIXED = True  # set to True to force sum of weight to 1.0


def inprod(f, g, t=None, dt=None):
    if t is not None:
        dt = t[1] - t[0]
        ns = len(t)
        T = ns * dt
    elif dt is not None:
        ns = len(f)
        T = ns * dt
    else:
        T = 1.0
    return np.trapz(f * np.conj(g), x=t) / T


def norm(x, t=None, dt=None):
    return np.sqrt(np.real(inprod(x, x, t=t, dt=dt)))


def single_gaussian(x, m, s, w):
    return w * np.exp(-((x - m) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


def double_gaussian(x, m0, s0, w0, m1, s1, w1):
    return single_gaussian(x, m0, s0, w0) + single_gaussian(x, m1, s1, w1)


def double_gaussian_fixed(x, m0, s0, w0, m1, s1):
    w1 = 1.0 - w0
    return double_gaussian(x, m0, s0, w0, m1, s1, w1)


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
    threshold = 0.5 * (norm(template_e) ** 2 - norm(template_g) ** 2)
    match_diff = match_e_data - match_g_data - threshold  # does |e> match better than |g>?
    match_diff_g = match_diff[0::2]  # qubit was prepared in |g>
    match_diff_e = match_diff[1::2]  # qubit was prepared in |e>
    idx_low_g = match_diff_g < 0
    idx_high_g = np.logical_not(idx_low_g)
    idx_low_e = match_diff_e < 0
    idx_high_e = np.logical_not(idx_low_e)
    mean_low_g = match_diff_g[idx_low_g].mean()
    mean_high_g = match_diff_g[idx_high_g].mean()
    mean_low_e = match_diff_e[idx_low_e].mean()
    mean_high_e = match_diff_e[idx_high_e].mean()
    std_low_g = match_diff_g[idx_low_g].std()
    std_high_g = match_diff_g[idx_high_g].std()
    std_low_e = match_diff_e[idx_low_e].std()
    std_high_e = match_diff_e[idx_high_e].std()
    weight_low_g = np.sum(idx_low_g) / len(idx_low_g)
    weight_high_g = 1.0 - weight_low_g
    weight_low_e = np.sum(idx_low_e) / len(idx_low_e)
    weight_high_e = 1.0 - weight_low_e
    std = max(std_low_g, std_high_g, std_low_e, std_high_e)
    x_min = min(mean_low_g, mean_low_e) - 5 * std
    x_max = max(mean_high_g, mean_high_e) + 5 * std

    H_g, xedges = np.histogram(match_diff_g, bins=100, range=(x_min, x_max), density=True)
    H_e, xedges = np.histogram(match_diff_e, bins=100, range=(x_min, x_max), density=True)
    xdata = 0.5 * (xedges[1:] + xedges[:-1])

    init_g = np.array(
        [mean_low_g, std_low_g, weight_low_g, mean_high_g, std_high_g, weight_high_g]
    )
    init_e = np.array(
        [mean_low_e, std_low_e, weight_low_e, mean_high_e, std_high_e, weight_high_e]
    )
    if FIXED:
        # skip second weight
        popt_g, pcov_g = curve_fit(double_gaussian_fixed, xdata, H_g, p0=init_g[:-1])
        popt_e, pcov_e = curve_fit(double_gaussian_fixed, xdata, H_e, p0=init_e[:-1])
        # add back second weight for ease of use
        popt_g = np.r_[popt_g, 1.0 - popt_g[2]]
        popt_e = np.r_[popt_e, 1.0 - popt_e[2]]
    else:
        popt_g, pcov_g = curve_fit(double_gaussian, xdata, H_g, p0=init_g)
        popt_e, pcov_e = curve_fit(double_gaussian, xdata, H_e, p0=init_e)
    fidelity_g = 0.5 * (1 + erf((0.0 - popt_g[0]) / np.sqrt(2 * popt_g[1] ** 2)))
    fidelity_e = 1.0 - 0.5 * (1 + erf((0.0 - popt_e[3]) / np.sqrt(2 * popt_e[4] ** 2)))

    fig2, ax2 = plt.subplots(
        1, 2, sharex=True, sharey=True, tight_layout=True, figsize=(12.8, 4.8)
    )
    ax21, ax22 = ax2
    for ax_ in ax2:
        ax_.axvline(0.0, c="tab:gray", alpha=0.25)
        ax_.axhline(0.0, c="tab:gray", alpha=0.25)

    hist_plot(ax21, H_g, xedges, lw=1)
    ax21.plot(xdata, double_gaussian(xdata, *popt_g), c="k")
    ax21.plot(
        xdata,
        single_gaussian(xdata, *popt_g[:3]),
        ls="--",
        label=f"$\\left|\\mathrm{{g}}\\right>$: {popt_g[2]:.1%}",
    )
    ax21.plot(
        xdata,
        single_gaussian(xdata, *popt_g[3:]),
        ls="--",
        label=f"$\\left|\\mathrm{{e}}\\right>$: {popt_g[5]:.1%}",
    )
    ax21.set_xlabel("Comparator result")
    ax21.set_title(
        f"Qubit prepared in $\\left|\\mathrm{{g}}\\right>$: $\\mathcal{{F}}$ = {fidelity_g:.1%}"
    )
    ax21.legend(title="Qubit measured in")

    hist_plot(ax22, H_e, xedges, lw=1)
    ax22.plot(xdata, double_gaussian(xdata, *popt_e), c="k")
    ax22.plot(
        xdata,
        single_gaussian(xdata, *popt_e[:3]),
        ls="--",
        label=f"$\\left|\\mathrm{{g}}\\right>$: {popt_e[2]:.1%}",
    )
    ax22.plot(
        xdata,
        single_gaussian(xdata, *popt_e[3:]),
        ls="--",
        label=f"$\\left|\\mathrm{{e}}\\right>$: {popt_e[5]:.1%}",
    )
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
