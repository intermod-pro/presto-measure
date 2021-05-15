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
        readout_freq = h5f.attrs["readout_freq"]
        readout_duration = h5f.attrs["readout_duration"]
        control_duration = h5f.attrs["control_duration"]
        readout_amp = h5f.attrs["readout_amp"]
        control_amp = h5f.attrs["control_amp"]
        sample_duration = h5f.attrs["sample_duration"]
        nr_delays = h5f.attrs["nr_delays"]
        dt_delays = h5f.attrs["dt_delays"]
        wait_delay = h5f.attrs["wait_delay"]
        readout_sample_delay = h5f.attrs["readout_sample_delay"]
        t_arr = h5f["t_arr"][()]
        store_arr = h5f["store_arr"][()]
        source_code = h5f["source_code"][()]

    t_low = 1500 * 1e-9
    t_high = 2000 * 1e-9
    t_span = t_high - t_low
    idx_low = np.argmin(np.abs(t_arr - t_low))
    idx_high = np.argmin(np.abs(t_arr - t_high))
    idx = np.arange(idx_low, idx_high)
    nr_samples = len(idx)

    # Plot raw store data for first iteration as a check
    fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
    ax11, ax12 = ax1
    ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    ax12.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    ax11.plot(1e9 * t_arr, np.abs(store_arr[0, 0, :]))
    ax12.plot(1e9 * t_arr, np.angle(store_arr[0, 0, :]))
    ax12.set_xlabel("Time [ns]")
    fig1.show()

    # Analyze T1
    resp_arr = np.mean(store_arr[:, 0, idx], axis=-1)
    resp_arr = rotate_opt(resp_arr)
    delay_arr = dt_delays * np.arange(nr_delays)

    # Fit data
    popt, perr = fit_simple(delay_arr, np.real(resp_arr))

    T1 = popt[0]
    T1_err = perr[0]
    print("T1 time I: {} +- {} us".format(1e6 * T1, 1e6 * T1_err))

    fig2, ax2 = plt.subplots(4, 1, sharex=True, figsize=(6.4, 6.4), tight_layout=True)
    ax21, ax22, ax23, ax24 = ax2
    ax21.plot(1e6 * delay_arr, np.abs(resp_arr))
    ax22.plot(1e6 * delay_arr, np.unwrap(np.angle(resp_arr)))
    ax23.plot(1e6 * delay_arr, np.real(resp_arr))
    ax23.plot(1e6 * delay_arr, decay(delay_arr, *popt), '--')
    ax24.plot(1e6 * delay_arr, np.imag(resp_arr))

    ax21.set_ylabel("Amplitude [FS]")
    ax22.set_ylabel("Phase [rad]")
    ax23.set_ylabel("I [FS]")
    ax24.set_ylabel("Q [FS]")
    ax2[-1].set_xlabel("Control-readout delay [us]")
    fig2.show()

    # bigger plot just for I quadrature
    data_max = np.abs(resp_arr.real).max()
    unit = ""
    mult = 1.0
    if data_max < 1e-6:
        unit = "n"
        mult = 1e9
    elif data_max < 1e-3:
        unit = "μ"
        mult = 1e6
    elif data_max < 1e0:
        unit = "m"
        mult = 1e3

    fig3, ax3 = plt.subplots(tight_layout=True)
    ax3.plot(1e6 * delay_arr, mult * np.real(resp_arr), '.')
    ax3.plot(1e6 * delay_arr, mult * decay(delay_arr, *popt), '--')
    ax3.set_ylabel(f"I quadrature [{unit:s}FS]")
    ax3.set_xlabel(r"Control-readout delay [$\mathrm{\mu}$s]")
    ax3.set_title("T1 = {:.0f} +- {:.0f} us".format(1e6 * T1, 1e6 * T1_err))
    fig3.show()

    return fig1, fig2, fig3


def decay(t, *p):
    T1, xe, xg = p
    return xg + (xe - xg) * np.exp(-t / T1)


def fit_simple(t, x):
    T1 = 0.5 * (t[-1] - t[0])
    xe, xg = x[0], x[-1]
    p0 = (T1, xe, xg)
    popt, pcov = curve_fit(decay, t, x, p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


if __name__ == "__main__":
    fig1, fig2, fig3 = load(load_filename)
