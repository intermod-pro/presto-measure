# -*- coding: utf-8 -*-
"""
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
        control_center_if = h5f.attrs["control_center_if"]
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
        control_freq_arr = h5f["control_freq_arr"][()]
        # source_code = h5f["source_code"][()]

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
    nr_freqs = len(control_freq_arr)
    resp_arr.shape = (nr_freqs, nr_delays)
    delay_arr = dt_delays * np.arange(nr_delays)
    data = rotate_opt(resp_arr)
    plot_data = data.real

    data_max = np.abs(plot_data).max()
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
    plot_data *= mult

    # choose limits for colorbar
    cutoff = 0.0  # %
    lowlim = np.percentile(plot_data, cutoff)
    highlim = np.percentile(plot_data, 100. - cutoff)

    # extent
    x_min = 1e+6 * delay_arr[0]
    x_max = 1e+6 * delay_arr[-1]
    dx = 1e+6 * (delay_arr[1] - delay_arr[0])
    y_min = 1e-9 * control_freq_arr[0]
    y_max = 1e-9 * control_freq_arr[-1]
    dy = 1e-9 * (control_freq_arr[1] - control_freq_arr[0])

    fig2, ax2 = plt.subplots(tight_layout=True)
    im = ax2.imshow(
        plot_data,
        origin='lower',
        aspect='auto',
        interpolation='none',
        extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
        vmin=lowlim,
        vmax=highlim,
    )
    ax2.set_xlabel("Ramsey delay [μs]")
    ax2.set_ylabel("Control frequency [GHz]")
    cb = fig2.colorbar(im)
    cb.set_label(f"Response I quadrature [{unit:s}FS]")
    fig2.show()

    fit_freq = np.zeros(nr_freqs)
    for jj in range(nr_freqs):
        try:
            res = fit_simple(delay_arr, plot_data[jj])
            fit_freq[jj] = np.abs(res[3])
        except Exception:
            fit_freq[jj] = np.nan

    n_fit = nr_freqs // 4
    pfit1 = np.polyfit(control_freq_arr[:n_fit], fit_freq[:n_fit], 1)
    pfit2 = np.polyfit(control_freq_arr[-n_fit:], fit_freq[-n_fit:], 1)
    x0 = np.roots(pfit1 - pfit2)[0]
    # x0 = np.roots(pfit1)[0]

    fig3, ax3 = plt.subplots(tight_layout=True)
    ax3.plot(control_freq_arr, fit_freq, '.')
    ax3.set_ylabel("Fitted detuning [Hz]")
    ax3.set_xlabel("Control frequency [Hz]$]")
    fig3.show()
    _lims = ax3.axis()
    ax3.plot(
        control_freq_arr,
        np.polyval(pfit1, control_freq_arr),
        '--',
        c='tab:orange',
    )
    ax3.plot(
        control_freq_arr,
        np.polyval(pfit2, control_freq_arr),
        '--',
        c='tab:green',
    )
    ax3.axhline(0.0, ls='--', c='tab:gray')
    ax3.axvline(x0, ls='--', c='tab:gray')
    ax3.axis(_lims)
    fig3.canvas.draw()
    print(f"Fitted qubit frequency: {x0} Hz")

    return fig1, fig2, fig3


def func(t, offset, amplitude, T2, frequency, phase):
    return offset + amplitude * np.exp(-t / T2) * np.cos(2. * np.pi * frequency * t + phase)


def fit_simple(x, y):
    pkpk = np.max(y) - np.min(y)
    offset = np.min(y) + pkpk / 2
    amplitude = 0.5 * pkpk
    T2 = 0.5 * (np.max(x) - np.min(x))
    freqs = np.fft.rfftfreq(len(x), x[1] - x[0])
    fft = np.fft.rfft(y)
    frequency = freqs[1 + np.argmax(np.abs(fft[1:]))]
    first = (y[0] - offset) / amplitude
    if first > 1.:
        first = 1.
    elif first < -1.:
        first = -1.
    phase = np.arccos(first)
    p0 = (
        offset,
        amplitude,
        T2,
        frequency,
        phase,
    )
    popt, cov = curve_fit(
        func,
        x,
        y,
        p0=p0,
    )
    offset, amplitude, T2, frequency, phase = popt
    return popt


if __name__ == "__main__":
    fig1, fig2, fig3 = load(load_filename)
