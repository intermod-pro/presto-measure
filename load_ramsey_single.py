# -*- coding: utf-8 -*-
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
        control_if = h5f.attrs["control_if"]
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

        print(f"Control frequency: {control_freq / 1e9:.2f} GHz")

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

    # Analyze T2
    resp_arr = np.mean(store_arr[:, 0, idx], axis=-1)
    data = rotate_opt(resp_arr)
    delay_arr = dt_delays * np.arange(nr_delays)

    # Fit data to I quadrature
    try:
        popt, perr, p0 = fit_simple(delay_arr, np.real(data))

        T2 = popt[2]
        T2_err = perr[2]
        print("T2 time: {} +- {} us".format(1e6 * T2, 1e6 * T2_err))
        det = popt[3]
        det_err = perr[3]
        print("detuning: {} +- {} Hz".format(det, det_err))
        sign = 1.0 if np.abs(popt[4]) < np.pi / 2 else -1.0
        print(f"sign: {sign}")
        i_at_e = popt[0] + sign * popt[1]
        i_at_g = popt[0] - sign * popt[1]
        print(f"|e>: {i_at_e} rad")
        print(f"|g>: {i_at_g} rad")

        success = True
    except Exception as err:
        print("Unable to fit data!")
        print(err)
        success = False

    fig2, ax2 = plt.subplots(4, 1, sharex=True, figsize=(6.4, 6.4), tight_layout=True)
    ax21, ax22, ax23, ax24 = ax2
    ax21.plot(1e6 * delay_arr, np.abs(data))
    ax22.plot(1e6 * delay_arr, np.unwrap(np.angle(data)))
    ax23.plot(1e6 * delay_arr, np.real(data))
    if success:
        ax23.plot(1e6 * delay_arr, func(delay_arr, *popt), '--')
    ax24.plot(1e6 * delay_arr, np.imag(data))

    ax21.set_ylabel("Amplitude [FS]")
    ax22.set_ylabel("Phase [rad]")
    ax23.set_ylabel("I [FS]")
    ax24.set_ylabel("Q [FS]")
    ax2[-1].set_xlabel("Ramsey delay [us]")
    fig2.show()

    data_max = np.abs(data.real).max()
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
    ax3.plot(1e6 * delay_arr, mult * np.real(data), '.')
    ax3.set_ylabel(f"I quadrature [{unit:s}FS]")
    ax3.set_xlabel("Ramsey delay [μs]")
    if success:
        ax3.plot(1e6 * delay_arr, mult * func(delay_arr, *p0), '--')
        ax3.plot(1e6 * delay_arr, mult * func(delay_arr, *popt), '--')
        ax3.set_title(f"T2* = {1e6*T2:.0f} ± {1e6*T2_err:.0f} μs")
    fig3.show()

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
    fft[0] = 0
    idx_max = np.argmax(np.abs(fft))
    frequency = freqs[idx_max]
    # first = (y[0] - offset) / amplitude
    # if first > 1.:
    #     first = 1.
    # elif first < -1.:
    #     first = -1.
    # phase = np.arccos(first)
    phase = np.angle(fft[idx_max])
    p0 = (
        offset,
        amplitude,
        T2,
        frequency,
        phase,
    )
    print(p0)
    popt, pcov = curve_fit(
        func,
        x,
        y,
        p0=p0,
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, p0


if __name__ == "__main__":
    fig1, fig2, fig3 = load(load_filename)
