import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import untwist_downconversion

rcParams['figure.dpi'] = 108.8

load_filename = ""


def load(load_filename):
    with h5py.File(load_filename, "r") as h5f:
        num_averages = h5f.attrs["num_averages"]
        control_freq = h5f.attrs["control_freq"]
        readout_freq = h5f.attrs["readout_freq"]
        readout_duration = h5f.attrs["readout_duration"]
        control_duration = h5f.attrs["control_duration"]
        readout_amp = h5f.attrs["readout_amp"]
        sample_duration = h5f.attrs["sample_duration"]
        rabi_n = h5f.attrs["rabi_n"]
        wait_delay = h5f.attrs["wait_delay"]
        readout_sample_delay = h5f.attrs["readout_sample_delay"]
        freq_if = h5f.attrs["freq_if"]
        control_amp_arr = h5f["control_amp_arr"][()]
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
    n_if = int(round(freq_if * t_span))
    dt = t_arr[1] - t_arr[0]
    f_arr = np.fft.rfftfreq(len(t_arr[idx]), dt)

    # Plot raw store data for first iteration as a check
    data_I = store_arr[0, 0, 0, :]
    data_Q = store_arr[0, 0, 1, :]
    data_I_fft = np.fft.rfft(data_I[idx]) / len(t_arr[idx])
    data_Q_fft = np.fft.rfft(data_Q[idx]) / len(t_arr[idx])
    data_L_fft, data_H_fft = untwist_downconversion(data_I_fft, data_Q_fft)

    fig1, ax1 = plt.subplots(2, 1, tight_layout=True)
    ax11, ax12 = ax1
    ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    ax12.set_facecolor("#dfdfdf")
    ax11.plot(1e9 * t_arr, data_I, label="I", c="tab:blue")
    ax11.plot(1e9 * t_arr, data_Q, label="Q", c="tab:orange")
    ax12.semilogy(1e-6 * f_arr, np.abs(data_L_fft), c="tab:green", label="L")
    ax12.semilogy(1e-6 * f_arr, np.abs(data_H_fft), c="tab:red", label="H")
    ax12.semilogy(1e-6 * f_arr[n_if], np.abs(data_L_fft[n_if]), '.', c="tab:green", ms=12)
    ax12.semilogy(1e-6 * f_arr[n_if], np.abs(data_H_fft[n_if]), '.', c="tab:red", ms=12)
    ax11.legend()
    ax12.legend()
    ax11.set_xlabel("Time [ns]")
    ax12.set_xlabel("Frequency [MHz]")
    fig1.show()

    # Analyze Rabi
    data_I = store_arr[:, 0, 0, idx]  # (rabi_n, nr_samples)
    data_Q = store_arr[:, 0, 1, idx]  # (rabi_n, nr_samples)
    data_I_fft = np.fft.rfft(data_I) / nr_samples
    data_Q_fft = np.fft.rfft(data_Q) / nr_samples
    data_L_fft, data_H_fft = untwist_downconversion(data_I_fft, data_Q_fft)
    resp_arr = data_L_fft[:, n_if]  # (rabi_n,)

    # Fit data
    popt_a, perr_a = fit_period(control_amp_arr, np.abs(resp_arr))
    popt_p, perr_p = fit_period(control_amp_arr, np.angle(resp_arr))
    popt_x, perr_x = fit_period(control_amp_arr, np.real(resp_arr))
    popt_y, perr_y = fit_period(control_amp_arr, np.imag(resp_arr))

    period = popt_x[3]
    period_err = perr_x[3]
    pi_amp = period / 2
    pi_2_amp = period / 4
    print("Tau pulse amplitude: {} +- {} FS".format(period, period_err))
    print("Pi pulse amplitude: {:.0f} FS".format(pi_amp))
    print("Pi/2 pulse amplitude: {:.0f} FS".format(pi_2_amp))

    fig2, ax2 = plt.subplots(4, 1, sharex=True, figsize=(6.4, 6.4), tight_layout=True)
    ax21, ax22, ax23, ax24 = ax2
    ax21.plot(control_amp_arr, np.abs(resp_arr))
    ax21.plot(control_amp_arr, func(control_amp_arr, *popt_a), '--')
    ax22.plot(control_amp_arr, np.angle(resp_arr))
    ax22.plot(control_amp_arr, func(control_amp_arr, *popt_p), '--')
    ax23.plot(control_amp_arr, np.real(resp_arr))
    ax23.plot(control_amp_arr, func(control_amp_arr, *popt_x), '--')
    ax24.plot(control_amp_arr, np.imag(resp_arr))
    ax24.plot(control_amp_arr, func(control_amp_arr, *popt_y), '--')

    ax21.set_ylabel("Amplitude [FS]")
    ax22.set_ylabel("Phase [rad]")
    ax23.set_ylabel("I [FS]")
    ax24.set_ylabel("Q [FS]")
    ax2[-1].set_xlabel("Pulse amplitude [FS]")
    fig2.show()

    return fig1, fig2


def func(t, offset, amplitude, T2, period, phase):
    frequency = 1 / period
    return offset + amplitude * np.exp(-t / T2) * np.cos(2. * np.pi * frequency * t + phase)


def fit_period(x, y):
    pkpk = np.max(y) - np.min(y)
    offset = np.min(y) + pkpk / 2
    amplitude = 0.5 * pkpk
    T2 = 0.5 * (np.max(x) - np.min(x))
    freqs = np.fft.rfftfreq(len(x), x[1] - x[0])
    fft = np.fft.rfft(y)
    frequency = freqs[1 + np.argmax(np.abs(fft[1:]))]
    period = 1 / frequency
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
        period,
        phase,
    )
    popt, pcov = curve_fit(func, x, y, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    offset, amplitude, T2, period, phase = popt
    return popt, perr


if __name__ == "__main__":
    fig1, fig2 = load(load_filename)
