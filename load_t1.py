import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import untwist_downconversion

rcParams['figure.dpi'] = 108.8

load_filename = "t1_20210306_201400.h5"


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
        freq_if = h5f.attrs["freq_if"]
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

    # Analyze T1
    data_I = store_arr[:, 0, 0, idx]  # (nr_delays, nr_samples)
    data_Q = store_arr[:, 0, 1, idx]  # (nr_delays, nr_samples)
    data_I_fft = np.fft.rfft(data_I) / nr_samples
    data_Q_fft = np.fft.rfft(data_Q) / nr_samples
    data_L_fft, data_H_fft = untwist_downconversion(data_I_fft, data_Q_fft)
    resp_arr = data_L_fft[:, n_if]  # (nr_delays,)
    delay_arr = dt_delays * np.arange(nr_delays)

    # Fit data
    popt_a, perr_a = fit_simple(delay_arr, np.abs(resp_arr))
    popt_p, perr_p = fit_simple(delay_arr, np.angle(resp_arr))
    popt_x, perr_x = fit_simple(delay_arr, np.real(resp_arr))
    popt_y, perr_y = fit_simple(delay_arr, np.imag(resp_arr))

    T1 = popt_p[0]
    T1_err = perr_p[0]
    print("T1 time: {} +- {} us".format(1e6 * T1, 1e6 * T1_err))

    fig2, ax2 = plt.subplots(4, 1, sharex=True, figsize=(6.4, 6.4), tight_layout=True)
    ax21, ax22, ax23, ax24 = ax2
    ax21.plot(1e9 * delay_arr, np.abs(resp_arr))
    ax21.plot(1e9 * delay_arr, decay(delay_arr, *popt_a), '--')
    ax22.plot(1e9 * delay_arr, np.angle(resp_arr))
    ax22.plot(1e9 * delay_arr, decay(delay_arr, *popt_p), '--')
    ax23.plot(1e9 * delay_arr, np.real(resp_arr))
    ax23.plot(1e9 * delay_arr, decay(delay_arr, *popt_x), '--')
    ax24.plot(1e9 * delay_arr, np.imag(resp_arr))
    ax24.plot(1e9 * delay_arr, decay(delay_arr, *popt_y), '--')

    ax21.set_ylabel("Amplitude [FS]")
    ax22.set_ylabel("Phase [rad]")
    ax23.set_ylabel("I [FS]")
    ax24.set_ylabel("Q [FS]")
    ax2[-1].set_xlabel("Control-readout delay [ns]")
    fig2.show()

    return fig1, fig2


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
    fig1, fig2 = load(load_filename)
