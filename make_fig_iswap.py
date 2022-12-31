import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from presto.utils import rotate_opt

plt.rcParams["figure.autolayout"] = False  # tight_layout
plt.rcParams["figure.constrained_layout.use"] = True  # constrained_layout

# plt.rcParams['figure.figsize'] = [6.69, 2.77]  # double column, 1/3 page
# plt.rcParams['figure.figsize'] = [6.69, 5.02]  # double column, 4:3
# plt.rcParams['figure.figsize'] = [3.37, 2.53]  # single column, 4:3
plt.rcParams["figure.figsize"] = [3.37, 2.70]  # single column, 5:4
# plt.rcParams['figure.figsize'] = [3.37, 2.81]  # single column, 6:5
# plt.rcParams['figure.figsize'] = [3.37, 3.37]  # single column, 1:1

plt.rcParams["legend.borderaxespad"] = 0.5 / 2
plt.rcParams["legend.borderpad"] = 0.4 / 2
plt.rcParams["legend.fontsize"] = 8.0
plt.rcParams["legend.handleheight"] = 0.7 / 2
plt.rcParams["legend.handlelength"] = 2.0 / 1
plt.rcParams["legend.handletextpad"] = 0.8 / 2
plt.rcParams["legend.labelspacing"] = 0.5 / 4
plt.rcParams["legend.title_fontsize"] = 8.0

plt.rcParams["axes.labelsize"] = 10.0
plt.rcParams["axes.labelpad"] = 4.0 / 4
plt.rcParams["axes.titlesize"] = 10.0
plt.rcParams["xtick.labelsize"] = 8.0
plt.rcParams["ytick.labelsize"] = 8.0
plt.rcParams["ytick.major.pad"] = 3.5 / 4

plt.rcParams["figure.dpi"] = 108.8  # my screen
plt.rcParams["savefig.dpi"] = 300.0  # RSI
plt.rcParams["savefig.format"] = "eps"

load_filename = "data/coupler_f_d_20211021_090322.h5"


def _func(t, offset, amplitude, period, phase):
    w = 2 * np.pi / period
    return offset + amplitude * np.cos(w * t + phase)


def _fit(x, y):
    pkpk = np.max(y) - np.min(y)
    offset = np.min(y) + pkpk / 2
    amplitude = 0.5 * pkpk
    freqs = np.fft.rfftfreq(len(x), x[1] - x[0])
    fft = np.fft.rfft(y)
    frequency = freqs[1 + np.argmax(np.abs(fft[1:]))]
    period = 1 / frequency
    first = (y[0] - offset) / amplitude
    if first > 1.0:
        first = 1.0
    elif first < -1.0:
        first = -1.0
    phase = np.arccos(first)
    p0 = (
        offset,
        amplitude,
        period,
        phase,
    )
    res = curve_fit(_func, x, y, p0=p0)
    popt = res[0]
    pcov = res[1]
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def rescale(data, min_, max_):
    rng = max_ - min_
    data = data - min_  # make copy
    data /= rng
    return data


if __name__ == "__main__":
    with h5py.File(load_filename, "r") as h5f:
        num_averages = h5f.attrs["num_averages"]
        control_freq_1 = h5f.attrs["control_freq_1"]
        control_freq_2 = h5f.attrs["control_freq_2"]
        control_if = h5f.attrs["control_if"]
        readout_freq_1 = h5f.attrs["readout_freq_1"]
        readout_freq_2 = h5f.attrs["readout_freq_2"]
        readout_duration = h5f.attrs["readout_duration"]
        control_duration = h5f.attrs["control_duration"]
        readout_amp = h5f.attrs["readout_amp"]
        control_amp_1 = h5f.attrs["control_amp_1"]
        control_amp_2 = h5f.attrs["control_amp_2"]
        sample_duration = h5f.attrs["sample_duration"]
        wait_delay = h5f.attrs["wait_delay"]
        readout_sample_delay = h5f.attrs["readout_sample_delay"]
        coupler_dc_bias = h5f.attrs["coupler_dc_bias"]
        nr_freqs = h5f.attrs["nr_freqs"]
        nr_steps = h5f.attrs["nr_steps"]
        dt_steps = h5f.attrs["dt_steps"]
        coupler_ac_amp = h5f.attrs["coupler_ac_amp"]
        t_arr = np.array(h5f["t_arr"])
        store_arr = np.array(h5f["store_arr"])
        coupler_ac_freq_arr = np.array(h5f["coupler_ac_freq_arr"])
        coupler_ac_duration_arr = np.array(h5f["coupler_ac_duration_arr"])
        readout_if_1 = h5f.attrs["readout_if_1"]
        readout_if_2 = h5f.attrs["readout_if_2"]
        readout_nco = h5f.attrs["readout_nco"]

    t_low = 1500 * 1e-9
    t_high = 2000 * 1e-9
    idx_low = np.argmin(np.abs(t_arr - t_low))
    idx_high = np.argmin(np.abs(t_arr - t_high))
    idx = np.arange(idx_low, idx_high)

    # multiplexed readout
    dt = t_arr[1] - t_arr[0]
    nr_samples = len(idx)
    freq_arr = np.fft.fftfreq(nr_samples, dt)
    # complex FFT should take care of upper/lower sideband
    resp_fft = np.fft.fft(store_arr[:, 0, idx], axis=-1) / len(idx)
    idx_1 = np.argmin(np.abs(freq_arr - readout_if_1))
    idx_2 = np.argmin(np.abs(freq_arr - readout_if_2))
    resp_arr_1 = 2 * resp_fft[:, idx_1]
    resp_arr_2 = 2 * resp_fft[:, idx_2]
    data_1 = np.real(rotate_opt(resp_arr_1))
    data_2 = np.real(rotate_opt(resp_arr_2))
    data_1.shape = (nr_freqs, nr_steps)
    data_2.shape = (nr_freqs, nr_steps)

    # convert to population
    if abs(np.min(data_1)) > abs(np.max(data_1)):
        data_1 *= -1
    if abs(np.min(data_2)) > abs(np.max(data_2)):
        data_2 *= -1

    # Line cut at the center frequency
    idx_cut = np.argmax(np.std(data_1, axis=-1))

    popt1, perr1 = _fit(coupler_ac_duration_arr, data_1[idx_cut, :])
    popt2, perr2 = _fit(coupler_ac_duration_arr, data_2[idx_cut, :])

    xg1 = popt1[0] - popt1[1]
    xe1 = popt1[0] + popt1[1]
    xg2 = popt2[0] - popt2[1]
    xe2 = popt2[0] + popt2[1]

    data_1 = rescale(data_1, xg1, xe1)
    data_2 = rescale(data_2, xg2, xe2)
    # fit again (could rescale instead)
    popt1, perr1 = _fit(coupler_ac_duration_arr, data_1[idx_cut, :])
    popt2, perr2 = _fit(coupler_ac_duration_arr, data_2[idx_cut, :])

    # choose limits for colorbar
    cutoff = 0.1  # %
    lowlim_1 = np.percentile(data_1, cutoff)
    highlim_1 = np.percentile(data_1, 100.0 - cutoff)
    lowlim_2 = np.percentile(data_2, cutoff)
    highlim_2 = np.percentile(data_2, 100.0 - cutoff)
    # alldata = (np.r_[data_1, data_2]).ravel()
    # lowlim = np.percentile(alldata, cutoff)
    # highlim = np.percentile(alldata, 100. - cutoff)
    # lowlim_1, highlim_1 = lowlim, highlim
    # lowlim_2, highlim_2 = lowlim, highlim

    # extent
    x_min = 1e9 * coupler_ac_duration_arr[0]
    x_max = 1e9 * coupler_ac_duration_arr[-1]
    dx = 1e9 * (coupler_ac_duration_arr[1] - coupler_ac_duration_arr[0])
    y_min = 1e-6 * coupler_ac_freq_arr[0]
    y_max = 1e-6 * coupler_ac_freq_arr[-1]
    dy = 1e-6 * (coupler_ac_freq_arr[1] - coupler_ac_freq_arr[0])

    fig = plt.figure()
    gs0 = fig.add_gridspec(3, 1)
    gs1 = gs0[:-1].subgridspec(1, 2)
    ax1, ax2 = gs1.subplots(sharex=True, sharey=True)
    ax3 = fig.add_subplot(gs0[-1])

    im1 = ax1.imshow(
        data_1,
        origin="lower",
        aspect="auto",
        interpolation="none",
        extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
        vmin=lowlim_1,
        vmax=highlim_1,
    )
    im2 = ax2.imshow(
        data_2,
        origin="lower",
        aspect="auto",
        interpolation="none",
        extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
        vmin=lowlim_2,
        vmax=highlim_2,
    )

    ax1.axhline(1e-6 * coupler_ac_freq_arr[idx_cut], c="tab:blue")
    ax2.axhline(1e-6 * coupler_ac_freq_arr[idx_cut], c="tab:orange")

    ax1.set_title("Qubit 1")
    ax2.set_title("Qubit 2")
    ax1.set_xlabel("Duration [ns]")
    ax2.set_xlabel("Duration [ns]")
    ax1.set_ylabel("Frequency [MHz]")
    ax1.set_xticks([0, 300, 600])
    ax1.set_yticks([530, 534, 538])
    cb = fig.colorbar(im1, ax=[ax1, ax2])
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_label("$P_\\mathrm{{e}}$")

    ax3.plot(1e9 * coupler_ac_duration_arr, data_1[idx_cut, :], ".", c="0.75", label="qubit 1")
    ax3.plot(1e9 * coupler_ac_duration_arr, data_2[idx_cut, :], ".", c="0.75", label="qubit 2")
    ax3.plot(1e9 * coupler_ac_duration_arr, _func(coupler_ac_duration_arr, *popt1))
    ax3.plot(1e9 * coupler_ac_duration_arr, _func(coupler_ac_duration_arr, *popt2))
    ax3.set_xlabel("Duration [ns]")
    ax3.set_ylabel("$P_\\mathrm{{e}}$")

    fig.text(0.00, 1.02, "(a)", fontsize=10, ha="center", va="bottom", transform=ax1.transAxes)
    fig.text(0.00, 1.02, "(b)", fontsize=10, ha="center", va="bottom", transform=ax2.transAxes)
    fig.text(0.00, 1.02, "(c)", fontsize=10, ha="right", va="bottom", transform=ax3.transAxes)

    # fig.savefig("iswap")
    fig.show()
