import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import remez, filtfilt

from presto.utils import rotate_opt

plt.rcParams["figure.autolayout"] = False  # tight_layout
plt.rcParams["figure.constrained_layout.use"] = True  # constrained_layout

# plt.rcParams['figure.figsize'] = [6.69, 2.77]  # double column, 1/3 page
# plt.rcParams['figure.figsize'] = [6.69, 5.02]  # double column, 4:3
plt.rcParams["figure.figsize"] = [3.37, 2.53]  # single column, 4:3

plt.rcParams["legend.borderaxespad"] = 0.5 / 2
plt.rcParams["legend.borderpad"] = 0.4 / 2
plt.rcParams["legend.fontsize"] = 8.0
plt.rcParams["legend.handleheight"] = 0.7 / 2
plt.rcParams["legend.handlelength"] = 2.0 / 1
plt.rcParams["legend.handletextpad"] = 0.8 / 2
plt.rcParams["legend.labelspacing"] = 0.5 / 4

plt.rcParams["axes.labelsize"] = 10.0
plt.rcParams["axes.labelpad"] = 4.0 / 4
plt.rcParams["xtick.labelsize"] = 8.0
plt.rcParams["ytick.labelsize"] = 8.0
plt.rcParams["ytick.major.pad"] = 3.5 / 4

plt.rcParams["figure.dpi"] = 108.8  # my screen
plt.rcParams["savefig.dpi"] = 300.0  # RSI
plt.rcParams["savefig.format"] = "eps"

load_filename = "data/rb_20220208_002442.h5"


def lowpass(s):
    # b = firwin(256, 2e6, fs=1e9, pass_zero=True)
    b = remez(256, [0, 4e6, 5e6, 0.5 * 1e9], [1, 0], Hz=1e9)
    # w, h = freqz(b, fs=1e9)
    # plt.plot(w, 20*np.log10(np.abs(h)))
    # plt.show()
    return filtfilt(b, 1, s)


def exp_fit_fn(x, A, B, C):
    return B + (A - B) * C**x


def rescale(data, min_, max_):
    rng = max_ - min_
    data = data - min_  # make copy
    data /= rng
    return 0.942 * data


if __name__ == "__main__":
    with h5py.File(load_filename, "r") as h5f:
        rb_lengths = np.array(h5f["rb_lengths"])
        t_arr = np.array(h5f["t_arr"])
        result = np.array(h5f["store_arr"])
        control_envelope = np.array(h5f["control_envelope"])

    result = lowpass(result)

    range_ = (t_arr >= 1.54e-6) & (t_arr < 2.14e-6)
    result_average = np.average(result[:, :, 0, 0, range_], axis=-1)
    rotated = np.real(rotate_opt(result_average))
    rotated_avg = np.average(rotated, axis=0)
    rotated_std = np.std(rotated, axis=0)
    rotated_75 = np.percentile(rotated, 75, axis=0)
    rotated_50 = np.percentile(rotated, 50, axis=0)
    rotated_25 = np.percentile(rotated, 25, axis=0)

    popt, pcov = curve_fit(
        exp_fit_fn, rb_lengths, rotated_avg, p0=(rotated_avg[0], rotated_avg[-1], 0.99)
    )
    A, B, C = popt
    perr = np.sqrt(np.diag(pcov))
    X = np.linspace(rb_lengths[0], rb_lengths[-1], 1000)
    X2 = np.linspace(0, 8192, 1000)
    print(popt)
    alpha = popt[-1]
    alpha_std = perr[-1]
    alpha_rel = alpha_std / alpha
    r = (1 - alpha) / 2
    r_rel = alpha_rel
    r_std = r * r_rel
    print(f"EPC: {r:e} +/- {r_std:e}")

    fid = 1.0 - r
    fid_std = r_std
    print(f"F: {fid} +/- {fid_std:e}")
    fid_label = f"$\\mathcal{{F}} = {100*fid:.3f}\\%$"

    xg = A
    xe = B - 2 * (A - B)

    fig, ax = plt.subplots()
    for d in rotated:
        (line,) = ax.plot(rb_lengths, rescale(d, xe, xg), ".", c="0.75")
    line.set_label("single realizations")

    ax.semilogx(rb_lengths, rescale(rotated_avg, xe, xg), ".", ms=9, label="average")
    ax.semilogx(X2, rescale(exp_fit_fn(X2, *popt), xe, xg), "--", label=fid_label)
    ax.legend(loc="lower left")
    f = ScalarFormatter()
    f.set_scientific(False)
    ax.xaxis.set_major_formatter(f)

    ax.set_xlim(6.6e-1, 6.2e3)
    ax.set_ylim(0.58, 1.02)
    ax.set_yticks([0.6, 0.8, 1.0])

    ax.set_xlabel("Sequence length")
    ax.set_ylabel("$P_\\mathrm{g}$")

    # fig.savefig("randomized_benchmarking")
    fig.show()

    center = rescale(rotated_50, xe, xg)
    higher = rescale(rotated_75, xe, xg)
    lower = rescale(rotated_25, xe, xg)
    # median = rescale(rotated_avg, xe, xg)
    # higher = rescale(rotated_avg + rotated_std, xe, xg)
    # lower = rescale(rotated_avg - rotated_std, xe, xg)
    err_h = higher - center
    err_l = center - lower
    err = [err_l, err_h]

    fig2, ax2 = plt.subplots()
    (line_fit,) = ax2.semilogx(
        X2,
        rescale(exp_fit_fn(X2, *popt), xe, xg),
        "--",
        c="tab:orange",
        label=fid_label,
    )
    eb = ax2.errorbar(
        rb_lengths,
        center,
        yerr=err,
        fmt=".",
        ms=9,
        c="tab:blue",
        ecolor="0.75",
        capsize=1,
    )

    (mock_m,) = ax2.plot([], [], ".", ms=9, c="tab:blue")
    mock_eb = ax2.errorbar(
        [],
        [],
        yerr=[],
        fmt=".",
        ms=0,
        c="tab:blue",
        ecolor="0.75",
        capsize=1,
    )

    ax2.set_xscale("log")
    ax2.set_xlim(6.6e-1, 6.2e3)
    ax2.set_ylim(0.58, 1.02)
    ax2.set_yticks([0.6, 0.8, 1.0])
    ax2.set_xlabel("Sequence length")
    ax2.set_ylabel("$P_\\mathrm{g}$")
    ax2.xaxis.set_major_formatter(f)

    # ax2.legend(loc="lower left")
    ax2.legend(
        [mock_m, mock_eb, line_fit],
        ["median over realizations", "interquartile range", fid_label],
        loc="lower left",
    )

    fig2.savefig("randomized_benchmarking")
    fig2.show()
