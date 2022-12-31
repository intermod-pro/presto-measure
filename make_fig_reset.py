import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann, Planck
from scipy.optimize import curve_fit
from scipy.special import erf

LOGSCALE = True
# EXCITED = True

plt.rcParams["figure.autolayout"] = False  # tight_layout
plt.rcParams["figure.constrained_layout.use"] = True  # constrained_layout

# plt.rcParams['figure.figsize'] = [6.69, 2.77]  # double column, 1/3 page
# plt.rcParams['figure.figsize'] = [6.69, 5.02]  # double column, 4:3
# plt.rcParams['figure.figsize'] = [3.37, 2.53]  # single column, 4:3
# plt.rcParams["figure.figsize"] = [3.37, 3.37]  # single column, 1:1
plt.rcParams["figure.figsize"] = [3.37, 4.49]  # single column, 3:4

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
plt.rcParams["xtick.labelsize"] = 8.0
plt.rcParams["ytick.labelsize"] = 8.0
plt.rcParams["ytick.major.pad"] = 3.5 / 4

plt.rcParams["figure.dpi"] = 108.8  # my screen
plt.rcParams["savefig.dpi"] = 300.0  # RSI
plt.rcParams["savefig.format"] = "eps"

load_filename = "data/Q1_Reset_20211214_144413.h5"
# load_filename = "data/Q1_Reset_20220304_214213.h5"

FIXED = True  # set to True to force sum of weight to 1.0


def single_gaussian(x, m, s, w):
    return w * np.exp(-((x - m) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


def double_gaussian(x, m0, s0, w0, m1, s1, w1):
    return single_gaussian(x, m0, s0, w0) + single_gaussian(x, m1, s1, w1)


def double_gaussian_fixed(x, m0, s0, w0, m1, s1):
    w1 = 1.0 - w0
    return double_gaussian(x, m0, s0, w0, m1, s1, w1)


def transparent(rgb, alpha):
    r = (rgb >> 16) & 0xFF
    g = (rgb >> 8) & 0xFF
    b = rgb & 0xFF
    new_r = int(round(0xFF * alpha + r * (1 - alpha)))
    new_g = int(round(0xFF * alpha + g * (1 - alpha)))
    new_b = int(round(0xFF * alpha + b * (1 - alpha)))
    new_rgb = new_b
    new_rgb |= new_g << 8
    new_rgb |= new_r << 16
    return new_rgb


def error(m, s):
    x = abs(m) / (np.sqrt(2) * s)
    # if m < 0.0:
    #     return 0.5 * (1 + erf(x))
    # else:
    return 0.5 * (1 - erf(x))


def t_eff(p, fq):
    return Planck * fq / (Boltzmann * np.log(1 / p - 1))


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


if __name__ == "__main__":
    with h5py.File(load_filename, "r") as h5f:
        match_g_data = np.array(h5f["result_match_g"])
        match_e_data = np.array(h5f["result_match_e"])
        threshold = float(h5f.attrs["th"])
        control_freq = float(h5f.attrs["freqQ"])

    match_diff = (
        match_e_data + match_g_data - threshold
    )  # does |e> match better than |g>? NOTE match_g already has minus sign
    match_diff.shape = (
        -1,
        2,
        2,
    )  # -1: repetitions; 2: prepared in |g> or |e>; 2: 1st and 2nd readout

    # 1 before reset, 2 reset to |g>, 3 reset to |e>
    _match_diff_1 = match_diff[:, 0, 0]  # prepared in |g>, first readout (before reset)
    _match_diff_2 = match_diff[:, 0, 1]  # prepared in |g>, second readout (after reset to |g>)
    _match_diff_3 = match_diff[:, 1, 1]  # prepared in |e>, second readout (after reset to |e>)
    match_diff_all = (_match_diff_1, _match_diff_2, _match_diff_3)

    idx_low = [None, None, None]
    idx_high = [None, None, None]
    mean_low = [None, None, None]
    mean_high = [None, None, None]
    std_low = [None, None, None]
    std_high = [None, None, None]
    weight_low = [None, None, None]
    weight_high = [None, None, None]

    for which in (0, 1, 2):
        idx_low[which] = match_diff_all[which] < 0
        idx_high[which] = np.logical_not(idx_low[which])
        mean_low[which] = match_diff_all[which][idx_low[which]].mean()
        mean_high[which] = match_diff_all[which][idx_high[which]].mean()
        std_low[which] = match_diff_all[which][idx_low[which]].std()
        std_high[which] = match_diff_all[which][idx_high[which]].std()
        weight_low[which] = np.sum(idx_low[which]) / len(idx_low[which])
        weight_high[which] = 1.0 - weight_low[which]

    std = max(max(std_low), max(std_high))
    x_min = min(mean_low) - 5 * std
    x_max = max(mean_high) + 5 * std
    ntot = match_diff.shape[0]
    nr_bins = int(round(np.sqrt(ntot)))

    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
    # ax1, ax2 = ax
    labels = ["thermal", "reset to |g>", "reset to |e>"]
    for which, ax_ in enumerate(ax):

        H, xedges = np.histogram(
            match_diff_all[which], bins=nr_bins, range=(x_min, x_max), density=True
        )
        xdata = 0.5 * (xedges[1:] + xedges[:-1])
        dx = xdata[1] - xdata[0]
        min_dens = 1.0 / ntot / dx

        init = np.array(
            [
                mean_low[which],
                std_low[which],
                weight_low[which],
                mean_high[which],
                std_high[which],
                weight_high[which],
            ]
        )
        if FIXED:
            # skip second weight
            popt, pcov = curve_fit(double_gaussian_fixed, xdata, H, p0=init[:-1])
            # add back second weight for ease of use
            popt = np.r_[popt, 1.0 - popt[2]]
        else:
            popt, pcov = curve_fit(double_gaussian, xdata, H, p0=init)

        # *** Effective temperature ***
        # Teff_1 = Planck * control_freq / (Boltzmann * np.log(1 / popt_1[5] - 1))
        # Teff_2 = Planck * control_freq / (Boltzmann * np.log(1 / popt_2[5] - 1))
        Teff = t_eff(popt[5], control_freq)

        print(f"Dataset {which+1}:")
        print(f"  |e>: {popt[5]:5.1%}")
        print(f"  T_e: {1e3 * Teff:.1f}mK")

        # *** Readout fidelity ***
        err_eg = error(popt[0], popt[1])  # measure |e> but it was |g>
        err_ge = error(popt[3], popt[4])  # measure |g> but it was |e>
        fidelity = 1.0 - 0.5 * (err_eg + err_ge)
        print(f"  err_eg = {err_eg:.1e}")
        print(f"  err_ge = {err_ge:.1e}")
        print(f"  F = {fidelity:.3%}")
        # fidelity_g = 0.5 * (1 + erf((0.0 - popt_g[0]) / np.sqrt(2 * popt_g[1]**2)))
        # fidelity_e = 1.0 - 0.5 * (1 + erf(
        #     (0.0 - popt_e[3]) / np.sqrt(2 * popt_e[4]**2)))

        ax_.axvline(0.0, c="0.75")

        # hist_color = transparent(0x1f77b4, 0.5)
        # hist_color = f"#{hist_color:06x}"
        hist_color = "0.75"
        hist_plot(ax_, H, xedges, lw=1, c=hist_color)
        # ax1.plot(xdata, double_gaussian(xdata, *popt_1), c="k")
        ax_.plot(
            xdata,
            single_gaussian(xdata, *popt[:3]),
            ls="-",
            c="tab:blue",
            label=f"$\\left|\\mathrm{{g}}\\right>$: {popt[2]:.1%}",
        )
        ax_.plot(
            xdata,
            single_gaussian(xdata, *popt[3:]),
            ls="-",
            c="tab:orange",
            label=f"$\\left|\\mathrm{{e}}\\right>$:   {popt[5]:.1%}",
        )

        # ax1.set_title(f"Before reset: $T_\\mathrm{{eff}}$ = {1e3*Teff_1:.0f} mK")
        ax_.set_ylabel("Norm. counts")
        legend_loc = "upper left" if which == 2 else "upper right"
        ax_.legend(title=labels[which], ncol=1, loc=legend_loc)  # TODO

        if LOGSCALE:
            ax_.set_yscale("log")
            ymin = np.log10(min_dens)
            # ymax = np.log10(max(H_1.max(), H_2.max()))
            ymax = np.log10(H.max())
            yrng = ymax - ymin
            ax_.set_ylim(10 ** (ymin - 0.05 * yrng), 10 ** (ymax + 0.05 * yrng))

        # if EXCITED:
        #     ax1.text(
        #         0.98,
        #         0.95,
        #         "(a)",
        #         fontsize=10,
        #         va="top",
        #         ha="right",
        #         transform=ax1.transAxes,
        #     )
        #     ax2.text(
        #         0.98,
        #         0.95,
        #         "(b)",
        #         fontsize=10,
        #         va="top",
        #         ha="right",
        #         transform=ax2.transAxes,
        #     )
        # else:
        #     ax1.text(0.02, 0.95, "(a)", fontsize=10, va="top", transform=ax1.transAxes)
        #     ax2.text(0.02, 0.95, "(b)", fontsize=10, va="top", transform=ax2.transAxes)

        # fig.savefig("reset")
        # fig.savefig("reset_e.png")

    ax[-1].set_xlabel(
        r"$\left< s, \tau_\mathrm{e} \right> - \left< s, \tau_{g} \right> - \theta_{eg}$"
    )

    ax1, ax2, ax3 = ax
    ax1.text(0.02, 0.95, "(a)", fontsize=10, va="top", transform=ax1.transAxes)
    ax2.text(0.02, 0.95, "(b)", fontsize=10, va="top", transform=ax2.transAxes)
    ax3.text(0.98, 0.95, "(c)", fontsize=10, va="top", ha="right", transform=ax3.transAxes)

    fig.savefig("reset")
    fig.show()
