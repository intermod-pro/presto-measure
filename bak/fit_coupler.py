import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams["figure.dpi"] = 108.8

xdata = np.array(
    [
        -3.866,
        -0.440,
        0.511,
        3.909,
        -3.708,
        -0.607,
        0.669,
        3.751,
        -2.589,
        -1.726,
        1.779,
        2.642,
        -2.756,
        -1.585,
        1.620,
        2.774,
    ]
)  # V
ydata = np.repeat(
    [
        6.166_333,
        6.028_275,
        3.557_866,
        4.091_777,
    ],
    4,
)  # GHz


def func(x, f0, d, x0, os):
    # global x0
    # global os
    xe = np.pi * (x - os) / x0
    # return f0 * np.abs(np.cos(xe))
    return f0 * np.sqrt(np.cos(xe) ** 2 + d * np.sin(xe) ** 2)


# p0 = [7.0, 2 * np.mean(np.abs(xdata)), np.mean(xdata)]
# global os
# os = np.mean(xdata)
# global x0
# x0 = 2 * np.mean(np.abs(xdata))
p0 = [6.5, 0.25, 4.4, 0.024]
pfit, pcov = curve_fit(func, xdata, ydata, p0)
f0, d, x0, os = pfit

xplot = np.linspace(-4.5, 4.5, 1001)
yplot = func(xplot, *pfit)

# Choose an operating point
# set f_c in between resonator 2 and qubit 2
sel_freq = 0.5 * (6.028_275 + 4.091_777)
# choose just one segment
idx_start = np.argmin(np.abs(xplot - os))
idx_stop = np.argmin(np.abs(xplot - (0.5 * x0 + os)))
idx_sel = np.argmin(np.abs(yplot[idx_start:idx_stop] - sel_freq)) + idx_start
sel_bias = xplot[idx_sel]
sel_r = (sel_bias - os) / x0

fig, ax = plt.subplots(tight_layout=True)

# ax.axvline(0.15 * x0 + os, c="tab:gray", ls="--")
# ax.axvline(0.30 * x0 + os, c="tab:brown", ls="--")

ax.axhline(sel_freq, c="tab:gray", ls="--")
ax.axvline(sel_bias, c="tab:gray", ls="--")

ax.plot(xdata[0:4], ydata[0:4], ".", label="resonator 1")
ax.plot(xdata[4:8], ydata[4:8], ".", label="resonator 2")
ax.plot(xdata[8:12], ydata[8:12], ".", label="qubit 1")
ax.plot(xdata[12:16], ydata[12:16], ".", label="qubit 2")
ax.plot(xplot, yplot, "--", label="fit")
ax.set_xlabel(r"Coupler DC bias $V_\mathrm{c}$ [V]")
ax.set_ylabel(r"Coupler frequency $f_\mathrm{c}$ [GHz]")
ax.legend()

# ax.text(0.55, 0.6, r"$0.15 \Phi_0$", ha='center', rotation="vertical", transform=ax.transAxes)
# ax.text(0.66, 0.6, r"$0.30 \Phi_0$", ha='center', rotation="vertical", transform=ax.transAxes)
ax.text(sel_bias, sel_freq, f"{sel_r:.2f}" + r"$ \Phi_0$", ha="center", rotation="vertical")

ax.text(
    0.28,
    0.90,
    r"$f_\mathrm{c} = f_0 \sqrt{\cos^2\phi_\mathrm{e} + d \sin^2\phi_\mathrm{e}}$",
    horizontalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    0.28,
    0.80,
    r"$\phi_\mathrm{e} = \pi \frac{V_\mathrm{c} - V_\mathrm{OS}}{V_0}$",
    horizontalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    0.28,
    0.70,
    r"$f_0 = $" + f"{pfit[0]:.2f} GHz",
    horizontalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    0.28,
    0.65,
    r"$V_0 = $" + f"{pfit[2]:.2f} V",
    horizontalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    0.28,
    0.60,
    r"$V_\mathrm{os} = $" + f"{1e3*pfit[3]:.1f} mV",
    horizontalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    0.28, 0.55, r"$d = $" + f"{pfit[1]:.3f}", horizontalalignment="center", transform=ax.transAxes
)

# fig.savefig("data/fit_coupler.png", dpi=300)
fig.show()
