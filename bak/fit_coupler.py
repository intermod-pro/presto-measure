import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams["figure.dpi"] = 108.8

# data/sweep_coupler_bias_20240516_134334.h5
# data/sweep_coupler_bias_20240516_133306.h5
# data/two_tone_pulsed_coupler_bias_20240516_111558.h5
# data/two_tone_pulsed_coupler_bias_20240516_130544.h5

xdata = np.array(
    [
        -9.57,
        -8.78,
        -7.59,
        -7.02,
        -4.40,
        -3.90,
        -2.69,
        -1.90,
        4.87,
        5.65,
        6.90,
        7.41,
    ]
)  # V
ydata = np.array(
    [
        6.751,
        6.195,
        4.978,
        4.243,
        4.243,
        4.978,
        6.195,
        6.751,
        6.751,
        6.195,
        4.978,
        4.243,
    ],
)  # GHz

# set f_c in between resonator 2 and qubit 2
sel_freq = 0.5 * (6.195 + 4.978)


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
p0 = [7.4, 0.25, 14.3, 1.35]
pfit, pcov = curve_fit(func, xdata, ydata, p0)
f0, d, x0, os = pfit

xplot = np.linspace(-10, 10, 1001)
yplot = func(xplot, *pfit)

# Choose an operating point
idx_os = np.argmin(np.abs(xplot - os))
# positive segment
idx_high = np.argmin(np.abs(xplot - (0.5 * x0 + os)))
idx_sel_p = np.argmin(np.abs(yplot[idx_os:idx_high] - sel_freq)) + idx_os
sel_bias_p = xplot[idx_sel_p]
# negative segment
idx_low = np.argmin(np.abs(xplot - (-0.5 * x0 + os)))
idx_sel_n = np.argmin(np.abs(yplot[idx_low:idx_os] - sel_freq)) + idx_low
sel_bias_n = xplot[idx_sel_n]
# choose min abs bias
if abs(sel_bias_p) > abs(sel_bias_n):
    sel_bias = sel_bias_n
else:
    sel_bias = sel_bias_p
sel_r = (sel_bias - os) / x0

fig, ax = plt.subplots(tight_layout=True)

# ax.axvline(0.15 * x0 + os, c="tab:gray", ls="--")
# ax.axvline(0.30 * x0 + os, c="tab:brown", ls="--")

ax.axhline(sel_freq, c="tab:gray", ls="--")
ax.axvline(sel_bias, c="tab:gray", ls="--")

ax.plot(xdata, ydata, ".")
ax.plot(xplot, yplot, "--")
ax.set_xlabel(r"Coupler DC bias $V_\mathrm{c}$ [V]")
ax.set_ylabel(r"Coupler frequency $f_\mathrm{c}$ [GHz]")

# ax.text(0.55, 0.6, r"$0.15 \Phi_0$", ha='center', rotation="vertical", transform=ax.transAxes)
# ax.text(0.66, 0.6, r"$0.30 \Phi_0$", ha='center', rotation="vertical", transform=ax.transAxes)
ax.text(sel_bias, sel_freq, f"{sel_r:.2f}" + r"$ \Phi_0$", ha="center", rotation="vertical")

x_pos = 0.18
ax.text(
    x_pos,
    0.90,
    r"$f_\mathrm{c} = f_0 \sqrt{\cos^2\phi_\mathrm{e} + d \sin^2\phi_\mathrm{e}}$",
    horizontalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    x_pos,
    0.80,
    r"$\phi_\mathrm{e} = \pi \frac{V_\mathrm{c} - V_\mathrm{OS}}{V_0}$",
    horizontalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    x_pos,
    0.70,
    r"$f_0 = $" + f"{pfit[0]:.2f} GHz",
    horizontalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    x_pos,
    0.65,
    r"$V_0 = $" + f"{pfit[2]:.1f} V",
    horizontalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    x_pos,
    0.60,
    r"$V_\mathrm{os} = $" + f"{pfit[3]:.2f} V",
    horizontalalignment="center",
    transform=ax.transAxes,
)
ax.text(
    x_pos, 0.55, r"$d = $" + f"{pfit[1]:.3f}", horizontalalignment="center", transform=ax.transAxes
)

# fig.savefig("data/fit_coupler.png", dpi=300)
fig.show()
