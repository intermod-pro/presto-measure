import h5py
import matplotlib.pyplot as plt
import numpy as np

from presto.utils import to_pm_pi

plt.rcParams["figure.autolayout"] = False  # tight_layout
plt.rcParams["figure.constrained_layout.use"] = True  # constrained_layout

# plt.rcParams['figure.figsize'] = [3.37, 2.53]  # single column, 4:3
plt.rcParams["figure.figsize"] = [3.37, 3.37]  # single column, 1:1

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

load_filename = "data/Q1_ReadoutJPA_20220321_182305.h5"  # 1400 ns


def _rotate_opt(trace_g, trace_e):
    """Rotate reference traces so that all difference is in the I quadrature."""
    data = trace_e - trace_g  # complex distance

    # calculate the mean of data.imag**2 in steps of 1 deg
    N = 360
    _mean = np.zeros(N)
    for ii in range(N):
        _data = data * np.exp(1j * 2 * np.pi / N * ii)
        _mean[ii] = np.mean(_data.imag**2)

    # the mean goes like cos(x)**2
    # FFT and find the phase at frequency "2"
    fft = np.fft.rfft(_mean) / N
    # first solution
    x_fft1 = -np.angle(fft[2])  # compensate for measured phase
    x_fft1 -= np.pi  # we want to be at the zero of cos(2x)
    x_fft1 /= 2  # move from frequency "2" to "1"
    # there's a second solution np.pi away (a minus sign)
    x_fft2 = x_fft1 + np.pi

    # convert to +/- interval
    x_fft1 = to_pm_pi(x_fft1)
    x_fft2 = to_pm_pi(x_fft2)
    # choose the closest to zero
    if np.abs(x_fft1) < np.abs(x_fft2):
        x_fft = x_fft1
    else:
        x_fft = x_fft2

    # rotate the data and return a copy
    trace_g = trace_g * np.exp(1j * x_fft)
    trace_e = trace_e * np.exp(1j * x_fft)

    return trace_g, trace_e


with h5py.File(load_filename, "r") as h5f:
    ampR1 = complex(h5f.attrs["ampR1"])
    ampR2 = complex(h5f.attrs["ampR2"])
    ampR4 = complex(h5f.attrs["ampR4"])
    ampR5 = complex(h5f.attrs["ampR5"])
    durationR1 = int(h5f.attrs["durationR1"])
    durationR2 = int(h5f.attrs["durationR2"])
    durationR4 = int(h5f.attrs["durationR4"])
    durationR5 = int(h5f.attrs["durationR5"])
    t_arr = np.array(h5f["t_arr"])
    data = np.array(h5f["data"])

t_arr = t_arr[0:2_000]
data_g = data[0, 0, 0:2_000]
data_e = data[1, 0, 0:2_000]
data_g, data_e = _rotate_opt(data_g, data_e)
distance = np.abs(data_e - data_g)

nr_samples = len(t_arr)
match_len = 1022  # ns

max_idx = 0
max_dist = 0.0
idx = -2
while idx + 2 + match_len <= nr_samples:
    idx += 2  # next clock cycle
    dist = np.sum(distance[idx : idx + match_len])
    if dist > max_dist:
        max_dist = dist
        max_idx = idx

drive_arr = np.zeros_like(t_arr)
idx = 1
for (_d, _a) in zip(
    [durationR1, durationR2, durationR4, durationR5], [ampR1, ampR2, ampR4, ampR5]
):
    drive_arr[idx : idx + _d] = np.real(_a)
    idx += _d

fig, ax = plt.subplots(3, 1, sharex=True)
ax1, ax2, ax3 = ax
for _ax in ax:
    _ax.axhline(0.0, c="#bfbfbf")
    _ax.axvspan(1e6 * t_arr[max_idx], 1e6 * t_arr[max_idx + match_len], facecolor="#e7e7e7")

scale = np.max(np.abs(drive_arr))
ax1.plot(t_arr * 1e6, drive_arr / scale, c="tab:red")
ax1.set_ylabel("Output")

# scale = max(np.max(np.abs(data_g.real)), np.max(np.abs(data_e.real)))
scale = 1.0
ax2.plot(t_arr * 1e6, data_g.real / scale, c="tab:blue", label=r"$\left| \mathrm{g} \right>$")
ax2.plot(t_arr * 1e6, data_e.real / scale, c="tab:orange", label=r"$\left| \mathrm{e} \right>$")
ax2.legend(loc="lower right", ncol=2)
ax2.set_ylabel("Input")

# scale = np.max(distance)
scale = 1.0
ax3.plot(t_arr * 1e6, distance / scale, "tab:green")
ax3.set_yticks([0, 1])
ax3.set_xticks([0, 1, 2])
ax3.set_xticks([0.5, 1.5], minor=True)
ax3.set_ylabel("Separation")
ax3.set_xlabel("Time [µs]")

ax1.text(0.98, 0.95, "(a)", fontsize=10, va="top", ha="right", transform=ax1.transAxes)
ax2.text(0.98, 0.95, "(b)", fontsize=10, va="top", ha="right", transform=ax2.transAxes)
ax3.text(0.98, 0.95, "(c)", fontsize=10, va="top", ha="right", transform=ax3.transAxes)

# fig.savefig("reference_calibration")
fig.show()

fig2, ax2 = plt.subplots()
ax2.plot(data_g.real, data_g.imag, c="tab:blue")
ax2.plot(data_e.real, data_e.imag, c="tab:orange")
fig2.show()
