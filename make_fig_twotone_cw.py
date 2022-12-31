import matplotlib.pyplot as plt
import numpy as np
import cv2

from presto.utils import rotate_opt

import two_tone_power

import set_rc_params

plt.rcParams["figure.figsize"] = [3.37, 2.53]  # single column, 4:3


def main():
    load_filename = "data/two_tone_power_20220422_063812.h5"

    m = two_tone_power.TwoTonePower.load(load_filename)

    assert m.control_freq_arr is not None
    assert m.resp_arr is not None

    nr_amps = len(m.control_amp_arr)

    data = rotate_opt(m.resp_arr).real
    title = "Response quadrature"
    data_max = np.abs(data).max()
    data /= -data_max
    data = cv2.resize(data, dsize=(512, 60), interpolation=cv2.INTER_CUBIC)

    amp_dBFS = 20 * np.log10(m.control_amp_arr / 1.0)

    # choose limits for colorbar
    # cutoff = 0.5  # %
    # lowlim = np.percentile(data, cutoff)
    # highlim = np.percentile(data, 100. - cutoff)
    lowlim, highlim = 0.0, 1.0

    # extent
    x_min = 1e-9 * m.control_freq_arr[0]
    x_max = 1e-9 * m.control_freq_arr[-1]
    # dx = 1e-9 * (m.control_freq_arr[1] - m.control_freq_arr[0])
    dx = (x_max - x_min) / (data.shape[1] - 1)
    y_min = amp_dBFS[0]
    y_max = amp_dBFS[-1]
    dy = amp_dBFS[1] - amp_dBFS[0]

    fig1 = plt.figure()
    gs = fig1.add_gridspec(1, 1)
    ax1 = fig1.add_subplot(gs[0, 0])

    im = ax1.imshow(
        data,
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        interpolation="none",
        extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
        vmin=lowlim,
        vmax=highlim,
    )

    # ax1.set_title(f"Probe frequency: {m.readout_freq/1e9:.2f} GHz")
    ax1.set_xlabel("Pump frequency [GHz]")
    ax1.set_ylabel("Pump power [dBFS]")
    cb = fig1.colorbar(im)
    cb.set_label("Norm. response")

    ax1.set_yticks([-60, -40, -20], minor=True)
    ax1.set_yticks([-50, -30, -10], minor=False)
    cb.set_ticks([0.0, 1.0])
    cb.minorticks_on()

    ax1.text(
        4.087,
        -50.0,
        r"$\left| \mathrm{g} \right> \rightarrow \left| \mathrm{e} \right>$",
        rotation="vertical",
        ha="right",
        va="center",
        c="w",
        fontsize=10.0,
    )
    ax1.text(
        3.864,
        -50.0,
        r"$\left| \mathrm{e} \right> \rightarrow \left| \mathrm{f} \right>$",
        rotation="vertical",
        ha="right",
        va="center",
        c="w",
        fontsize=10.0,
    )
    ax1.text(
        3.975,
        -24.0,
        r"$\left| \mathrm{g} \right> \rightarrow \left| \mathrm{f} \right>$",
        rotation="vertical",
        ha="right",
        va="center",
        c="w",
        fontsize=10.0,
    )
    ax1.text(
        3.855,
        -13.0,
        r"$\left| \mathrm{g} \right> \rightarrow \left| \mathrm{h} \right>$",
        rotation="vertical",
        ha="right",
        va="center",
        c="w",
        fontsize=10.0,
    )

    fig1.savefig("twotone_cw")
    fig1.show()

    return fig1


if __name__ == "__main__":
    main()
