# -*- coding: utf-8 -*-
"""
Loader for files saved by jpa_bias_sweep_w_pump.py
"""
import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

rcParams["figure.dpi"] = 108.8

DIVERGENT = False  # use a divergent color map
CUTOFF = 0.5  # throw away 2*CUTOFF% of the data
LINECUT = False  # Plot an horizontal line cut of the 2D sweep. Interactive.
BLIT = True  # Use blitting when plotting. Faster if it works.
BIAS_IDX = 0  # internal use

load_filename = "data/jpa_bias_sweep_w_pump_20210315_151830.h5"
load_filename = "data/jpa_bias_sweep_w_pump_20210315_204118.h5"


def load(load_filename):
    with h5py.File(load_filename, "r") as h5f:
        df = h5f.attrs["df"]
        Navg = h5f.attrs["Navg"]
        amp = h5f.attrs["amp"]
        dither = h5f.attrs["dither"]
        input_port = h5f.attrs["input_port"]
        output_port = h5f.attrs["output_port"]
        bias_port = h5f.attrs["bias_port"]
        pump_freq = h5f.attrs["pump_freq"]
        pump_pwr = h5f.attrs["pump_pwr"]
        freq_arr = h5f["freq_arr"][()]
        bias_arr = h5f["bias_arr"][()]
        ref_arr = h5f["ref_arr"][()]
        resp_arr = h5f["resp_arr"][()]
        source_code = h5f["source_code"][()]

    ref_db = 20 * np.log10(np.abs(ref_arr))
    data_db = 20 * np.log10(np.abs(resp_arr))
    gain_db = data_db - ref_db
    idx_pump = np.argmin(np.abs(freq_arr - pump_freq / 2))
    gain_db[:, idx_pump] = np.nan

    low = np.nanpercentile(gain_db, CUTOFF)
    high = np.nanpercentile(gain_db, 100 - CUTOFF)
    if DIVERGENT:
        lim = max(abs(low), abs(high))
        vmin = -lim
        vmax = +lim
    else:
        vmin = 0.0
        vmax = high

    # signal frequency
    freq_arr *= 1e-9  # GHz

    # current bias
    rs = 1e3 + 50  # ohm, series resistance
    bias_arr /= rs  # A
    bias_arr *= 1e6  # uA

    global BIAS_IDX
    BIAS_IDX = len(bias_arr) // 2

    # extent
    x_min = freq_arr[0]
    x_max = freq_arr[-1]
    dx = freq_arr[1] - freq_arr[0]
    y_min = bias_arr[0]
    y_max = bias_arr[-1]
    dy = bias_arr[1] - bias_arr[0]

    if LINECUT:
        fig1 = plt.figure(tight_layout=True, figsize=(6.4, 7.2))
        ax1 = fig1.add_subplot(2, 1, 1)
    else:
        fig1 = plt.figure(tight_layout=True, figsize=(6.4, 4.8))
        ax1 = fig1.add_subplot(1, 1, 1)
    im = ax1.imshow(
        gain_db,
        origin="lower",
        aspect="auto",
        extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r" if DIVERGENT else "viridis",
    )
    if LINECUT:
        line_sel = ax1.axhline(bias_arr[BIAS_IDX], ls="--", c="k", lw=3, animated=BLIT)
    cb = fig1.colorbar(im)
    cb.set_label("Signal gain [dB]")
    ax1.set_xlabel("Signal frequency [GHz]")
    ax1.set_ylabel("JPA bias [μA]")
    ax1.set_title(f"JPA pump at {1e-9 * pump_freq:.1f} GHz")

    if LINECUT:
        ax2 = fig1.add_subplot(3, 1, 3)

        (line_a,) = ax2.plot(freq_arr, gain_db[BIAS_IDX], animated=BLIT)

        f_min = freq_arr.min()
        f_max = freq_arr.max()
        f_rng = f_max - f_min
        a_min = np.nanmin(gain_db)
        a_max = np.nanmax(gain_db)
        a_rng = a_max - a_min
        ax2.set_xlim(f_min - 0.05 * f_rng, f_max + 0.05 * f_rng)
        ax2.set_ylim(a_min - 0.05 * a_rng, a_max + 0.05 * a_rng)
        ax2.set_xlabel("Frequency [GHz]")
        ax2.set_ylabel("Signal gain [dB]")

        def onbuttonpress(event):
            if event.inaxes == ax1:
                global BIAS_IDX
                BIAS_IDX = np.argmin(np.abs(bias_arr - event.ydata))
                update()

        def onkeypress(event):
            global BIAS_IDX
            if event.inaxes == ax1:
                if event.key == "up":
                    BIAS_IDX += 1
                    if BIAS_IDX >= len(bias_arr):
                        BIAS_IDX = len(bias_arr) - 1
                    update()
                elif event.key == "down":
                    BIAS_IDX -= 1
                    if BIAS_IDX < 0:
                        BIAS_IDX = 0
                    update()

        def update():
            global BIAS_IDX
            line_sel.set_ydata([bias_arr[BIAS_IDX], bias_arr[BIAS_IDX]])
            # ax1.set_title(f"amp = {amp_arr[BIAS_IDX]:.2e}")
            print(f"bias {BIAS_IDX:d}: {bias_arr[BIAS_IDX]:.2e} uA")
            line_a.set_ydata(gain_db[BIAS_IDX])
            if BLIT:
                global bg
                fig1.canvas.restore_region(bg)
                ax1.draw_artist(line_sel)
                ax2.draw_artist(line_a)
                fig1.canvas.blit(fig1.bbox)
                # fig1.canvas.flush_events()
            else:
                fig1.canvas.draw()

        fig1.canvas.mpl_connect("button_press_event", onbuttonpress)
        fig1.canvas.mpl_connect("key_press_event", onkeypress)

    fig1.show()
    if LINECUT and BLIT:
        fig1.canvas.draw()
        # fig1.canvas.flush_events()
        global bg
        bg = fig1.canvas.copy_from_bbox(fig1.bbox)
        ax1.draw_artist(line_sel)
        ax2.draw_artist(line_a)
        fig1.canvas.blit(fig1.bbox)

    return fig1


if __name__ == "__main__":
    fig1 = load(load_filename)
