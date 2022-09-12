# -*- coding: utf-8 -*-
"""
Loader for files saved by two_tone_pulsed_coupler_bias.py
"""
import os
import sys

import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

from presto.utils import rotate_opt

rcParams["figure.dpi"] = 108.8

if len(sys.argv) == 2:
    load_filename = sys.argv[1]
    print(f"Loading: {os.path.realpath(load_filename)}")
else:
    load_filename = None

LOGSCALE = False  # Plot response in logarithmic scale (dBFS), both in colormap and line cut.
LINECUT = False  # Plot an horizontal line cut of the 2D sweep. Interactive.
BLIT = True  # Use blitting when plotting. Faster if it works.
BIAS_IDX = 0  # internal use


def load(load_filename):
    with h5py.File(load_filename, "r") as h5f:
        num_averages = h5f.attrs["num_averages"]
        readout_freq = h5f.attrs["readout_freq"]
        readout_duration = h5f.attrs["readout_duration"]
        control_duration = h5f.attrs["control_duration"]
        readout_amp = h5f.attrs["readout_amp"]
        control_amp = h5f.attrs["control_amp"]
        sample_duration = h5f.attrs["sample_duration"]
        nr_bias = h5f.attrs["nr_bias"]
        nr_freqs = h5f.attrs["nr_freqs"]
        wait_delay = h5f.attrs["wait_delay"]
        readout_sample_delay = h5f.attrs["readout_sample_delay"]
        coupler_bias_arr = h5f["coupler_bias_arr"][()]
        control_freq_arr = h5f["control_freq_arr"][()]
        t_arr = h5f["t_arr"][()]
        store_arr = h5f["store_arr"][()]

    t_low = 1500 * 1e-9
    t_high = 2000 * 1e-9
    t_span = t_high - t_low
    idx_low = np.argmin(np.abs(t_arr - t_low))
    idx_high = np.argmin(np.abs(t_arr - t_high))
    idx = np.arange(idx_low, idx_high)
    nr_samples = len(idx)

    # # Plot raw store data for first iteration as a check
    # fig0, ax0 = plt.subplots(2, 1, sharex=True, tight_layout=True)
    # ax01, ax02 = ax0
    # ax01.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    # ax02.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
    # ax01.plot(1e9 * t_arr, np.abs(store_arr[0, 0, :]))
    # ax02.plot(1e9 * t_arr, np.angle(store_arr[0, 0, :]))
    # ax02.set_xlabel("Time [ns]")
    # fig0.show()

    # Analyze
    resp_arr = np.mean(store_arr[:, :, idx], axis=-1)
    # data = rotate_opt(resp_arr)

    # Plot 2D sweep
    global BIAS_IDX
    BIAS_IDX = nr_bias // 2

    if LOGSCALE:
        data = 20.0 * np.log10(np.abs(resp_arr))
    else:
        data = np.abs(resp_arr)
        data_max = data.max()
        unit = ""
        if data_max < 1e-6:
            unit = "n"
            data *= 1e9
        elif data_max < 1e-3:
            unit = "μ"
            data *= 1e6
        elif data_max < 1e0:
            unit = "m"
            data *= 1e3

    # choose limits for colorbar
    cutoff = 1.0  # %
    lowlim = np.percentile(data, cutoff)
    highlim = np.percentile(data, 100.0 - cutoff)

    # extent
    x_min = 1e-9 * control_freq_arr[0]
    x_max = 1e-9 * control_freq_arr[-1]
    dx = 1e-9 * (control_freq_arr[1] - control_freq_arr[0])
    y_min = coupler_bias_arr[0]
    y_max = coupler_bias_arr[-1]
    dy = coupler_bias_arr[1] - coupler_bias_arr[0]

    if LINECUT:
        fig1 = plt.figure(tight_layout=True, figsize=(6.4, 9.6))
        ax1 = fig1.add_subplot(2, 1, 1)
    else:
        fig1 = plt.figure(tight_layout=True, figsize=(6.4, 4.8))
        ax1 = fig1.add_subplot(1, 1, 1)
    im = ax1.imshow(
        data,
        origin="lower",
        aspect="auto",
        interpolation="none",
        extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
        vmin=lowlim,
        vmax=highlim,
    )
    if LINECUT:
        line_sel = ax1.axhline(coupler_bias_arr[BIAS_IDX], ls="--", c="k", lw=3, animated=BLIT)
    # ax1.set_title(f"Probe frequency: {readout_freq/1e9:.2f} GHz")
    ax1.set_title("Qubit 2")
    ax1.set_xlabel("Control frequency [GHz]")
    ax1.set_ylabel("Coupler bias [V]")
    cb = fig1.colorbar(im)
    if LOGSCALE:
        cb.set_label("Response amplitude [dBFS]")
    else:
        cb.set_label(f"Response amplitude [{unit:s}FS]")

    if LINECUT:
        ax2 = fig1.add_subplot(4, 1, 3)
        ax3 = fig1.add_subplot(4, 1, 4, sharex=ax2)

        (line_a,) = ax2.plot(1e-9 * control_freq_arr, data[BIAS_IDX], animated=BLIT)
        (line_fit_a,) = ax2.plot(
            1e-9 * control_freq_arr, np.full_like(control_freq_arr, np.nan), ls="--", animated=BLIT
        )
        (line_p,) = ax3.plot(1e-9 * control_freq_arr, np.angle(resp_arr[BIAS_IDX]), animated=BLIT)
        (line_fit_p,) = ax3.plot(
            1e-9 * control_freq_arr, np.full_like(control_freq_arr, np.nan), ls="--", animated=BLIT
        )

        f_min = 1e-9 * control_freq_arr.min()
        f_max = 1e-9 * control_freq_arr.max()
        f_rng = f_max - f_min
        a_min = data.min()
        a_max = data.max()
        a_rng = a_max - a_min
        p_min = -np.pi
        p_max = np.pi
        p_rng = p_max - p_min
        ax2.set_xlim(f_min - 0.05 * f_rng, f_max + 0.05 * f_rng)
        ax2.set_ylim(a_min - 0.05 * a_rng, a_max + 0.05 * a_rng)
        ax3.set_xlim(f_min - 0.05 * f_rng, f_max + 0.05 * f_rng)
        ax3.set_ylim(p_min - 0.05 * p_rng, p_max + 0.05 * p_rng)

        ax3.set_xlabel("Frequency [GHz]")
        if LOGSCALE:
            ax2.set_ylabel("Response amplitude [dB]")
        else:
            ax2.set_ylabel(f"Response amplitude [{unit:s}FS]")
        ax3.set_ylabel("Response phase [rad]")

        def onbuttonpress(event):
            if event.inaxes == ax1:
                global BIAS_IDX
                BIAS_IDX = np.argmin(np.abs(coupler_bias_arr - event.ydata))
                update()

        def onkeypress(event):
            global BIAS_IDX
            if event.inaxes == ax1:
                if event.key == "up":
                    BIAS_IDX += 1
                    if BIAS_IDX >= len(coupler_bias_arr):
                        BIAS_IDX = len(coupler_bias_arr) - 1
                    update()
                elif event.key == "down":
                    BIAS_IDX -= 1
                    if BIAS_IDX < 0:
                        BIAS_IDX = 0
                    update()

        def update():
            global BIAS_IDX
            line_sel.set_ydata([coupler_bias_arr[BIAS_IDX], coupler_bias_arr[BIAS_IDX]])
            # ax1.set_title(f"amp = {amp_arr[BIAS_IDX]:.2e}")
            print(
                f"drive amp {BIAS_IDX:d}: {qubit_amp_arr[BIAS_IDX]:.2e} FS = {coupler_bias_arr[BIAS_IDX]:.1f} dBFS"
            )
            line_a.set_ydata(data[BIAS_IDX])
            line_p.set_ydata(np.angle(resp_arr[BIAS_IDX]))
            line_fit_a.set_ydata(np.full_like(control_freq_arr, np.nan))
            line_fit_p.set_ydata(np.full_like(control_freq_arr, np.nan))
            # ax2.set_title("")
            if BLIT:
                global bg
                fig1.canvas.restore_region(bg)
                ax1.draw_artist(line_sel)
                ax2.draw_artist(line_a)
                ax3.draw_artist(line_p)
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
        ax3.draw_artist(line_p)
        fig1.canvas.blit(fig1.bbox)

    return fig1


if __name__ == "__main__":
    fig1 = load(load_filename)
