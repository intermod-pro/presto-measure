# -*- coding: utf-8 -*-
"""
Loader for files saved by sweep_power.py
Copyright (C) 2021  Intermodulation Products AB.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
<https://www.gnu.org/licenses/>.
"""
import h5py
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
from resonator_tools import circuit

NORM = True
AMP_IDX = 0
BLIT = False

# load_filename = "sweep_power_20210205_083929.h5"

# load_filename = "sweep_power_20210206_221320.h5"  # 6.025 GHz
# load_filename = "sweep_power_20210207_012855.h5"  # 6.025 GHz, 6425 uA, dither False
# load_filename = "sweep_power_20210207_104941.h5"  # 6.025 GHz, 6425 uA, dither False, with JPA

# load_filename = "sweep_power_20210206_110947.h5"  # 6.164 GHz
# load_filename = "sweep_power_20210206_133635.h5"  # 6.306 GHz
# load_filename = "sweep_power_20210206_165933.h5"  # 6.607 GHz
# load_filename = "sweep_power_20210206_194753.h5"  # 6.770 GHz

# load_filename = "sweep_power_20210226_000906.h5"  # 32_000 uA
load_filename = "sweep_power_20210226_191421.h5"  # 6_425 uA, jumps
# load_filename = "sweep_power_20210227_094842.h5"  # 6_425 uA, jumps
# load_filename = "sweep_power_20210227_130531.h5"  # 20_000 uA, jumps
# load_filename = "sweep_power_20210227_154445.h5"  # 32_000 uA, few jumps
# load_filename = "sweep_power_20210227_195737.h5"  # 32_000 uA, no jumps?
# load_filename = "sweep_power_20210228_001222.h5"  # 32_000 uA, no jumps?


def load(load_filename):
    with h5py.File(load_filename, "r") as h5f:
        df = h5f.attrs["df"]
        dither = h5f.attrs["dither"]
        input_port = h5f.attrs["input_port"]
        output_port = h5f.attrs["output_port"]
        freq_arr = h5f["freq_arr"][()]
        amp_arr = h5f["amp_arr"][()]
        resp_arr = h5f["resp_arr"][()]
        source_code = h5f["source_code"][()]

    nr_amps = len(amp_arr)

    global AMP_IDX
    AMP_IDX = nr_amps // 2

    if NORM:
        resp_scaled = np.zeros_like(resp_arr)
        for jj in range(nr_amps):
            resp_scaled[jj] = resp_arr[jj] / amp_arr[jj]
    else:
        resp_scaled = resp_arr

    resp_dB = 20. * np.log10(np.abs(resp_scaled))
    amp_dBFS = 20 * np.log10(amp_arr / 1.0)

    # choose limits for colorbar
    cutoff = 1.  # %
    lowlim = np.percentile(resp_dB, cutoff)
    highlim = np.percentile(resp_dB, 100. - cutoff)

    # extent
    x_min = 1e-9 * freq_arr[0]
    x_max = 1e-9 * freq_arr[-1]
    dx = 1e-9 * (freq_arr[1] - freq_arr[0])
    y_min = amp_dBFS[0]
    y_max = amp_dBFS[-1]
    dy = amp_dBFS[1] - amp_dBFS[0]

    fig1 = plt.figure(tight_layout=True, figsize=(6.4, 9.6))
    ax1 = fig1.add_subplot(2, 1, 1)
    im = ax1.imshow(
        resp_dB,
        origin='lower',
        aspect='auto',
        interpolation='none',
        extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
        vmin=lowlim,
        vmax=highlim,
    )
    line_sel = ax1.axhline(amp_dBFS[AMP_IDX], ls="--", c="k", lw=3, animated=BLIT)
    # ax1.set_title(f"amp = {amp_arr[AMP_IDX]:.2e}")
    ax1.set_xlabel("Frequency [GHz]")
    ax1.set_ylabel("Drive amplitude [dBFS]")
    cb = fig1.colorbar(im)
    cb.set_label("Response amplitude [dB]")

    ax2 = fig1.add_subplot(4, 1, 3)
    ax3 = fig1.add_subplot(4, 1, 4, sharex=ax2)

    line_a, = ax2.plot(1e-9 * freq_arr, resp_dB[AMP_IDX], animated=BLIT)
    line_fit_a, = ax2.plot(1e-9 * freq_arr, np.full_like(freq_arr, np.nan), ls="--", animated=BLIT)
    line_p, = ax3.plot(1e-9 * freq_arr, np.angle(resp_arr[AMP_IDX]), animated=BLIT)
    line_fit_p, = ax3.plot(1e-9 * freq_arr, np.full_like(freq_arr, np.nan), ls="--", animated=BLIT)

    f_min = 1e-9 * freq_arr.min()
    f_max = 1e-9 * freq_arr.max()
    f_rng = f_max - f_min
    a_min = resp_dB.min()
    a_max = resp_dB.max()
    a_rng = a_max - a_min
    p_min = -np.pi
    p_max = np.pi
    p_rng = p_max - p_min
    ax2.set_xlim(f_min - 0.05 * f_rng, f_max + 0.05 * f_rng)
    ax2.set_ylim(a_min - 0.05 * a_rng, a_max + 0.05 * a_rng)
    ax3.set_xlim(f_min - 0.05 * f_rng, f_max + 0.05 * f_rng)
    ax3.set_ylim(p_min - 0.05 * p_rng, p_max + 0.05 * p_rng)

    ax3.set_xlabel("Frequency [GHz]")
    ax2.set_ylabel("Response amplitude [dB]")
    ax3.set_ylabel("Response phase [rad]")

    def onbuttonpress(event):
        if event.inaxes == ax1:
            global AMP_IDX
            AMP_IDX = np.argmin(np.abs(amp_dBFS - event.ydata))
            update()

    def onkeypress(event):
        global AMP_IDX
        if event.inaxes == ax1:
            if event.key == "up":
                AMP_IDX += 1
                if AMP_IDX >= len(amp_dBFS):
                    AMP_IDX = len(amp_dBFS) - 1
                update()
            elif event.key == "down":
                AMP_IDX -= 1
                if AMP_IDX < 0:
                    AMP_IDX = 0
                update()

    def update():
        global AMP_IDX
        line_sel.set_ydata([amp_dBFS[AMP_IDX], amp_dBFS[AMP_IDX]])
        # ax1.set_title(f"amp = {amp_arr[AMP_IDX]:.2e}")
        print(f"drive amp {AMP_IDX:d}: {amp_arr[AMP_IDX]:.2e} FS = {amp_dBFS[AMP_IDX]:.1f} dBFS")
        line_a.set_ydata(resp_dB[AMP_IDX])
        line_p.set_ydata(np.angle(resp_arr[AMP_IDX]))
        line_fit_a.set_ydata(np.full_like(freq_arr, np.nan))
        line_fit_p.set_ydata(np.full_like(freq_arr, np.nan))
        # ax2.set_title("")
        if BLIT:
            global bg
            fig1.canvas.restore_region(bg)
            ax1.draw_artist(line_sel)
            ax2.draw_artist(line_a)
            ax3.draw_artist(line_p)
            fig1.canvas.blit(fig1.bbox)
            fig1.canvas.flush_events()
        else:
            fig1.canvas.draw()

    def onselect(xmin, xmax):
        global AMP_IDX
        port = circuit.notch_port(freq_arr, resp_arr[AMP_IDX])
        port.autofit(fcrop=(xmin * 1e9, xmax * 1e9))
        if NORM:
            line_fit_a.set_data(1e-9 * port.f_data, 20 * np.log10(np.abs(port.z_data_sim / amp_arr[AMP_IDX])))
        else:
            line_fit_a.set_data(1e-9 * port.f_data, 20 * np.log10(np.abs(port.z_data_sim)))
        line_fit_p.set_data(1e-9 * port.f_data, np.angle(port.z_data_sim))
        # print(port.fitresults)
        print("----------------")
        print(f"fr = {port.fitresults['fr']}")
        print(f"Qi = {port.fitresults['Qi_dia_corr']}")
        print(f"Qc = {port.fitresults['Qc_dia_corr']}")
        print(f"Ql = {port.fitresults['Ql']}")
        print(f"kappa = {port.fitresults['fr'] / port.fitresults['Qc_dia_corr']}")
        print("----------------")
        # ax2.set_title(
        #     f"fr = {1e-6*fr:.0f} MHz, Ql = {Ql:.0f}, Qi = {Qi:.0f}, Qc = {Qc:.0f}, kappa = {1e-3*kappa:.0f} kHz")
        if BLIT:
            global bg
            fig1.canvas.restore_region(bg)
            ax1.draw_artist(line_sel)
            ax2.draw_artist(line_a)
            ax2.draw_artist(line_fit_a)
            ax3.draw_artist(line_p)
            ax3.draw_artist(line_fit_p)
            fig1.canvas.blit(fig1.bbox)
            fig1.canvas.flush_events()
        else:
            fig1.canvas.draw()

    fig1.canvas.mpl_connect('button_press_event', onbuttonpress)
    fig1.canvas.mpl_connect('key_press_event', onkeypress)
    rectprops = dict(facecolor='tab:gray', alpha=0.5)
    span_a = mwidgets.SpanSelector(ax2, onselect, 'horizontal', rectprops=rectprops, useblit=BLIT)
    span_p = mwidgets.SpanSelector(ax3, onselect, 'horizontal', rectprops=rectprops, useblit=BLIT)
    fig1.show()
    if BLIT:
        fig1.canvas.draw()
        fig1.canvas.flush_events()
        global bg
        bg = fig1.canvas.copy_from_bbox(fig1.bbox)
        ax1.draw_artist(line_sel)
        ax2.draw_artist(line_a)
        ax3.draw_artist(line_p)
        fig1.canvas.blit(fig1.bbox)

    fig1.span_a = span_a
    fig1.span_p = span_p

    return fig1


if __name__ == "__main__":
    fig1 = load(load_filename)
