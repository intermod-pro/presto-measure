# -*- coding: utf-8 -*-
"""
Loader for files saved by sweep_nco_time.py
"""
import os
import sys

import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
from resonator_tools import circuit

rcParams['figure.dpi'] = 108.8

if len(sys.argv) == 2:
    load_filename = sys.argv[1]
    print(f"Loading: {os.path.realpath(load_filename)}")
else:
    load_filename = None


def load(load_filename):
    with h5py.File(load_filename, "r") as h5f:
        df = h5f.attrs["df"]
        amp = h5f.attrs["amp"]
        phase = h5f.attrs["phase"]
        dither = h5f.attrs["dither"]
        input_port = h5f.attrs["input_port"]
        output_port = h5f.attrs["output_port"]
        freq_arr = h5f["freq_arr"][()]
        resp_arr = h5f["resp_arr"][()]
        source_code = h5f["source_code"][()]

    resp_dB = 20. * np.log10(np.abs(resp_arr))

    fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
    ax11, ax12 = ax1
    ax11.plot(1e-9 * freq_arr, resp_dB)
    # ax11.plot(1e-9 * freq_arr, np.abs(resp_arr))
    line_fit_a, = ax11.plot(1e-9 * freq_arr, np.full_like(freq_arr, np.nan), ls="--")
    ax12.plot(1e-9 * freq_arr, np.angle(resp_arr))
    line_fit_p, = ax12.plot(1e-9 * freq_arr, np.full_like(freq_arr, np.nan), ls="--")
    ax12.set_xlabel("Frequency [GHz]")
    ax11.set_ylabel("Response amplitude [dB]")
    ax12.set_ylabel("Response phase [rad]")

    def onselect(xmin, xmax):
        port = circuit.notch_port(freq_arr, resp_arr)
        port.autofit(fcrop=(xmin * 1e9, xmax * 1e9))
        line_fit_a.set_data(1e-9 * port.f_data, 20 * np.log10(np.abs(port.z_data_sim)))
        line_fit_p.set_data(1e-9 * port.f_data, np.angle(port.z_data_sim))
        print("----------------")
        print(f"fr = {port.fitresults['fr']}")
        print(f"Qi = {port.fitresults['Qi_dia_corr']}")
        print(f"Qc = {port.fitresults['Qc_dia_corr']}")
        print(f"Ql = {port.fitresults['Ql']}")
        print(f"kappa = {port.fitresults['fr'] / port.fitresults['Qc_dia_corr']}")
        print("----------------")
        fig1.canvas.draw()

    rectprops = dict(facecolor='tab:gray', alpha=0.5)
    span_a = mwidgets.SpanSelector(ax11, onselect, 'horizontal', rectprops=rectprops)
    span_p = mwidgets.SpanSelector(ax12, onselect, 'horizontal', rectprops=rectprops)
    fig1.show()

    return fig1, span_a, span_p


if __name__ == "__main__":
    fig1, span_a, span_p = load(load_filename)
