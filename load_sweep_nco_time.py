# -*- coding: utf-8 -*-
"""
Loader for files saved by sweep_nco_time.py
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

load_filename = "sweep_nco_time_20210205_081214.h5"
load_filename = "sweep_nco_time_20210205_093012.h5"
load_filename = "sweep_nco_time_20210205_134801.h5"
load_filename = "sweep_nco_time_20210205_162123.h5"
load_filename = "sweep_nco_time_20210205_185837.h5"
load_filename = "sweep_nco_time_20210205_194103.h5"


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
        print(port.fitresults)
        fr = port.fitresults['fr']
        Qi = port.fitresults['Qi_dia_corr']
        Qc = port.fitresults['Qc_dia_corr']
        Ql = port.fitresults['Ql']
        kappa = fr / Qc
        ax11.set_title(
            f"fr = {1e-6*fr:.0f} MHz, Ql = {Ql:.0f}, Qi = {Qi:.0f}, Qc = {Qc:.0f}, kappa = {1e-3*kappa:.0f} kHz")
        fig1.canvas.draw()

    rectprops = dict(facecolor='tab:gray', alpha=0.5)
    span_a = mwidgets.SpanSelector(ax11, onselect, 'horizontal', rectprops=rectprops)
    span_p = mwidgets.SpanSelector(ax12, onselect, 'horizontal', rectprops=rectprops)
    fig1.show()

    return fig1, span_a, span_p


if __name__ == "__main__":
    fig1, span_a, span_p = load(load_filename)
