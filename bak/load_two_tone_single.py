# -*- coding: utf-8 -*-
"""
Loader for files saved by two_tone_single.py
"""
import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

rcParams["figure.dpi"] = 331.3

load_filename = ""


def load(load_filename):
    with h5py.File(load_filename, "r") as h5f:
        df = h5f.attrs["df"]
        dither = h5f.attrs["dither"]
        input_port = h5f.attrs["input_port"]
        cavity_port = h5f.attrs["cavity_port"]
        qubit_port = h5f.attrs["qubit_port"]
        cavity_amp = h5f.attrs["cavity_amp"]
        qubit_amp = h5f.attrs["qubit_amp"]
        cavity_freq = h5f.attrs["cavity_freq"]
        qubit_freq_arr = h5f["qubit_freq_arr"][()]
        resp_arr = h5f["resp_arr"][()]
        source_code = h5f["source_code"][()]

    resp_dB = 20.0 * np.log10(np.abs(resp_arr))

    fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
    ax1, ax2 = ax

    # ax1.plot(1e-9 * qubit_freq_arr, resp_dB)
    ax1.plot(1e-9 * qubit_freq_arr, np.abs(resp_arr))
    ax2.plot(1e-9 * qubit_freq_arr, np.angle(resp_arr))

    ax2.set_xlabel("Frequency [GHz]")
    # ax1.set_ylabel("Response amplitude [dBFS]")
    ax1.set_ylabel("Response amplitude [FS]")
    ax2.set_ylabel("Response phase [rad]")

    fig.show()
    return fig


if __name__ == "__main__":
    fig = load(load_filename)
