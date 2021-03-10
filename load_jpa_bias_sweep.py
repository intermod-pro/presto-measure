# -*- coding: utf-8 -*-
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend

load_filename = "jpa_bias_sweep_20210205_102845.h5"

load_filename = "jpa_bias_sweep_20210226_074737.h5"

with h5py.File(load_filename, "r") as h5f:
    df = h5f.attrs["df"]
    Navg = h5f.attrs["Navg"]
    amp = h5f.attrs["amp"]
    dither = h5f.attrs["dither"]
    input_port = h5f.attrs["input_port"]
    output_port = h5f.attrs["output_port"]
    bias_port = h5f.attrs["bias_port"]
    freq_arr = h5f["freq_arr"][()]
    bias_arr = h5f["bias_arr"][()]
    resp_arr = h5f["resp_arr"][()]
    source_code = h5f["source_code"][()]

resp_db = 20 * np.log10(np.abs(resp_arr))
resp_phase = np.unwrap(np.unwrap(np.angle(resp_arr), axis=1), axis=0)
dw = 2 * np.pi * df
resp_grpdly = -np.diff(resp_phase, axis=1) / dw
db = bias_arr[1] - bias_arr[0]
resp_dpdb = np.diff(resp_phase, axis=0) / db

# data_plot = resp_grpdly * 1e9
data_plot = resp_dpdb

vmin = np.percentile(data_plot, 1)
vmax = np.percentile(data_plot, 99)

fig, ax = plt.subplots(tight_layout=True)
im = ax.imshow(
    data_plot,
    origin='lower',
    aspect='auto',
    extent=(1e-9 * freq_arr[0], 1e-9 * freq_arr[-1], bias_arr[0], bias_arr[-1]),
    vmin=vmin,
    vmax=vmax,
)
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Bias [V]')
cb = fig.colorbar(im)
# cb.set_label("Group delay [ns]")
cb.set_label(r"$\mathrm{d}\phi / \mathrm{d} V$ [rad / V]")
fig.show()
