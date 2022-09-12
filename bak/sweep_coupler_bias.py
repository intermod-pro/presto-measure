# -*- coding: utf-8 -*-
"""
Simple frequency sweep using the Lockin mode. Also sweep DC bias on coupler. So it's a 2D sweep!
"""
import os
import time

import h5py
import numpy as np

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import lockin
from presto.utils import format_sec, get_sourcecode

from mla_server import set_dc_bias, set_amp

# import load_sweep_coupler_bias

# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42874
EXT_CLK_REF = False

output_port = 1
input_port = 1

amp = 0.1
phase = 0.0
dither = True

f_center = 6.0285 * 1e9  # resonator 2
# f_center = 6.1666 * 1e9  # resonator 1
f_span = 20.44 * 1e6
f_start = f_center - f_span / 2
f_stop = f_center + f_span / 2
df = 4e4  # Hz
Navg = 400

# Coupler bias sweep
nr_bias = 512
bias_min = -4.5  # V
bias_max = +4.5  # V
coupler_bias_arr = np.linspace(bias_min, bias_max, nr_bias)
coupler_bias_port = 2  # MLA

# Tune
ns = int(round(500e6 / df))
df = 500e6 / ns
n_start = int(round(f_start / df))
n_stop = int(round(f_stop / df))
n_arr = np.arange(n_start, n_stop + 1)
nr_freqs = len(n_arr)
freq_arr = df * n_arr
resp_arr = np.zeros((nr_bias, nr_freqs), np.complex128)

set_amp(coupler_bias_port, True)
set_dc_bias(coupler_bias_port, coupler_bias_arr[0])
time.sleep(1.0)

with lockin.Lockin(
    address=ADDRESS,
    port=PORT,
    ext_ref_clk=EXT_CLK_REF,
    adc_mode=AdcMode.Mixed,
    adc_fsample=AdcFSample.G2,
    dac_mode=DacMode.Mixed42,
    dac_fsample=DacFSample.G10,
) as lck:
    lck.hardware.set_adc_attenuation(input_port, 0.0)
    lck.hardware.set_dac_current(output_port, 32_000)
    lck.hardware.set_inv_sinc(output_port, 0)

    lck.hardware.configure_mixer(
        freq=freq_arr[0],
        in_ports=input_port,
        out_ports=output_port,
    )
    lck.set_df(df)
    og = lck.add_output_group(output_port, 1)
    og.set_frequencies(0.0)
    og.set_amplitudes(amp)
    og.set_phases(phase, phase)

    lck.set_dither(dither, output_port)
    ig = lck.add_input_group(input_port, 1)
    ig.set_frequencies(0.0)

    lck.apply_settings()

    t_start = time.time()
    t_last = t_start
    prev_print_len = 0
    print()

    for bb, coupler_bias in enumerate(coupler_bias_arr):
        set_dc_bias(coupler_bias_port, coupler_bias)
        time.sleep(1.0)

        for ii in range(len(n_arr)):
            f = freq_arr[ii]

            lck.hardware.configure_mixer(
                freq=f,
                in_ports=input_port,
                out_ports=output_port,
            )
            lck.hardware.sleep(1e-3, False)

            _d = lck.get_pixels(Navg)
            data_i = _d[input_port][1][:, 0]
            data_q = _d[input_port][2][:, 0]

            avg_i = np.mean(data_i)
            avg_q = np.mean(data_q)
            resp_arr[bb, ii] = avg_i.real + 1j * avg_q.real

            # Calculate and print remaining time
            t_now = time.time()
            if t_now - t_last > np.pi / 3 / 5:
                t_last = t_now
                t_sofar = t_now - t_start
                nr_sofar = (bb * nr_freqs) + (ii + 1)
                nr_left = nr_bias * nr_freqs - nr_sofar
                t_avg = t_sofar / nr_sofar
                t_left = t_avg * nr_left
                str_left = format_sec(t_left)
                msg = "Time remaining: {:s}".format(str_left)
                print_len = len(msg)
                if print_len < prev_print_len:
                    msg += " " * (prev_print_len - print_len)
                print(msg, end="\r", flush=True)
                prev_print_len = print_len

    # Mute outputs at the end of the sweep
    og.set_amplitudes(0.0)
    lck.apply_settings()

set_dc_bias(coupler_bias_port, 0.0)
set_amp(coupler_bias_port, False)

# *************************
# *** Save data to HDF5 ***
# *************************
script_path = os.path.realpath(__file__)  # full path of current script
current_dir, script_basename = os.path.split(script_path)
script_filename = os.path.splitext(script_basename)[0]  # name of current script
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # current date and time
save_basename = f"{script_filename:s}_{timestamp:s}.h5"  # name of save file
save_path = os.path.join(current_dir, "data", save_basename)  # full path of save file
source_code = get_sourcecode(
    __file__
)  # save also the sourcecode of the script for future reference
with h5py.File(save_path, "w") as h5f:
    dt = h5py.string_dtype(encoding="utf-8")
    ds = h5f.create_dataset("source_code", (len(source_code),), dt)
    for ii, line in enumerate(source_code):
        ds[ii] = line
    h5f.attrs["df"] = df
    h5f.attrs["amp"] = amp
    h5f.attrs["phase"] = phase
    h5f.attrs["dither"] = dither
    h5f.attrs["input_port"] = input_port
    h5f.attrs["output_port"] = output_port
    h5f.create_dataset("coupler_bias_arr", data=coupler_bias_arr)
    h5f.create_dataset("freq_arr", data=freq_arr)
    h5f.create_dataset("resp_arr", data=resp_arr)
print(f"Data saved to: {save_path}")

# *****************
# *** Plot data ***
# *****************
# fig1, span_a, span_p = load_sweep_coupler_bias.load(save_path)
