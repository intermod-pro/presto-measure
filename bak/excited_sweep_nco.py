# -*- coding: utf-8 -*-
import os
import time

import h5py
import numpy as np

from mla_server import set_dc_bias
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.utils import get_sourcecode, sin2

import load_excited_sweep

WHICH_QUBIT = 2  # 1 (higher resonator) or 2 (lower resonator)
USE_JPA = False

# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42874
EXT_REF_CLK = False  # set to True to lock to an external reference clock
jpa_bias_port = 1

if WHICH_QUBIT == 1:
    readout_center_freq = 6.166_600 * 1e9  # Hz, frequency for resonator readout
    control_freq = 3.557_866 * 1e9  # Hz
    control_amp = 0.1026  # FS <-- pi pulse
    control_port = 3
    jpa_pump_freq = 2 * 6.169e9  # Hz
    jpa_pump_pwr = 11  # lmx units
    jpa_bias = +0.437  # V
elif WHICH_QUBIT == 2:
    readout_center_freq = 6.028_450 * 1e9  # Hz, frequency for resonator readout
    control_freq = 4.093_372 * 1e9  # Hz
    control_amp = 0.7617  # FS <-- pi pulse
    control_port = 4
    jpa_pump_freq = 2 * 6.031e9  # Hz
    jpa_pump_pwr = 9  # lmx units
    jpa_bias = +0.449  # V
else:
    raise ValueError

# cavity drive: readout
readout_amp = 0.4  # FS
readout_duration = 2 * 1e-6  # s, duration of the readout pulse
readout_port = 1

# qubit drive: control
control_duration = 20 * 1e-9  # s, duration of the control pulse

# cavity readout: sample
sample_duration = 1 * 1e-6  # s, duration of the sampling window
sample_port = 1

# sweep experiment
num_averages = 10_000
nr_freqs = 128
df = 0.1 * 1e6  # Hz
readout_if_center = 250 * 1e6  # Hz
wait_delay = 500e-6  # s, delay between repetitions to allow the qubit to decay
readout_sample_delay = (
    290 * 1e-9 + 1 * 1e-6
)  # s, delay between readout pulse and sample window to account for latency
readout_if_arr = readout_if_center + df * (np.arange(nr_freqs) - nr_freqs // 2)
readout_nco = readout_center_freq - readout_if_center
readout_freq_arr = readout_nco + readout_if_arr

# Instantiate interface class
with pulsed.Pulsed(
    address=ADDRESS,
    port=PORT,
    ext_ref_clk=EXT_REF_CLK,
    adc_mode=AdcMode.Mixed,
    adc_fsample=AdcFSample.G2,
    dac_mode=[DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
    dac_fsample=[DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
) as pls:
    pls.hardware.set_adc_attenuation(sample_port, 0.0)
    pls.hardware.set_dac_current(readout_port, 32_000)
    pls.hardware.set_dac_current(control_port, 32_000)
    pls.hardware.set_inv_sinc(readout_port, 0)
    pls.hardware.set_inv_sinc(control_port, 0)
    pls.hardware.configure_mixer(
        freq=readout_nco,
        in_ports=sample_port,
        out_ports=readout_port,
        sync=False,  # sync in next call
    )
    pls.hardware.configure_mixer(
        freq=control_freq,
        out_ports=control_port,
        sync=True,  # sync here
    )
    if USE_JPA:
        pls.hardware.set_lmx(jpa_pump_freq, jpa_pump_pwr)
        set_dc_bias(jpa_bias_port, jpa_bias)
        time.sleep(1.0)

    # ************************************
    # *** Setup measurement parameters ***
    # ************************************

    # Setup lookup tables for frequencies
    pls.setup_freq_lut(
        output_ports=readout_port,
        group=0,
        frequencies=readout_if_arr,
        phases=np.full_like(readout_if_arr, 0.0),
        phases_q=np.full_like(readout_if_arr, -np.pi / 2),  # HSB
    )
    pls.setup_freq_lut(
        output_ports=control_port,
        group=0,
        frequencies=0.0,
        phases=0.0,
        phases_q=0.0,
    )

    # Setup lookup tables for amplitudes
    pls.setup_scale_lut(
        output_ports=readout_port,
        group=0,
        scales=readout_amp,
    )
    pls.setup_scale_lut(
        output_ports=control_port,
        group=0,
        scales=control_amp,
    )

    # Setup readout and control pulses
    # use setup_long_drive to create a pulse with square envelope
    # setup_long_drive supports smooth rise and fall transitions for the pulse,
    # but we keep it simple here
    readout_pulse = pls.setup_long_drive(
        output_port=readout_port,
        group=0,
        duration=readout_duration,
        amplitude=1.0,
        amplitude_q=1.0,
        rise_time=0e-9,
        fall_time=0e-9,
    )
    control_ns = int(
        round(control_duration * pls.get_fs("dac"))
    )  # number of samples in the control template
    control_envelope = sin2(control_ns)
    control_pulse = pls.setup_template(
        output_port=control_port,
        group=0,
        template=control_envelope,
        template_q=control_envelope,
        envelope=True,
    )

    # Setup sampling window
    pls.set_store_ports(sample_port)
    pls.set_store_duration(sample_duration)

    # ******************************
    # *** Program pulse sequence ***
    # ******************************
    T = 0.0  # s, start at time zero ...
    for ii in range(2):
        if ii > 0:
            # pi pulse
            pls.reset_phase(T, control_port)
            pls.output_pulse(T, control_pulse)
        # Readout pulse starts after control pulse
        T += control_duration
        pls.reset_phase(T, readout_port)
        pls.output_pulse(T, readout_pulse)
        # Sampling window
        pls.store(T + readout_sample_delay)
        # Move to next iteration
        T += readout_duration
        if ii > 0:
            pls.next_frequency(T, readout_port)
        T += wait_delay

    # **************************
    # *** Run the experiment ***
    # **************************
    pls.run(
        period=T,
        repeat_count=nr_freqs,
        num_averages=num_averages,
        print_time=True,
    )
    if USE_JPA:
        pls.hardware.set_lmx(0.0, 0.0)
        set_dc_bias(jpa_bias_port, 0.0)
    t_arr, (data_I, data_Q) = pls.get_store_data()

store_arr = data_I + 1j * data_Q

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
    h5f.attrs["num_averages"] = num_averages
    h5f.attrs["control_freq"] = control_freq
    h5f.attrs["readout_duration"] = readout_duration
    h5f.attrs["control_duration"] = control_duration
    h5f.attrs["readout_amp"] = readout_amp
    h5f.attrs["control_amp"] = control_amp
    h5f.attrs["sample_duration"] = sample_duration
    h5f.attrs["nr_freqs"] = nr_freqs
    h5f.attrs["df"] = df
    h5f.attrs["readout_nco"] = readout_nco
    h5f.attrs["readout_if_center"] = readout_if_center
    h5f.attrs["wait_delay"] = wait_delay
    h5f.attrs["readout_sample_delay"] = readout_sample_delay
    h5f.create_dataset("t_arr", data=t_arr)
    h5f.create_dataset("store_arr", data=store_arr)
    h5f.create_dataset("readout_freq_arr", data=readout_freq_arr)
    h5f.create_dataset("readout_if_arr", data=readout_if_arr)
print(f"Data saved to: {save_path}")

# *****************
# *** Plot data ***
# *****************
fig1, fig2 = load_excited_sweep.load(save_path)
