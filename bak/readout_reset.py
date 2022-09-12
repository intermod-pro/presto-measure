# -*- coding: utf-8 -*-
import os
import time

import h5py
import numpy as np

from mla_server import set_dc_bias
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.utils import get_sourcecode, sin2

import load_readout_reset

WHICH_QUBIT = 2  # 1 (higher resonator) or 2 (lower resonator)
USE_JPA = False

# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42874
EXT_REF_CLK = False  # set to True to lock to an external reference clock
jpa_bias_port = 1

if WHICH_QUBIT == 1:
    readout_freq = 6.166_600 * 1e9  # Hz, frequency for resonator readout
    control_freq = 3.557_866 * 1e9  # Hz
    control_amp = 0.1026  # FS <-- pi pulse
    control_port = 3
    jpa_pump_freq = 2 * 6.169e9  # Hz
    jpa_pump_pwr = 11  # lmx units
    jpa_bias = +0.437  # V
    template_filename = None
elif WHICH_QUBIT == 2:
    readout_freq = 6.028_448 * 1e9  # Hz, frequency for resonator readout
    control_freq = 4.091_777 * 1e9  # Hz
    control_amp = 0.1537  # FS <-- pi pulse
    control_port = 4
    jpa_pump_freq = 2 * 6.031e9  # Hz
    jpa_pump_pwr = 9  # lmx units
    jpa_bias = +0.449  # V
    template_filename = "data/readout_avg_20210610_140856.h5"
else:
    raise ValueError

# cavity drive: readout
readout_amp = 0.3  # FS
readout_duration = 2 * 1e-6  # s, duration of the readout pulse
readout_port = 1

# qubit drive: control
control_duration = 100 * 1e-9  # s, duration of the control pulse

# cavity readout: sample
sample_duration = 4 * 1e-6  # s, duration of the sampling window
sample_port = 1
readout_sample_delay = (
    290 * 1e-9 + 0 * 1e-6
)  # s, delay between readout pulse and sample window to account for latency

# IQ readout experiment
num_averages = 10_000
wait_delay = 500e-6  # s, delay between repetitions to allow the qubit to decay
readout_decay = 2e-6  # s, delay between the two readouts to allow photons to decay

# template matching
with h5py.File(template_filename, "r") as h5f:
    match_t_in_store = h5f.attrs["match_t_in_store"]
    template_g = h5f["template_g"][()]
    template_e = h5f["template_e"][()]
    threshold = 0.5 * (np.sum(np.abs(template_e) ** 2) - np.sum(np.abs(template_g) ** 2))

# Instantiate interface class
with pulsed.Pulsed(
    address=ADDRESS,
    port=PORT,
    ext_ref_clk=EXT_REF_CLK,
    adc_mode=AdcMode.Mixed,
    adc_fsample=AdcFSample.G2,
    dac_mode=[DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
    dac_fsample=[DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
    dry_run=True,
) as pls:
    pls.hardware.set_adc_attenuation(sample_port, 0.0)
    pls.hardware.set_dac_current(readout_port, 32_000)
    pls.hardware.set_dac_current(control_port, 32_000)
    pls.hardware.set_inv_sinc(readout_port, 0)
    pls.hardware.set_inv_sinc(control_port, 0)
    pls.hardware.configure_mixer(
        freq=readout_freq,
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
        frequencies=0.0,
        phases=0.0,
        phases_q=0.0,
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
    control_pulse_pi_div2 = pls.setup_template(
        output_port=control_port,
        group=0,
        template=control_envelope / 2,
        template_q=control_envelope / 2,
        envelope=True,
    )

    # Setup sampling window
    pls.set_store_ports(sample_port)
    pls.set_store_duration(sample_duration)

    # Setup template matching
    templ_g = np.zeros(len(template_g) * 2, np.float64)
    templ_g[0::2] = template_g.real
    templ_g[1::2] = template_g.imag
    templ_e = np.zeros(len(template_e) * 2, np.float64)
    templ_e[0::2] = template_e.real
    templ_e[1::2] = template_e.imag
    match_g, match_e = pls.setup_template_matching_pair(
        input_port=sample_port,
        template1=-templ_g,  # NOTE minus sign
        template2=templ_e,
        threshold=threshold,
    )  # success when match_e - match_g - threshold > 0

    # Setup feedback
    # make the control pulse conditional
    pls.setup_condition([match_g, match_e], control_pulse)

    # ******************************
    # *** Program pulse sequence ***
    # ******************************
    T = 0.0  # s, start at time zero ...
    # Unconditional pi/2 pulse
    pls.reset_phase(T, control_port)
    pls.output_pulse(T, control_pulse_pi_div2)
    T += control_duration
    # First readout, store and match
    pls.reset_phase(T, readout_port)
    pls.output_pulse(T, readout_pulse)
    pls.store(T + readout_sample_delay)
    pls.match(T + readout_sample_delay + match_t_in_store, [match_g, match_e])
    T += readout_duration
    # Wait for first readout to decay
    T += readout_decay
    # Conditional pi pulse
    # pls.reset_phase(T, control_port)
    # pls.output_pulse(T, control_pulse)
    T += control_duration
    # Second readout, store and match
    # right after conditional pulse
    pls.reset_phase(T, readout_port)
    pls.output_pulse(T, readout_pulse)
    pls.store(T + readout_sample_delay)
    pls.match(T + readout_sample_delay + match_t_in_store, [match_g, match_e])
    T += readout_duration
    # Move to next iteration
    T += wait_delay

    # **************************
    # *** Run the experiment ***
    # **************************
    pls.run(
        period=T,
        repeat_count=1,
        num_averages=num_averages,
        print_time=True,
    )
    if USE_JPA:
        pls.hardware.set_lmx(0.0, 0.0)
        set_dc_bias(jpa_bias_port, 0.0)
    t_arr, (data_I, data_Q) = pls.get_store_data()
    match_g_data, match_e_data = pls.get_template_matching_data([match_g, match_e])

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
    h5f.attrs["readout_freq"] = readout_freq
    h5f.attrs["control_freq"] = control_freq
    h5f.attrs["readout_duration"] = readout_duration
    h5f.attrs["control_duration"] = control_duration
    h5f.attrs["readout_amp"] = readout_amp
    h5f.attrs["control_amp"] = control_amp
    h5f.attrs["sample_duration"] = sample_duration
    h5f.attrs["wait_delay"] = wait_delay
    h5f.attrs["readout_decay"] = readout_decay
    h5f.attrs["readout_sample_delay"] = readout_sample_delay
    h5f.attrs["match_t_in_store"] = match_t_in_store
    h5f.attrs["threshold"] = threshold
    h5f.create_dataset("t_arr", data=t_arr)
    h5f.create_dataset("store_arr", data=store_arr)
    h5f.create_dataset("match_g_data", data=match_g_data)
    h5f.create_dataset("match_e_data", data=match_e_data)
    h5f.create_dataset("template_g", data=template_g)
    h5f.create_dataset("template_e", data=template_e)
print(f"Data saved to: {save_path}")

# *****************
# *** Plot data ***
# *****************
fig1, fig2 = load_readout_reset.load(save_path)
