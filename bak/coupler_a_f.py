# -*- coding: utf-8 -*-
import os
import time

import h5py
import numpy as np

from mla_server import set_dc_bias
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.utils import get_sourcecode, sin2

# import load_coupler_a_f

# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42874
EXT_REF_CLK = False  # set to True to lock to an external reference clock
USE_JPA = True

# cavity drive: readout
readout_freq_1 = 6.167_009 * 1e9  # Hz, resonator 1, with coupler bias
readout_freq_2 = 6.029_130 * 1e9  # Hz, resonator 2, with coupler bias
readout_amp = 0.1  # FS
readout_duration = 2 * 1e-6  # s, duration of the readout pulse
readout_port = 1

# qubit drive: control
control_freq_1 = 3.556_520 * 1e9  # Hz, with coupler bias
control_freq_2 = 4.093_042 * 1e9  # Hz, with coupler bias
control_amp_1 = 0.533  # FS <-- pi pulse, with coupler bias
control_amp_2 = 0.760  # FS <-- pi pulse, with coupler bias
control_duration = 20 * 1e-9  # s, duration of the control pulse
control_port_1 = 3  # qubit 1
control_port_2 = 4  # qubit 2
control_if = 0.0  # Hz, >0 for HSB

# coupler
coupler_ac_port = 5
# coupler_dc_port = 2  # MLA
coupler_dc_bias = 1.143  # V, ≈ 0.257 * Phi_0
# nr_amps = 64  # <-- NOTE
nr_amps = 128
max_amp = 0.15
coupler_ac_amp_arr = np.linspace(0.0, max_amp, nr_amps)  # FS
coupler_ac_duration = 4 * 1e-6  # s
coupler_ac_freq_center = abs(control_freq_2 - control_freq_1)  # Hz
# coupler_ac_freq_span = 10 * 1e6  # Hz <-- NOTE
coupler_ac_freq_span = 4 * 1e6  # Hz
coupler_ac_if = 100 * 1e6  # Hz
# nr_freqs = 64  # <-- NOTE
nr_freqs = 128
_fstart = coupler_ac_freq_center - coupler_ac_freq_span / 2
_fstop = coupler_ac_freq_center + coupler_ac_freq_span / 2
coupler_ac_freq_arr = np.linspace(_fstart, _fstop, nr_freqs)
coupler_ac_nco = coupler_ac_freq_center - coupler_ac_if  # Hz, use upper sideband
coupler_ac_if_arr = coupler_ac_freq_arr - coupler_ac_nco  # Hz

# cavity readout: sample
sample_duration = 4 * 1e-6  # s, duration of the sampling window
sample_port = 1

# other
num_averages = 1_000  # <-- NOTE
# num_averages = 500
wait_delay = 200 * 1e-6  # s, delay between repetitions to allow the qubit to decay
readout_sample_delay = (
    290 * 1e-9
)  # s, delay between readout pulse and sample window to account for latency

# JPA
jpa_pump_freq = 2 * 6.031e9  # Hz
jpa_pump_pwr = 9  # lmx units
jpa_bias = +0.449  # V
jpa_bias_port = 1  # MLA

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
    if USE_JPA:
        pls.hardware.set_lmx(jpa_pump_freq, jpa_pump_pwr)
        set_dc_bias(jpa_bias_port, jpa_bias)
        time.sleep(1.0)

    pls.hardware.set_adc_attenuation(sample_port, 0.0)
    pls.hardware.set_dac_current(readout_port, 32_000)
    pls.hardware.set_dac_current(control_port_1, 32_000)
    pls.hardware.set_dac_current(control_port_2, 32_000)
    pls.hardware.set_dac_current(coupler_ac_port, 32_000)
    pls.hardware.set_inv_sinc(readout_port, 0)
    pls.hardware.set_inv_sinc(control_port_1, 0)
    pls.hardware.set_inv_sinc(control_port_2, 0)
    pls.hardware.set_inv_sinc(coupler_ac_port, 0)
    pls.hardware.configure_mixer(
        freq=readout_freq_2,  # readout only one reasonator TODO: read both!
        in_ports=sample_port,
        out_ports=readout_port,
        sync=False,  # sync in last call
    )
    pls.hardware.configure_mixer(
        freq=control_freq_1 - control_if,
        out_ports=control_port_1,
        sync=False,  # sync in last call
    )
    pls.hardware.configure_mixer(
        freq=control_freq_2 - control_if,
        out_ports=control_port_2,
        sync=False,  # sync in last call
    )
    pls.hardware.configure_mixer(
        freq=coupler_ac_nco,
        out_ports=coupler_ac_port,
        sync=True,  # sync here
    )

    # ************************************
    # *** Setup measurement parameters ***
    # ************************************

    # Setup lookup tables for frequencies
    # we only need to use carrier 1
    pls.setup_freq_lut(
        output_ports=readout_port,
        group=0,
        frequencies=0.0,  # TODO: use a nonzero intermediate frequency to read out both resonators
        phases=0.0,
        phases_q=0.0,
    )
    pls.setup_freq_lut(
        output_ports=control_port_1,
        group=0,
        frequencies=control_if,
        phases=0.0,
        phases_q=0.0 if control_if == 0.0 else -np.pi / 2,
    )
    pls.setup_freq_lut(
        output_ports=control_port_2,
        group=0,
        frequencies=control_if,
        phases=0.0,
        phases_q=0.0 if control_if == 0.0 else -np.pi / 2,
    )
    pls.setup_freq_lut(
        output_ports=coupler_ac_port,
        group=0,
        frequencies=coupler_ac_if_arr,
        phases=np.full_like(coupler_ac_if_arr, 0.0),
        phases_q=np.full_like(coupler_ac_if_arr, -np.pi / 2),  # upper sideband
    )

    # Setup lookup tables for amplitudes
    pls.setup_scale_lut(
        output_ports=readout_port,
        group=0,
        scales=readout_amp,
    )
    pls.setup_scale_lut(
        output_ports=control_port_1,
        group=0,
        scales=control_amp_1,
    )
    pls.setup_scale_lut(
        output_ports=control_port_2,
        group=0,
        scales=control_amp_2,
    )
    pls.setup_scale_lut(
        output_ports=coupler_ac_port,
        group=0,
        scales=coupler_ac_amp_arr,
    )

    # Setup readout and control pulses
    # use setup_long_drive to create a pulse with square envelope
    # setup_long_drive supports smooth rise and fall transitions for the pulse,
    # but we keep it simple here
    readout_pulse = pls.setup_long_drive(
        output_port=readout_port,
        group=0,
        duration=readout_duration,
    )
    control_ns = int(
        round(control_duration * pls.get_fs("dac"))
    )  # number of samples in the control template
    control_envelope = sin2(control_ns)
    control_pulse_1 = pls.setup_template(
        output_port=control_port_1,
        group=0,
        template=control_envelope,
        template_q=control_envelope,
        envelope=True,
    )
    control_pulse_2 = pls.setup_template(
        output_port=control_port_2,
        group=0,
        template=control_envelope,
        template_q=control_envelope,
        envelope=True,
    )
    coupler_ac_pulse = pls.setup_long_drive(
        output_port=coupler_ac_port,
        group=0,
        duration=coupler_ac_duration,
        rise_time=50 * 1e-9,
        fall_time=50 * 1e-9,
    )

    # Setup sampling window
    pls.set_store_ports(sample_port)
    pls.set_store_duration(sample_duration)

    # ******************************
    # *** Program pulse sequence ***
    # ******************************
    T0 = 0.0  # s, start at time zero ...
    T = T0
    for _ii in range(nr_freqs):
        # start in |00>
        # pi pulse on qubit 2
        pls.reset_phase(T, control_port_2)
        pls.output_pulse(T, control_pulse_2)
        # we are in |01>
        # turn on coupler
        T += control_duration
        pls.output_pulse(T, coupler_ac_pulse)
        # we are somewhere between |01> and |10>
        # readout right after coupler pulse
        T += coupler_ac_duration
        pls.reset_phase(T, readout_port)
        pls.output_pulse(T, readout_pulse)
        # sample the readout pulse, taking latency into account
        pls.store(T + readout_sample_delay)
        # Move to next iteration
        T += readout_duration
        pls.next_frequency(T, coupler_ac_port)  # prepare for next iteration
        T += wait_delay
    pls.next_scale(T, coupler_ac_port)  # prepare for next iteration
    T += wait_delay

    # **************************
    # *** Run the experiment ***
    # **************************
    pls.run(
        period=T,
        repeat_count=nr_amps,
        num_averages=num_averages,
        print_time=True,
    )
    t_arr, (data_I, data_Q) = pls.get_store_data()

    if USE_JPA:
        pls.hardware.set_lmx(0.0, 0.0)
        set_dc_bias(jpa_bias_port, 0.0)

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
    h5f.attrs["control_freq_1"] = control_freq_1
    h5f.attrs["control_freq_2"] = control_freq_2
    h5f.attrs["control_if"] = control_if
    h5f.attrs["readout_freq_1"] = readout_freq_1
    h5f.attrs["readout_freq_2"] = readout_freq_2
    h5f.attrs["readout_duration"] = readout_duration
    h5f.attrs["control_duration"] = control_duration
    h5f.attrs["readout_amp"] = readout_amp
    h5f.attrs["control_amp_1"] = control_amp_1
    h5f.attrs["control_amp_2"] = control_amp_2
    h5f.attrs["sample_duration"] = sample_duration
    h5f.attrs["wait_delay"] = wait_delay
    h5f.attrs["readout_sample_delay"] = readout_sample_delay
    h5f.attrs["coupler_dc_bias"] = coupler_dc_bias
    h5f.attrs["nr_freqs"] = nr_freqs
    h5f.attrs["nr_amps"] = nr_amps
    h5f.attrs["coupler_ac_duration"] = coupler_ac_duration

    h5f.create_dataset("t_arr", data=t_arr)
    h5f.create_dataset("store_arr", data=store_arr)
    h5f.create_dataset("coupler_ac_freq_arr", data=coupler_ac_freq_arr)
    h5f.create_dataset("coupler_ac_amp_arr", data=coupler_ac_amp_arr)
print(f"Data saved to: {save_path}")

# fig1, fig2 = load_coupler_a_f.load(save_path)
