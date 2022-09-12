# -*- coding: utf-8 -*-
import os
import time

import h5py
import numpy as np

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.utils import get_sourcecode, sin2

import load_ramsey_single

# Presto's IP address or hostname
ADDRESS = "192.0.2.53"
EXT_REF_CLK = False  # set to True to lock to an external reference clock

# cavity drive: readout
readout_freq = 6.213095 * 1e9  # Hz, resonator 1
# readout_freq = 6.376650 * 1e9  # Hz, resonator 2
readout_amp = 10 ** (-10.0 / 20)  # FS
readout_duration = 2e-6  # s, duration of the readout pulse
readout_port = 1

# qubit drive: control
control_freq = 4.146391 * 1e9  # Hz <-- from data/ramsey_20210428_041929.h5, qubit 1
# control_freq = 4.776805 * 1e9  # Hz <-- from data/ramsey_20210428_160241.h5 , qubit 2
control_freq += 100e3  # Hz, detune from qubit frequency
control_amp = 0.07968  # FS <-- pi/2 pulse from data/rabi_amp_20210428_101944.h5, qubit 1
# control_amp = 0.05941  # FS <-- pi/2 pulse from data/rabi_amp_20210428_164251.h5, qubit 2
control_if = 0 * 1e6  # Hz
control_duration = 100 * 1e-9  # s, duration of the control pulse
control_port = 5  # qubit 1
# control_port = 7  # qubit 2

# cavity readout: sample
sample_duration = 4 * 1e-6  # s, duration of the sampling window
sample_port = 1

# coupler
coupler_dc_port = 13
coupler_dc_bias = 0.4  # FS, ~ 500 mV into 50 ohm, ~ 0.3 phi_0

# Ramsey experiment
num_averages = 1_000
nr_delays = 256  # number of steps when changing delay between control and readout pulses
dt_delays = 0.1 * 1e-6  # s, step size when changing delay between control and readout pulses
wait_delay = 500e-6  # s, delay between repetitions to allow the qubit to decay
readout_sample_delay = (
    300 * 1e-9
)  # s, delay between readout pulse and sample window to account for latency

# Instantiate interface class
with pulsed.Pulsed(
    address=ADDRESS,
    ext_ref_clk=EXT_REF_CLK,
    adc_mode=AdcMode.Mixed,
    adc_fsample=AdcFSample.G2,
    dac_mode=[DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
    dac_fsample=[DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
) as pls:
    pls.hardware.set_adc_attenuation(sample_port, 0.0)
    pls.hardware.set_dac_current(readout_port, 32_000)
    pls.hardware.set_dac_current(control_port, 32_000)
    pls.hardware.set_dac_current(coupler_dc_port, 32_000)
    pls.hardware.set_inv_sinc(readout_port, 0)
    pls.hardware.set_inv_sinc(control_port, 0)
    pls.hardware.set_inv_sinc(coupler_dc_port, 0)
    pls.hardware.configure_mixer(
        freq=readout_freq,
        in_ports=sample_port,
        out_ports=readout_port,
        sync=False,  # sync in last call
    )
    pls.hardware.configure_mixer(
        freq=control_freq - control_if,
        out_ports=control_port,
        sync=True,  # sync in last call
    )
    pls.hardware.configure_mixer(
        freq=0.0,
        out_ports=coupler_dc_port,
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
        frequencies=0.0,
        phases=0.0,
        phases_q=0.0,
    )
    pls.setup_freq_lut(
        output_ports=control_port,
        group=0,
        frequencies=control_if,
        phases=0.0,
        phases_q=0.0 if control_if == 0.0 else -np.pi / 2,
    )
    pls.setup_freq_lut(
        output_ports=coupler_dc_port,
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
    pls.setup_scale_lut(
        output_ports=coupler_dc_port,
        group=0,
        scales=coupler_dc_bias,
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
    T0 = 0.0  # s, start at time zero ...
    T = T0
    # we'll start the coupler DC bias at time 0
    # the DC port of the bias tee has some f_3dB ~ 20 kHz
    # so wait at least 5 x 8 us (5 x 1 / 2*pi*f_3dB) ~ 40 us
    # measured rise/fall time 10-90% 15 us
    T += 40e-6  # s
    for ii in range(nr_delays):
        # first pi/2 pulse
        pls.reset_phase(T, control_port)
        pls.output_pulse(T, control_pulse)
        # second pi/2 pulse
        T += control_duration + ii * dt_delays
        pls.output_pulse(T, control_pulse)
        # Readout pulse starts after control pulse,
        # with an increasing delay
        T += control_duration
        pls.reset_phase(T, readout_port)
        pls.output_pulse(T, readout_pulse)
        # Sampling window
        pls.store(T + readout_sample_delay)
        # Move to next iteration
        T += readout_duration
        T += wait_delay

    coupler_dc_pulse = pls.setup_long_drive(
        output_port=coupler_dc_port,
        group=0,
        duration=T,  # total duration of experiment so far!
        amplitude=1.0,
        amplitude_q=1.0,
    )
    pls.output_pulse(T0, coupler_dc_pulse)  # start at the beginning
    T += 40e-6  # s

    # **************************
    # *** Run the experiment ***
    # **************************
    pls.run(
        period=T,
        repeat_count=1,
        num_averages=num_averages,
        print_time=True,
    )
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
    h5f.attrs["control_if"] = control_if
    h5f.attrs["readout_freq"] = readout_freq
    h5f.attrs["readout_duration"] = readout_duration
    h5f.attrs["control_duration"] = control_duration
    h5f.attrs["readout_amp"] = readout_amp
    h5f.attrs["control_amp"] = control_amp
    h5f.attrs["sample_duration"] = sample_duration
    h5f.attrs["nr_delays"] = nr_delays
    h5f.attrs["dt_delays"] = dt_delays
    h5f.attrs["wait_delay"] = wait_delay
    h5f.attrs["readout_sample_delay"] = readout_sample_delay
    h5f.create_dataset("t_arr", data=t_arr)
    h5f.create_dataset("store_arr", data=store_arr)
print(f"Data saved to: {save_path}")

# *****************
# *** Plot data ***
# *****************
load_ramsey_single.load(save_path)
