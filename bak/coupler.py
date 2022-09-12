# -*- coding: utf-8 -*-
import os
import time

import h5py
import numpy as np

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.utils import get_sourcecode, sin2

# Presto's IP address or hostname
ADDRESS = "192.0.2.53"
EXT_REF_CLK = False  # set to True to lock to an external reference clock

# cavity drive: readout
readout_freq_1 = 6.213095 * 1e9  # Hz, resonator 1
readout_freq_2 = 6.376650 * 1e9  # Hz, resonator 2
readout_amp = 10 ** (-10.0 / 20)  # FS
readout_duration = 2e-6  # s, duration of the readout pulse
readout_port = 1

# qubit drive: control
control_freq_1 = 4.146391 * 1e9  # Hz <-- from data/ramsey_20210428_041929.h5, qubit 1
control_freq_1 -= 128e3  # Hz, detuning due to coupler
control_freq_2 = 4.776805 * 1e9  # Hz <-- from data/ramsey_20210428_160241.h5 , qubit 2
control_freq_2 -= 296e3  # Hz, detuning due to coupler
control_amp_1 = 0.1594  # FS <-- pi pulse from data/rabi_amp_20210428_101944.h5, qubit 1
control_amp_2 = 0.1188  # FS <-- pi pulse from data/rabi_amp_20210428_164251.h5, qubit 2
control_if = 0 * 1e6  # Hz
control_duration = 100 * 1e-9  # s, duration of the control pulse
control_port_1 = 5  # qubit 1
control_port_2 = 7  # qubit 2

# coupler
coupler_ac_port = 9
coupler_dc_port = 13
coupler_dc_bias = 0.4  # FS, ~ 500 mV into 50 ohm, ~ 0.3 phi_0
coupler_ac_amp = 1.0  # FS
coupler_ac_duration = 4 * 1e-6  # s
coupler_ac_freq_center = abs(control_freq_2 - control_freq_1)  # Hz
coupler_ac_freq_span = 10 * 1e6  # Hz
coupler_ac_if = 100 * 1e6  # Hz
nr_freqs = 512
_fstart = coupler_ac_freq_center - coupler_ac_freq_span / 2
_fstop = coupler_ac_freq_center + coupler_ac_freq_span / 2
coupler_ac_freq_arr = np.linspace(_fstart, _fstop, nr_freqs)
coupler_ac_nco = coupler_ac_freq_center - coupler_ac_if  # Hz, use upper sideband
coupler_ac_if_arr = coupler_ac_freq_arr - coupler_ac_nco  # Hz

# cavity readout: sample
sample_duration = 4 * 1e-6  # s, duration of the sampling window
sample_port = 1

# other
num_averages = 1_000
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
    pls.hardware.set_dac_current(control_port_1, 32_000)
    pls.hardware.set_dac_current(control_port_2, 32_000)
    pls.hardware.set_dac_current(coupler_ac_port, 32_000)
    pls.hardware.set_dac_current(coupler_dc_port, 32_000)
    pls.hardware.set_inv_sinc(readout_port, 0)
    pls.hardware.set_inv_sinc(control_port_1, 0)
    pls.hardware.set_inv_sinc(control_port_2, 0)
    pls.hardware.set_inv_sinc(coupler_ac_port, 0)
    pls.hardware.set_inv_sinc(coupler_dc_port, 0)
    pls.hardware.configure_mixer(
        freq=readout_freq_1,  # readout only first reasonator TODO: read both!
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
        sync=False,  # sync in last call
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
        scales=coupler_ac_amp,
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
        amplitude=1.0,
        amplitude_q=1.0,
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
    # we'll start the coupler DC bias at time 0
    # the DC port of the bias tee has some f_3dB ~ 20 kHz
    # so wait at least 5 x 8 us (5 x 1 / 2*pi*f_3dB) ~ 40 us
    # measured rise/fall time 10-90% 15 us
    T += 40e-6  # s
    for _ii in range(nr_freqs):
        # start in |00>
        # pi pulse on qubit 1
        pls.reset_phase(T, control_port_1)
        pls.output_pulse(T, control_pulse_1)
        # we are in |10>
        # turn on coupler
        T += control_duration
        pls.output_pulse(T, coupler_ac_pulse)
        # we are somewhere between |10> and |01>
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
    h5f.attrs["coupler_ac_amp"] = coupler_ac_amp
    h5f.attrs["nr_freqs"] = nr_freqs
    h5f.attrs["coupler_ac_duration"] = coupler_ac_duration

    h5f.create_dataset("t_arr", data=t_arr)
    h5f.create_dataset("store_arr", data=store_arr)
    h5f.create_dataset("coupler_ac_freq_arr", data=coupler_ac_freq_arr)
print(f"Data saved to: {save_path}")
