# -*- coding: utf-8 -*-
"""
Perform a readout of a resonator with the Pulsed mode.
"""
import sys
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

from mla_server import set_dc_bias
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed

rcParams["figure.dpi"] = 108.8

# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42874
EXT_REF_CLK = False  # set to True to lock to an external reference clock

# Use JPA?
USE_JPA = True

freq_if = 0 * 1e6

# cavity drive: readout
readout_freq = 6.028_448 * 1e9  # Hz, frequency for resonator readout
readout_amp = 0.1  # FS
readout_duration = 2e-6  # s, duration of the readout pulse
readout_port = 1

# cavity readout: sample
sample_duration = 4 * 1e-6  # s, duration of the sampling window
sample_port = 1

num_averages = 100_000
wait_delay = 500e-6  # s, delay between repetitions to allow the qubit to decay
readout_sample_delay = (
    290 * 1e-9
)  # s, delay between readout pulse and sample window to account for latency

# Instantiate interface class
if USE_JPA:
    jpa_pump_freq = 2 * 6.031e9  # Hz
    jpa_pump_pwr = 7  # lmx units
    jpa_bias = +0.455  # V
    bias_port = 1
with pulsed.Pulsed(
    address=ADDRESS,
    port=PORT,
    ext_ref_clk=EXT_REF_CLK,
    adc_mode=AdcMode.Mixed,
    adc_fsample=AdcFSample.G4,
    dac_mode=DacMode.Mixed42,
    dac_fsample=DacFSample.G10,
) as pls:
    pls.hardware.set_adc_attenuation(sample_port, 0.0)
    pls.hardware.set_dac_current(readout_port, 32_000)
    pls.hardware.set_inv_sinc(readout_port, 0)
    pls.hardware.configure_mixer(
        freq=readout_freq - freq_if,
        in_ports=sample_port,
        out_ports=readout_port,
    )
    if USE_JPA:
        pls.hardware.set_lmx(jpa_pump_freq, jpa_pump_pwr)
        set_dc_bias(bias_port, jpa_bias)
        time.sleep(1.0)
    else:
        pls.hardware.set_lmx(0.0, 0)

    # ************************************
    # *** Setup measurement parameters ***
    # ************************************

    # Setup lookup tables for frequencies
    # we only need to use carrier 1
    pls.setup_freq_lut(
        output_ports=readout_port,
        group=0,
        frequencies=freq_if,
        phases=0.0,
        phases_q=-np.pi / 2 if freq_if > 0.0 else 0.0,
    )

    # Setup lookup tables for amplitudes
    pls.setup_scale_lut(
        output_ports=readout_port,
        group=0,
        scales=readout_amp,
    )

    # Setup readout and control pulses
    # use setup_long_drive to create a pulse with square envelope
    # turn on the global output scaler
    # setup_long_drive supports smooth rise and fall transitions for the pulse,
    # but we keep it simple here
    readout_pulse = pls.setup_long_drive(
        output_port=readout_port,
        group=0,
        duration=readout_duration,
        rise_time=0e-9,
        fall_time=0e-9,
    )

    # Setup sampling window
    pls.set_store_ports(sample_port)
    pls.set_store_duration(sample_duration)

    # ******************************
    # *** Program pulse sequence ***
    # ******************************
    T = 0.0  # s, start at time zero
    pls.reset_phase(T, readout_port)
    pls.output_pulse(T, readout_pulse)
    # Sampling window
    pls.store(T + readout_sample_delay)
    # Move to next iteration
    T += readout_duration
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
    t_arr, (data_I, data_Q) = pls.get_store_data()
    if USE_JPA:
        pls.hardware.set_lmx(0.0, 0.0)
        set_dc_bias(bias_port, 0.0)

data_I = data_I[0, 0, :]
data_Q = data_Q[0, 0, :]

t_low = 1500 * 1e-9
t_high = 2000 * 1e-9
t_span = t_high - t_low
idx_low = np.argmin(np.abs(t_arr - t_low))
idx_high = np.argmin(np.abs(t_arr - t_high))
idx = np.arange(idx_low, idx_high)
n_if = int(round(freq_if * t_span))
f_arr = np.fft.fftfreq(len(t_arr[idx]), 1 / pls.get_fs("adc"))

data = data_I + 1j * data_Q
data_fft = np.fft.fft(data[idx]) / len(data[idx])

fig1, ax1 = plt.subplots(2, 1, tight_layout=True)
ax11, ax12 = ax1
ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
ax12.set_facecolor("#dfdfdf")
data_I_fft = np.fft.rfft(data_I[idx]) / len(t_arr[idx])
data_Q_fft = np.fft.rfft(data_Q[idx]) / len(t_arr[idx])
ax11.plot(1e9 * t_arr, np.abs(data), label="A", c="tab:blue")
ax12.semilogy(1e-6 * f_arr, np.abs(data_fft), c="tab:blue", label="A")
ax12.semilogy(1e-6 * f_arr[n_if], np.abs(data_fft[n_if]), ".", c="tab:green", ms=12)
ax11.legend()
ax12.legend()
ax11.set_xlabel("Time [ns]")
ax12.set_xlabel("Frequency [MHz]")
fig1.show()
