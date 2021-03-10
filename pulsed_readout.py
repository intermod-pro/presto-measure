import sys
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

import pulsed
from utils import untwist_downconversion

rcParams['figure.dpi'] = 108.8

# Presto's IP address or hostname
ADDRESS = "130.237.35.90"
PORT = 42878
EXT_REF_CLK = False  # set to True to lock to an external reference clock

# Use JPA?
JPA = False
if JPA:
    if '/home/riccardo/IntermodulatorSuite' not in sys.path:
        sys.path.append('/home/riccardo/IntermodulatorSuite')
    from mlaapi import mla_api
    from mlaapi import mla_globals
    settings = mla_globals.read_config()
    mla = mla_api.MLA(settings)

freq_if = 100 * 1e6

# cavity drive: readout
readout_freq = 6.029 * 1e9  # Hz
readout_amp = 3e-3  # FS
readout_duration = 2e-6  # s, duration of the readout pulse
readout_port = 1

# cavity readout: sample
sample_duration = 2100 * 1e-9  # s, duration of the sampling window
sample_port = 1

num_averages = 10_000
wait_delay = 500e-6  # s, delay between repetitions to allow the qubit to decay
readout_sample_delay = 250 * 1e-9  # s, delay between readout pulse and sample window to account for latency

# Instantiate interface class
if JPA:
    jpa_pump_freq = 2 * 6.031e9  # Hz
    jpa_pump_pwr = 7  # lmx units
    jpa_bias = +0.432  # V
    bias_port = 1
    mla.connect()
with pulsed.Pulsed(
        address=ADDRESS,
        port=PORT,
        ext_ref_clk=EXT_REF_CLK,
        adc_mode=pulsed.MODE_MIX,
        dac_mode=pulsed.MODE_LSB if freq_if > 0.0 else pulsed.MODE_MIX,
) as pls:
    pls.hardware.set_adc_attenuation(sample_port, 0.0)
    pls.hardware.set_dac_current(readout_port, 32_000)
    pls.hardware.set_inv_sinc(readout_port, 0)
    pls.hardware.configure_mixer(
        freq=readout_freq + freq_if,
        in_ports=None if pls.adc_mode == pulsed.MODE_DIRECT else sample_port,
        out_ports=readout_port,
    )
    if JPA:
        pls.hardware.set_lmx(jpa_pump_freq, jpa_pump_pwr)
        mla.lockin.set_dc_offset(bias_port, jpa_bias)
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
        carrier=1,
        frequencies=freq_if,
        phases=0.0,
    )

    # Setup lookup tables for amplitudes
    pls.setup_scale_lut(
        output_ports=readout_port,
        scales=readout_amp,
    )

    # Setup readout and control pulses
    # use setup_long_drive to create a pulse with square envelope
    # turn on the global output scaler
    # setup_long_drive supports smooth rise and fall transitions for the pulse,
    # but we keep it simple here
    readout_pulse = pls.setup_long_drive(
        output_port=readout_port,
        carrier=1,
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
    t_arr, result = pls.run(
        period=T,
        repeat_count=1,
        num_averages=num_averages,
        print_time=True,
    )

t_low = 1500 * 1e-9
t_high = 2000 * 1e-9
t_span = t_high - t_low
idx_low = np.argmin(np.abs(t_arr - t_low))
idx_high = np.argmin(np.abs(t_arr - t_high))
idx = np.arange(idx_low, idx_high)
n_if = int(round(freq_if * t_span))
f_arr = np.fft.rfftfreq(len(t_arr[idx]), 1 / pls.get_fs("adc"))

fig1, ax1 = plt.subplots(2, 1, tight_layout=True)
ax11, ax12 = ax1
ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
ax12.set_facecolor("#dfdfdf")
if pls.adc_mode == pulsed.MODE_DIRECT:
    data = result[0, 0, :]
    data_fft = np.fft.rfft(data[idx]) / len(t_arr[idx])
    ax11.plot(1e9 * t_arr, data)
    ax12.semilogy(1e-6 * f_arr, np.abs(data_fft))
else:
    data_I = result[0, 0, 0, :]
    data_Q = result[0, 0, 1, :]
    data_I_fft = np.fft.rfft(data_I[idx]) / len(t_arr[idx])
    data_Q_fft = np.fft.rfft(data_Q[idx]) / len(t_arr[idx])
    data_L_fft, data_H_fft = untwist_downconversion(data_I_fft, data_Q_fft)
    ax11.plot(1e9 * t_arr, data_I, label="I", c="tab:blue")
    ax11.plot(1e9 * t_arr, data_Q, label="Q", c="tab:orange")
    ax12.semilogy(1e-6 * f_arr, np.abs(data_L_fft), c="tab:green", label="L")
    ax12.semilogy(1e-6 * f_arr, np.abs(data_H_fft), c="tab:red", label="H")
    ax12.semilogy(1e-6 * f_arr[n_if], np.abs(data_L_fft[n_if]), '.', c="tab:green", ms=12)
    ax12.semilogy(1e-6 * f_arr[n_if], np.abs(data_H_fft[n_if]), '.', c="tab:red", ms=12)
    ax11.legend()
    ax12.legend()
ax11.set_xlabel("Time [ns]")
ax12.set_xlabel("Frequency [MHz]")
fig1.show()
