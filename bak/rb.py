# -*- coding: utf-8 -*-
"""
Randomized benchmarking of a single qubit.

Use π/2 pulses (SX gates) and virtual Z gates (RZ gates).
Requires third-party package qiskit.
"""
import glob
import os
import sys
import time
from typing import List, Tuple

import h5py

# import matplotlib.pyplot as plt
import numpy as np

# from scipy.optimize import curve_fit
# from scipy.signal import remez, freqz, filtfilt

from qiskit.circuit import QuantumCircuit, instruction
from qiskit.circuit.library import Barrier, Measure, RZGate, SXGate
from qiskit.compiler import transpile
from qiskit.ignis.verification import randomized_benchmarking_seq

from presto import pulsed
from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto.utils import rotate_opt, sin2

from _base import Base

DAC_CURRENT = 32_000  # uA
CONVERTER_CONFIGURATION = {
    "adc_mode": AdcMode.Mixed,
    "adc_fsample": AdcFSample.G4,
    "dac_mode": [DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
    "dac_fsample": [DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
}
IDX_LOW = 1_500
IDX_HIGH = 2_000


class Rb(Base):
    def __init__(
        self,
        readout_freq: float,
        control_freq: float,
        readout_amp: float,
        control_amp: float,
        readout_duration: float,
        control_duration: float,
        sample_duration: float,
        readout_port: int,
        control_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
        rb_len_arr: List[int],
        rb_nr_realizations: int,
        jpa_params: dict = None,
        drag: float = 0.0,
    ) -> None:
        self.readout_freq = readout_freq
        self.control_freq = control_freq
        self.readout_amp = readout_amp
        self.control_amp = control_amp
        self.readout_duration = readout_duration
        self.control_duration = control_duration
        self.sample_duration = sample_duration
        self.readout_port = readout_port
        self.control_port = control_port
        self.sample_port = sample_port
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages
        self.rb_len_arr = np.atleast_1d(rb_len_arr).astype(np.int64)
        self.rb_nr_realizations = rb_nr_realizations
        self.drag = drag

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run
        self.rb_sequences = []  # replaced by run

        self.jpa_params = jpa_params

    def run(
        self,
        presto_address: str,
        presto_port: int = None,
        ext_ref_clk: bool = False,
    ) -> str:
        rb_nr_lengths = len(self.rb_len_arr)
        print("Generating random sequences...")
        self.rb_sequences = self._rbgen()
        assert len(self.rb_sequences) == self.rb_nr_realizations
        assert len(self.rb_sequences[0]) == rb_nr_lengths
        print("Done!")

        cnt = 0
        tot = self.rb_nr_realizations * rb_nr_lengths
        outer = []
        for i, a in enumerate(self.rb_sequences):
            inner = []
            outer.append(inner)
            for j, seq in enumerate(a):
                cnt = cnt + 1
                print()
                print(f"****** {cnt}/{tot} ******")
                t_arr, (data_i, data_q) = self._run_inner(
                    seq, presto_address, presto_port, ext_ref_clk
                )
                inner.append(data_i + 1j * data_q)

        result = np.array(outer)

    def _run_inner(
        self,
        seq,
        presto_address: str,
        presto_port: int = None,
        ext_ref_clk: bool = False,
    ) -> str:
        # Instantiate interface class
        with pulsed.Pulsed(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            **CONVERTER_CONFIGURATION,
        ) as pls:
            assert pls.hardware is not None

            pls.hardware.set_adc_attenuation(self.sample_port, 0.0)
            pls.hardware.set_dac_current(self.readout_port, DAC_CURRENT)
            pls.hardware.set_dac_current(self.control_port, DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.readout_port, 0)
            pls.hardware.set_inv_sinc(self.control_port, 0)
            pls.hardware.configure_mixer(
                freq=self.readout_freq,
                in_ports=self.sample_port,
                out_ports=self.readout_port,
                sync=False,  # sync in next call
            )
            pls.hardware.configure_mixer(
                freq=self.control_freq,
                out_ports=self.control_port,
                sync=True,  # sync here
            )
            if self.jpa_params is not None:
                pls.hardware.set_lmx(
                    self.jpa_params["pump_freq"],
                    self.jpa_params["pump_pwr"],
                    self.jpa_params["pump_port"],
                )
                pls.hardware.set_dc_bias(self.jpa_params["bias"], self.jpa_params["bias_port"])
                pls.hardware.sleep(1.0, False)

    def _rbgen(self):
        rb_circs, xdata = randomized_benchmarking_seq(
            nseeds=self.rb_nr_realizations,  # Number of seeds (random sequences)
            length_vector=self.rb_len_arr,  # Number of Cliffords in the sequence
            rb_pattern=[[0]],  # 1-qubit RB sequence on qubit Q0 (default pattern)
        )
        # rb_circs is List[List[QuantumCircuit]]
        # with "shape"=(nseeds, len(length_vector))

        result = []
        for circuits in rb_circs:
            # circuits is List[QuantumCircuit]
            # with len=len(length_vector)
            outer = list()
            result.append(outer)
            for circuit in circuits:
                # circuit is QuantumCircuit
                inner = list()
                outer.append(inner)
                circuit_t: QuantumCircuit = transpile(
                    circuit, basis_gates=["rz", "sx"], optimization_level=3
                )
                for g in circuit_t.data:
                    instruction = g[0]
                    if isinstance(instruction, RZGate):
                        param = float(instruction.params[0])
                        iparam = int(round(2 * param / np.pi))
                        if np.abs(iparam * np.pi / 2 - float(instruction.params[0])) > 1e-6:
                            print(param, iparam, iparam * np.pi / 2)
                            print("Rounding error!")
                        inner.append(("Z", iparam))
                        # print('Z', end='')
                        # print(instruction.params, end='')
                    elif isinstance(instruction, SXGate):
                        inner.append(("X",))
                        # print('X', end='')
                    elif isinstance(instruction, Barrier):
                        pass
                        # print('Barrier')
                    elif isinstance(instruction, Measure):
                        pass
                        # print('Measure')
                    else:
                        print("Unknown")
        return result


def setup_template_matching():
    template_filename = "template.h5"
    with h5py.File(template_filename, "r") as h5f:
        match_t_in_store = h5f.attrs["match_t_in_store"]
        readout_sample_delay = h5f.attrs["readout_sample_delay"]
        template_g = h5f["template_g"][()]
        template_e = h5f["template_e"][()]
        threshold = 0.5 * (np.sum(np.abs(template_e) ** 2) - np.sum(np.abs(template_g) ** 2))

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

    pls.reset_phase(T, ReadoutPort[0])
    pls.output_pulse(T, ReadoutPulse[0])
    pls.match(T + readout_sample_delay + match_t_in_store, [match_g, match_e])


def read_template_from_file():
    with h5py.File("template.h5", "w") as h5f:
        h5f.attrs["readout_sample_delay"] = 0.5e-6
        h5f.attrs["match_t_in_store"] = 0.5e-6
        h5f.create_dataset("template_g", data=np.ones(128))
        h5f.create_dataset("template_e", data=np.ones(128))


def run_sequence(seq, settings):
    with pulsed.Pulsed(
        dry_run=settings["dry_run"],
        address="130.237.35.90",
        port=42874,
        adc_mode=AdcMode.Mixed,
        adc_fsample=AdcFSample.G2,
        dac_mode=[DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
        dac_fsample=[DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
    ) as pls:
        pls.hardware.set_adc_attenuation(settings["sample_port"], 0.0)
        pls.hardware.set_dac_current(settings["readout_port"], 32_000)
        pls.hardware.set_dac_current(settings["control_port"], 32_000)
        pls.hardware.set_inv_sinc(settings["readout_port"], 0)
        pls.hardware.set_inv_sinc(settings["control_port"], 0)
        pls.hardware.configure_mixer(
            settings["qubit_frequency"][0], out_ports=settings["control_port"][0], sync=False
        )
        pls.hardware.configure_mixer(
            settings["readout_frequency"][0],
            out_ports=settings["readout_port"][0],
            in_ports=settings["sample_port"],
        )

        ReadoutPulse = [
            pls.setup_template(
                settings["readout_port"][0],
                0,
                settings["readout_envelope"],
                settings["readout_envelope"],
            ),
        ]

        XTemplates = [
            pls.setup_template(
                settings["control_port"][0],
                0,
                np.real(settings["control_envelope"]),
                np.imag(settings["control_envelope"]),
            ),
            pls.setup_template(
                settings["control_port"][0],
                0,
                np.imag(settings["control_envelope"]),
                -np.real(settings["control_envelope"]),
            ),
            pls.setup_template(
                settings["control_port"][0],
                0,
                -np.real(settings["control_envelope"]),
                -np.imag(settings["control_envelope"]),
            ),
            pls.setup_template(
                settings["control_port"][0],
                0,
                -np.imag(settings["control_envelope"]),
                np.real(settings["control_envelope"]),
            ),
            # pls.setup_template(settings["control_port"][0], 0, settings["control_envelope"], settings["control_envelope"]),
            # pls.setup_template(settings["control_port"][0], 0, settings["control_envelope"], -settings["control_envelope"]),
            # pls.setup_template(settings["control_port"][0], 0, -settings["control_envelope"], -settings["control_envelope"]),
            # pls.setup_template(settings["control_port"][0], 0, -settings["control_envelope"], settings["control_envelope"]),
        ]

        pls.set_store_ports(settings["sample_port"])
        pls.set_store_duration(settings["store_duration"])

        pls.setup_scale_lut(settings["control_port"], 0, 1.0)
        pls.setup_scale_lut(settings["control_port"], 1, 1.0)
        pls.setup_scale_lut(settings["readout_port"], 0, 1.0)

        Tgate = XTemplates[0].get_duration()
        T = 0
        vphase = 0

        # reset phase here if using IF

        pulse_count = 0
        for g in seq:
            if g[0] == "Z":
                vphase = (vphase + g[1]) % 4
            elif g[0] == "X":
                # print(f"X({vphase}) at {T}")
                if pulse_count < 500:
                    pls.output_pulse(T, XTemplates[vphase])
                else:
                    pls.output_pulse(T, XTemplates[vphase + 4])
                pulse_count = pulse_count + 1
                T += Tgate
            else:
                print("Unknown")

        print(pulse_count)

        # reset phase here if using IF
        pls.output_pulse(T, ReadoutPulse[0])
        pls.store(T)

        period = T + 501e-6 * 2
        pls.run(period, 1, settings["nr_averages"], verbose=False)

        try:
            return pls.get_store_data()
        except:
            return np.arange(1000), np.ones(1000) * len(seq)


def run_rb():
    settings = {}
    settings["dry_run"] = False
    settings["nr_averages"] = 1_000
    settings["control_envelope"] = drag.dragpulse(0.5)[1] * 0.446729
    # settings["control_envelope"] = drag.dragpulse(0.0)[1] * 0.446729
    settings["readout_envelope"] = np.ones(2000) * 0.4
    settings["qubit_frequency"] = np.array(
        [
            4.087703443926388 * 1e9 + 98454,
        ]
    )
    settings["readout_frequency"] = np.array(
        [
            6.028_100 * 1e9,
        ]
    )
    settings["control_port"] = np.array(
        [
            4,
        ]
    )
    settings["readout_port"] = np.array(
        [
            1,
        ]
    )
    settings["sample_port"] = np.array(
        [
            1,
        ]
    )
    settings["store_duration"] = 4e-6

    settings["rb_lengths"] = np.array(
        [0, 1, 2, 5, 10, 20, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450]
    )
    settings["rb_iterations"] = 100

    USE_JPA = True

    jpa_bias_port = 1
    jpa_pump_freq = 2 * 6.031e9  # Hz
    jpa_pump_pwr = 9  # lmx units
    jpa_bias = +0.449  # V
    jpa_params = (
        {
            "jpa_bias": jpa_bias,
            "jpa_bias_port": jpa_bias_port,
            "jpa_pump_freq": jpa_pump_freq,
            "jpa_pump_pwr": jpa_pump_pwr,
        }
        if USE_JPA
        else None
    )

    print("Generating gate sequences...")
    rb = rbgen(settings["rb_lengths"][1:], settings["rb_iterations"], [[0]])
    print("Done!")
    # rb = [[
    #     [('X',), ('X',)],
    # ]]

    if jpa_params is not None:
        with hardware.Hardware(address="130.237.35.90", port=42874) as pls:
            pls.init_clock(False)
            pls.set_lmx(jpa_params["jpa_pump_freq"], jpa_params["jpa_pump_pwr"])
            set_dc_bias(jpa_params["jpa_bias_port"], jpa_params["jpa_bias"])
            time.sleep(1.0)

    try:
        cnt = 0
        tot = len(rb) * (len(rb[0]) + 1)
        outer = []
        for i, a in enumerate(rb):
            inner = []
            outer.append(inner)
            cnt = cnt + 1
            print()
            print(f"****** {cnt}/{tot} ******")
            t_arr, (data_i, data_q) = run_sequence((), settings)
            inner.append(data_i + 1j * data_q)
            for j, seq in enumerate(a):
                cnt = cnt + 1
                print()
                print(f"****** {cnt}/{tot} ******")
                t_arr, (data_i, data_q) = run_sequence(seq, settings)
                inner.append(data_i + 1j * data_q)

        result = np.array(outer)
    finally:
        if jpa_params is not None:
            with hardware.Hardware(address="130.237.35.90", port=42874) as pls:
                pls.set_lmx(0.0, 0.0)
                set_dc_bias(jpa_params["jpa_bias_port"], 0.0)
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

        for key, value in settings.items():
            if isinstance(value, np.ndarray):
                h5f.create_dataset(key, data=value)
            else:
                h5f.attrs[key] = value

        h5f.create_dataset("t_arr", data=t_arr)
        h5f.create_dataset("store_arr", data=result)
    print(f"Data saved to: {save_path}")


def lowpass(s):
    # b = firwin(256, 2e6, fs=1e9, pass_zero=True)
    b = remez(256, [0, 4e6, 5e6, 0.5 * 1e9], [1, 0], Hz=1e9)
    # w, h = freqz(b, fs=1e9)
    # plt.plot(w, 20*np.log10(np.abs(h)))
    # plt.show()
    return filtfilt(b, 1, s)


def exp_fit_fn(x, A, B, C):
    return B + (A - B) * C**x


def get_filename(s):
    if s is None:
        s = "0"

    if os.path.isfile(s):
        return s

    index = -1 - int(s)

    script_path = os.path.realpath(__file__)  # full path of current script
    current_dir, script_basename = os.path.split(script_path)
    script_filename = os.path.splitext(script_basename)[0]  # name of current script
    datafile_pattern = os.path.join(current_dir, "data", script_filename + "*")
    names = glob.glob(datafile_pattern)
    return sorted(names)[index]


def load_rb(arg):
    load_filename = get_filename(arg)
    print(load_filename)

    with h5py.File(load_filename, "r") as h5f:
        rb_lengths = h5f["rb_lengths"][()]
        t_arr = h5f["t_arr"][()]
        result = h5f["store_arr"][()]
        control_envelope = h5f["control_envelope"][()]

    result = lowpass(result)
    if False:
        fig, ax = plt.subplots(
            result.shape[0], result.shape[1], sharex=True, sharey=True, squeeze=False
        )
        for i, inner in enumerate(result):
            for j, d in enumerate(inner):
                real = np.real(d[0, 0, :])
                imag = np.imag(d[0, 0, :])
                ax[i][j].plot(t_arr, real)
                ax[i][j].plot(t_arr, imag)
                ax[i][j].axvline(1.54e-6)
                ax[i][j].axvline(2.14e-6)
        fig.show()

    range_ = (t_arr >= 1.54e-6) & (t_arr < 2.14e-6)
    result_average = np.average(result[:, :, 0, 0, range_], axis=-1)
    rotated = np.real(rotate_opt(result_average))
    rotated_avg = np.average(rotated, axis=0)

    fig, ax = plt.subplots(tight_layout=True)
    for d in rotated:
        ax.plot(rb_lengths, 1e3 * d, ".", c="tab:gray", alpha=0.1)
    ax.plot(rb_lengths, 1e3 * rotated_avg, ".", ms=9)
    # ax[1].plot(control_envelope.real)
    # ax[1].plot(control_envelope.imag)

    try:
        popt, pcov = curve_fit(
            exp_fit_fn, rb_lengths, rotated_avg, p0=(rotated_avg[0], rotated_avg[-1], 0.99)
        )
        perr = np.sqrt(np.diag(pcov))
        ax.plot(rb_lengths, 1e3 * exp_fit_fn(rb_lengths, *popt), "--")
        print(popt)
        alpha = popt[-1]
        alpha_std = perr[-1]
        alpha_rel = alpha_std / alpha
        r = (1 - alpha) / 2
        r_rel = alpha_rel
        r_std = r * r_rel
        print(f"EPC: {r:e} +/- {r_std:e}")
    except RuntimeError:
        r = -1
        print("Failed to fit exponential")

    ax.set_xlabel("Number of Cliffords")
    ax.set_ylabel("I quadrature [mFS]")
    # ax.set_title(f"{load_filename}, EPC = {np.round(r, 5)}")
    ax.set_title(f"EPC = {r:.1e}                F = {1 - r:.3%}")
    fig.show()

    return fig


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rb.py run")
        print("       python rb.py load [file]")
    elif sys.argv[1] == "run":
        run_rb()
        fig = load_rb(None)
    elif sys.argv[1] == "load":
        if len(sys.argv) > 2:
            fig = load_rb(sys.argv[2])
        else:
            fig = load_rb(None)
    else:
        print(f"invalid argument: {sys.argv[1]}")

# s = sin2(20*10)
# s = np.pad(s, (1024, 1024)) / s.shape[0]
# fig, ax = plt.subplots(2)
# ax[0].plot(s)
# ax[1].plot(20*np.log10(np.abs(np.fft.rfft(s))))
# fig.show()
