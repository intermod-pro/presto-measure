"""
Randomized benchmarking of a single qubit.

Use π/2 pulses (SX gates) and virtual Z gates (RZ gates).
Requires third-party packages:
  - qiskit 0.45.1
  - qiskit_experiments 0.5.4
"""

from __future__ import annotations

import ast
import time

import h5py
import numpy as np
import numpy.typing as npt
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_experiments.library import StandardRB

from presto import pulsed
from presto.utils import asarray, rotate_opt, sin2

from _base import PlsBase

Gate = tuple[str, int]
GateSeq = list[Gate]

IntAny = int | list[int] | npt.NDArray[np.integer]


class Rb(PlsBase):
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
        rb_len_arr: IntAny,
        rb_nr_realizations: int,
        drag: float = 0.0,
        jpa_params: dict | None = None,
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
        self.rb_len_arr = asarray(rb_len_arr, np.int64)
        self.rb_nr_realizations = rb_nr_realizations
        self.drag = drag
        self.jpa_params = jpa_params

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run
        self._rb_sequences: list[list[GateSeq]] = []  # replaced by run

    def run(
        self,
        presto_address: str,
        presto_port: int | None = None,
        ext_ref_clk: bool = False,
    ) -> str:
        rb_nr_lengths = len(self.rb_len_arr)
        print("Generating random sequences, this might take a while...")
        self._rb_sequences = self._rbgen()
        assert len(self._rb_sequences) == self.rb_nr_realizations
        assert len(self._rb_sequences[0]) == rb_nr_lengths
        print("Done!")

        cnt = 0
        tot = self.rb_nr_realizations * rb_nr_lengths
        samples_per_store = int(round(self.sample_duration * 1e9))
        self.store_arr = np.zeros(
            (self.rb_nr_realizations, rb_nr_lengths, samples_per_store), np.complex128
        )
        for i, a in enumerate(self._rb_sequences):
            for j, seq in enumerate(a):
                cnt = cnt + 1
                print()
                print(f"****** {cnt}/{tot} ******")
                self.t_arr, data = self._run_sequence(
                    seq, presto_address, presto_port, ext_ref_clk
                )
                self.store_arr[i, j, :] = data[0, 0, :]
                time.sleep(0.01)

        return self.save()

    def save(self, save_filename: str | None = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> Rb:
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = float(h5f.attrs["readout_freq"])  # type: ignore
            control_freq = float(h5f.attrs["control_freq"])  # type: ignore
            readout_amp = float(h5f.attrs["readout_amp"])  # type: ignore
            control_amp = float(h5f.attrs["control_amp"])  # type: ignore
            readout_duration = float(h5f.attrs["readout_duration"])  # type: ignore
            control_duration = float(h5f.attrs["control_duration"])  # type: ignore
            sample_duration = float(h5f.attrs["sample_duration"])  # type: ignore
            readout_port = int(h5f.attrs["readout_port"])  # type: ignore
            control_port = int(h5f.attrs["control_port"])  # type: ignore
            sample_port = int(h5f.attrs["sample_port"])  # type: ignore
            wait_delay = float(h5f.attrs["wait_delay"])  # type: ignore
            readout_sample_delay = float(h5f.attrs["readout_sample_delay"])  # type: ignore
            num_averages = int(h5f.attrs["num_averages"])  # type: ignore
            rb_len_arr: npt.NDArray[np.int64] = h5f["rb_len_arr"][()]  # type: ignore
            rb_nr_realizations = int(h5f.attrs["rb_nr_realizations"])  # type: ignore
            drag = float(h5f.attrs["drag"])  # type: ignore

            try:
                jpa_params: dict | None = ast.literal_eval(h5f.attrs["jpa_params"])  # type: ignore
            except KeyError:
                jpa_params = None

            t_arr: npt.NDArray[np.float64] = h5f["t_arr"][()]  # type: ignore
            store_arr: npt.NDArray[np.complex128] = h5f["store_arr"][()]  # type: ignore

        self = cls(
            readout_freq=readout_freq,
            control_freq=control_freq,
            readout_amp=readout_amp,
            control_amp=control_amp,
            readout_duration=readout_duration,
            control_duration=control_duration,
            sample_duration=sample_duration,
            readout_port=readout_port,
            control_port=control_port,
            sample_port=sample_port,
            wait_delay=wait_delay,
            readout_sample_delay=readout_sample_delay,
            num_averages=num_averages,
            rb_len_arr=rb_len_arr,
            rb_nr_realizations=rb_nr_realizations,
            drag=drag,
            jpa_params=jpa_params,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr

        return self

    def _run_sequence(
        self,
        sequence: GateSeq,
        presto_address: str,
        presto_port: int | None = None,
        ext_ref_clk: bool = False,
    ):
        with pulsed.Pulsed(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            **self.DC_PARAMS,
        ) as pls:
            pls.hardware.set_adc_attenuation(self.sample_port, self.ADC_ATTENUATION)
            pls.hardware.set_dac_current(self.readout_port, self.DAC_CURRENT)
            pls.hardware.set_dac_current(self.control_port, self.DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.readout_port, 0)
            pls.hardware.set_inv_sinc(self.control_port, 0)

            pls.hardware.configure_mixer(
                self.readout_freq,
                in_ports=self.sample_port,
                out_ports=self.readout_port,
            )
            pls.hardware.configure_mixer(self.control_freq, out_ports=self.control_port)

            self._jpa_setup(pls)

            pls.setup_scale_lut(self.readout_port, 0, self.readout_amp)
            # we lose 3 dB compared to e.g. rabi_amp, so increase control amplitude
            pls.setup_scale_lut(self.control_port, 0, self.control_amp * np.sqrt(2))

            readout_pulse = pls.setup_flat_pulse(
                self.readout_port,
                group=0,
                duration=self.readout_duration,
            )

            control_ns = int(round(self.control_duration * pls.get_fs("dac")))
            control_envelope = sin2(control_ns, drag=self.drag)
            control_pulses = [
                pls.setup_template(
                    self.control_port,
                    0,
                    np.real(control_envelope),
                    np.imag(control_envelope),
                ),
                pls.setup_template(
                    self.control_port,
                    0,
                    np.imag(control_envelope),
                    -np.real(control_envelope),
                ),
                pls.setup_template(
                    self.control_port,
                    0,
                    -np.real(control_envelope),
                    -np.imag(control_envelope),
                ),
                pls.setup_template(
                    self.control_port,
                    0,
                    -np.imag(control_envelope),
                    np.real(control_envelope),
                ),
            ]

            pls.setup_store(self.sample_port, self.sample_duration)

            T = 0
            vphase: int = 0

            # reset phase on control_port here if using IF
            pulse_count = 0
            for gate in sequence:
                if gate[0] == "rz":
                    vphase = (vphase + gate[1]) % 4
                elif gate[0] == "sx":
                    pls.output_pulse(T, control_pulses[vphase])
                    T += self.control_duration
                    pulse_count = pulse_count + 1
                else:
                    raise NotImplementedError(f"unknown gate {gate}")

            print(f"{pulse_count = }")

            pls.output_pulse(T, readout_pulse)
            pls.store(T + self.readout_sample_delay)

            # wait for qubit decay
            T += self.wait_delay

            T = self._jpa_tweak(T, pls)

            pls.run(T, 1, self.num_averages)
            ret = pls.get_store_data()

            self._jpa_stop(pls)

            return ret

    def _rbgen(self) -> list[list[GateSeq]]:
        return _singlequbitrb(self.rb_len_arr.tolist(), self.rb_nr_realizations)  # pyright: ignore[reportArgumentType]

    def analyze(self):
        if self.t_arr is None:
            raise RuntimeError
        if self.store_arr is None:
            raise RuntimeError

        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit

        idx_low, idx_high = self._store_idx_analysis()
        result = _lowpass(self.store_arr[:, :, idx_low:idx_high])
        result_average = np.average(result, axis=-1)
        rotated = np.real(rotate_opt(result_average))
        rotated_avg = np.average(rotated, axis=0)

        fig, ax = plt.subplots(tight_layout=True)
        for d in rotated:
            ax.plot(self.rb_len_arr, 1e3 * d, ".", c="tab:gray", alpha=0.1)
        ax.plot(self.rb_len_arr, 1e3 * rotated_avg, ".", ms=9)

        try:
            popt, pcov = curve_fit(
                _exp_fit_fn,
                self.rb_len_arr,
                rotated_avg,
                p0=(rotated_avg[0], rotated_avg[-1], 0.99),
            )
            perr = np.sqrt(np.diag(pcov))
            ax.plot(self.rb_len_arr, 1e3 * _exp_fit_fn(self.rb_len_arr, *popt), "--")
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

    def analyze_new(self):
        if self.t_arr is None:
            raise RuntimeError
        if self.store_arr is None:
            raise RuntimeError
        ret_fig = []

        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
        from scipy.optimize import curve_fit

        idx_low, idx_high = self._store_idx_analysis()
        result = _lowpass(self.store_arr[:, :, idx_low:idx_high])
        result_average = np.average(result, axis=-1)
        rotated = np.real(rotate_opt(result_average))
        rotated_avg = np.average(rotated, axis=0)

        popt, pcov = curve_fit(
            _exp_fit_fn, self.rb_len_arr, rotated_avg, p0=(rotated_avg[0], rotated_avg[-1], 0.99)
        )
        A, B, _C = popt
        perr = np.sqrt(np.diag(pcov))
        X = np.linspace(self.rb_len_arr[0], self.rb_len_arr[-1], 1000)
        alpha = popt[-1]
        alpha_std = perr[-1]
        alpha_rel = alpha_std / alpha
        r = (1 - alpha) / 2
        r_rel = alpha_rel
        r_std = r * r_rel
        print(f"EPC: {r:e} +/- {r_std:e}")

        fid = 1.0 - r
        fid_std = r_std
        print(f"F: {fid} +/- {fid_std:e}")
        fid_label = f"$\\mathcal{{F}} = {100 * fid:.3f}\\%$"

        xg = A
        xe = B - 2 * (A - B)

        fig, ax = plt.subplots()
        for d in rotated:
            (line,) = ax.plot(self.rb_len_arr, _rescale(d, xe, xg), ".", c="0.75")
        line.set_label("single realizations")  # pyright: ignore[reportPossiblyUnboundVariable]

        ax.semilogx(self.rb_len_arr, _rescale(rotated_avg, xe, xg), ".", ms=9, label="average")
        ax.semilogx(X, _rescale(_exp_fit_fn(X, *popt), xe, xg), "--", label=fid_label)
        ax.legend(loc="lower left")
        f = ScalarFormatter()
        f.set_scientific(False)
        ax.xaxis.set_major_formatter(f)

        ax.set_xlim(6.6e-1, 6.2e3)
        ax.set_ylim(0.58, 1.02)
        ax.set_yticks([0.6, 0.8, 1.0])

        ax.set_xlabel("Sequence length")
        ax.set_ylabel("$P_\\mathrm{g}$")

        fig.show()
        ret_fig.append(fig)

        rescaled = _rescale(rotated, xe, xg)
        higher = np.percentile(rescaled, 75, axis=0)
        center = np.percentile(rescaled, 50, axis=0)
        lower = np.percentile(rescaled, 25, axis=0)
        err_h = higher - center
        err_l = center - lower
        err = [err_l, err_h]

        fig2, ax2 = plt.subplots()
        (line_fit,) = ax2.semilogx(
            X,
            _rescale(_exp_fit_fn(X, *popt), xe, xg),
            "--",
            c="tab:orange",
            label=fid_label,
        )
        _eb = ax2.errorbar(
            self.rb_len_arr,
            center,
            yerr=err,
            fmt=".",
            ms=9,
            c="tab:blue",
            ecolor="0.75",
            capsize=1,
        )

        (mock_m,) = ax2.plot([], [], ".", ms=9, c="tab:blue")
        mock_eb = ax2.errorbar(
            [],
            [],
            yerr=[],
            fmt=".",
            ms=0,
            c="tab:blue",
            ecolor="0.75",
            capsize=1,
        )

        ax2.set_xscale("log")
        ax2.set_xlabel("Sequence length")
        ax2.set_ylabel("$P_\\mathrm{g}$")
        ax2.xaxis.set_major_formatter(f)

        # ax2.legend(loc="lower left")
        ax2.legend(
            [mock_m, mock_eb, line_fit],
            ["median over realizations", "interquartile range", fid_label],
            loc="lower left",
        )

        fig2.savefig("randomized_benchmarking")
        fig2.show()
        ret_fig.append(fig2)

        return ret_fig


def _singlequbitrb(lengths: list[int], num_samples: int) -> list[list[GateSeq]]:
    qubits = [1]

    exp = StandardRB(qubits, lengths, num_samples=num_samples)
    basis_gates = ["sx", "rz"]
    ct = transpile(exp.circuits(), basis_gates=basis_gates)
    sequences: list[list[GateSeq]] = []
    for i in range(num_samples):
        inner: list[GateSeq] = []
        sequences.append(inner)
        for j in range(len(lengths)):
            inner.append(_translateseq(ct[i * len(lengths) + j]))
    return sequences


def _translateseq(quantum_circuit: QuantumCircuit) -> GateSeq:
    result = []
    for circuit_instruction in quantum_circuit:
        name = circuit_instruction.operation.name
        params = circuit_instruction.operation.params  # type: ignore

        if name == "rz":
            fparam = float(params[0])  # rad
            iparam = int(round(2 * fparam / np.pi))  # units of np.pi/2
            if np.abs(iparam * np.pi / 2 - fparam) > 1e-6:
                raise ValueError(f"rz is not a multiple of π/2: {fparam}")
            result.append(("rz", iparam))
        elif name == "sx":
            result.append(("sx",))
        elif name == "barrier":
            pass
        elif name == "measure":
            pass
        else:
            raise NotImplementedError

    return result


def _lowpass(s):
    from scipy.signal import filtfilt, remez

    # b = firwin(256, 2e6, fs=1e9, pass_zero=True)
    b = remez(256, [0, 4e6, 5e6, 0.5 * 1e9], [1, 0], fs=1e9)
    # w, h = freqz(b, fs=1e9)
    # plt.plot(w, 20*np.log10(np.abs(h)))
    # plt.show()
    return filtfilt(b, 1, s)


def _exp_fit_fn(x, A, B, C):
    return B + (A - B) * C**x


def _rescale(data, min_, max_):
    rng = max_ - min_
    data = data - min_  # make copy
    data /= rng
    return data
