# -*- coding: utf-8 -*-
"""
Find |e> -> |f> transition with two-tone spectroscopy with Pulsed mode.
"""

import ast
from typing import Literal, overload

import h5py
import numpy as np
import numpy.typing as npt

from presto import pulsed
from presto.utils import rotate_opt, sin2

from _base import PlsBase


class TwoToneEF(PlsBase):
    def __init__(
        self,
        readout_freq: float,
        ge_freq: float,
        alpha_center: float,
        alpha_span: float,
        alpha_nr: int,
        readout_amp: float,
        ge_amp: float,
        ef_amp: float,
        readout_duration: float,
        ge_duration: float,
        ef_duration: float,
        sample_duration: float,
        readout_port: int,
        control_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
        jpa_params: dict | None = None,
    ) -> None:
        self.readout_freq = readout_freq
        self.ge_freq = ge_freq
        self.alpha_center = alpha_center
        self.alpha_span = alpha_span
        self.alpha_nr = alpha_nr
        self.readout_amp = readout_amp
        self.ge_amp = ge_amp
        self.ef_amp = ef_amp
        self.readout_duration = readout_duration
        self.ge_duration = ge_duration
        self.ef_duration = ef_duration
        self.sample_duration = sample_duration
        self.readout_port = readout_port
        self.control_port = control_port
        self.sample_port = sample_port
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run
        self.alpha_arr = None  # replaced by run

        self.jpa_params = jpa_params

    def run(
        self,
        presto_address: str,
        presto_port: int | None = None,
        ext_ref_clk: bool = False,
    ) -> str:
        with pulsed.Pulsed(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            **self.DC_PARAMS,
        ) as pls:
            # figure out frequencies
            max_freq = pls.get_fs("dac") / 2  # fits in sideband
            assert self.alpha_span > 0
            half = self.alpha_span / 2
            assert abs(self.alpha_center) + half < max_freq
            assert abs(self.alpha_center) - half > 0
            alpha_start = self.alpha_center - half
            alpha_stop = self.alpha_center + half
            self.alpha_arr = np.linspace(alpha_start, alpha_stop, self.alpha_nr)

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
            pls.hardware.configure_mixer(freq=self.ge_freq, out_ports=self.control_port)

            self._jpa_setup(pls)

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************
            # Setup lookup tables for frequencies that we sweep
            if self.alpha_center >= 0.0:
                phase_q = -np.pi / 2  # USB
            else:
                phase_q = +np.pi / 2  # LSB
            pls.setup_freq_lut(
                self.control_port,
                group=1,
                frequencies=np.abs(self.alpha_arr),
                phases=np.full_like(self.alpha_arr, 0.0),
                phases_q=np.full_like(self.alpha_arr, phase_q),
            )

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(self.readout_port, group=0, scales=self.readout_amp)
            pls.setup_scale_lut(self.control_port, group=0, scales=self.ge_amp)
            pls.setup_scale_lut(self.control_port, group=1, scales=self.ef_amp)

            # Setup readout and control pulses
            # use setup_flat_pulse to create a pulse with square envelope
            # setup_flat_pulse supports smooth rise and fall transitions for the pulse,
            # but we keep it simple here
            readout_pulse = pls.setup_flat_pulse(
                self.readout_port,
                group=0,
                duration=self.readout_duration,
            )

            # For the control pulse we create a sine-squared envelope,
            # and use setup_template to use the user-defined envelope
            # number of samples in the control template
            ge_ns = int(round(self.ge_duration * pls.get_fs("dac")))
            ge_envelope = sin2(ge_ns)
            ge_pulse = pls.setup_template(
                self.control_port,
                group=0,
                template=ge_envelope + 1j * ge_envelope,
                envelope=False,
            )

            ef_ns = int(round(self.ef_duration * pls.get_fs("dac")))
            ef_envelope = sin2(ef_ns)
            ef_pulse = pls.setup_template(
                self.control_port,
                group=1,
                template=ef_envelope + 1j * ef_envelope,
                envelope=True,
            )

            # Setup sampling window
            pls.setup_store(self.sample_port, self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...

            # |g> --> |e>
            pls.reset_phase(T, self.control_port, 0)
            pls.output_pulse(T, ge_pulse)
            T += self.ge_duration

            # |e> --> |f>
            pls.reset_phase(T, self.control_port, 1)
            pls.output_pulse(T, ef_pulse)
            T += self.ef_duration

            # readout
            pls.reset_phase(T, self.readout_port, 0)
            pls.output_pulse(T, readout_pulse)  # Readout pulse
            pls.store(T + self.readout_sample_delay)  # Sampling window
            T += self.readout_duration

            # Move to next ef frequency
            pls.next_frequency(T, self.control_port, 1)

            # wait for decay
            T += self.wait_delay

            T = self._jpa_tweak(T, pls)

            # **************************
            # *** Run the experiment ***
            # **************************
            # repeat the whole sequence `control_freq_nr` times
            # then average `num_averages` times
            pls.run(period=T, repeat_count=self.alpha_nr, num_averages=self.num_averages)
            self.t_arr, self.store_arr = pls.get_store_data()

            self._jpa_stop(pls)

        return self.save()

    def save(self, save_filename: str | None = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "TwoToneEF":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = float(h5f.attrs["readout_freq"])  # type: ignore
            ge_freq = float(h5f.attrs["ge_freq"])  # type: ignore
            alpha_center = float(h5f.attrs["alpha_center"])  # type: ignore
            alpha_span = float(h5f.attrs["alpha_span"])  # type: ignore
            alpha_nr = int(h5f.attrs["alpha_nr"])  # type: ignore
            readout_amp = float(h5f.attrs["readout_amp"])  # type: ignore
            ge_amp = float(h5f.attrs["ge_amp"])  # type: ignore
            ef_amp = float(h5f.attrs["ef_amp"])  # type: ignore
            readout_duration = float(h5f.attrs["readout_duration"])  # type: ignore
            ge_duration = float(h5f.attrs["ge_duration"])  # type: ignore
            ef_duration = float(h5f.attrs["ef_duration"])  # type: ignore
            sample_duration = float(h5f.attrs["sample_duration"])  # type: ignore
            readout_port = int(h5f.attrs["readout_port"])  # type: ignore
            control_port = int(h5f.attrs["control_port"])  # type: ignore
            sample_port = int(h5f.attrs["sample_port"])  # type: ignore
            wait_delay = float(h5f.attrs["wait_delay"])  # type: ignore
            readout_sample_delay = float(h5f.attrs["readout_sample_delay"])  # type: ignore
            num_averages = int(h5f.attrs["num_averages"])  # type: ignore

            jpa_params: dict = ast.literal_eval(h5f.attrs["jpa_params"])  # type: ignore

            t_arr: npt.NDArray[np.float64] = h5f["t_arr"][()]  # type: ignore
            store_arr: npt.NDArray[np.complex128] = h5f["store_arr"][()]  # type: ignore
            alpha_arr: npt.NDArray[np.float64] = h5f["alpha_arr"][()]  # type: ignore

        self = cls(
            readout_freq=readout_freq,
            ge_freq=ge_freq,
            alpha_center=alpha_center,
            alpha_span=alpha_span,
            alpha_nr=alpha_nr,
            readout_amp=readout_amp,
            ge_amp=ge_amp,
            ef_amp=ef_amp,
            readout_duration=readout_duration,
            ge_duration=ge_duration,
            ef_duration=ef_duration,
            sample_duration=sample_duration,
            readout_port=readout_port,
            control_port=control_port,
            sample_port=sample_port,
            wait_delay=wait_delay,
            readout_sample_delay=readout_sample_delay,
            num_averages=num_averages,
            jpa_params=jpa_params,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr
        self.alpha_arr = alpha_arr

        return self

    @overload
    def analyze(self, *, all_plots: bool = False, batch: Literal[True]) -> float: ...

    @overload
    def analyze(self, *, all_plots: bool = False, batch: bool = False): ...

    def analyze(self, *, all_plots: bool = False, batch: bool = False):
        assert self.t_arr is not None
        assert self.store_arr is not None
        assert self.alpha_arr is not None

        from scipy.optimize import curve_fit

        # Analyze
        idx_low, idx_high = self._store_idx_analysis()
        resp_arr = np.mean(self.store_arr[:, 0, idx_low:idx_high], axis=-1)
        data = rotate_opt(resp_arr)

        data_max = np.abs(data).max()
        unit = ""
        mult = 1.0
        if data_max < 1e-6:
            unit = "n"
            mult = 1e9
        elif data_max < 1e-3:
            unit = "μ"
            mult = 1e6
        elif data_max < 1e0:
            unit = "m"
            mult = 1e3

        try:
            data_min = data.real.min()
            data_max = data.real.max()
            data_rng = data_max - data_min
            p0 = [self.alpha_center, self.alpha_span / 4, data_rng, data_min]
            popt, _ = curve_fit(_gaussian, self.alpha_arr, data.real, p0)
            print(f"f0 = {popt[0]} Hz")
            print(f"sigma = {abs(popt[1])} Hz")
        except Exception:
            print("fit failed")
            popt = None

        if not batch:
            import matplotlib.pyplot as plt

            ret_fig = []
            if all_plots:
                # Plot raw store data for first iteration as a check
                t_low, t_high = self._store_t_analysis()
                fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
                ax11, ax12 = ax1
                ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
                ax12.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
                ax11.plot(1e9 * self.t_arr, np.abs(self.store_arr[0, 0, :]))
                ax12.plot(1e9 * self.t_arr, np.angle(self.store_arr[0, 0, :]))
                ax12.set_xlabel("Time [ns]")
                fig1.show()
                ret_fig.append(fig1)

            if all_plots:
                fig2, ax2 = plt.subplots(4, 1, sharex=True, figsize=(6.4, 6.4), tight_layout=True)
                ax21, ax22, ax23, ax24 = ax2
                ax21.plot(1e-6 * self.alpha_arr, mult * np.abs(data))
                ax22.plot(1e-6 * self.alpha_arr, np.angle(data))
                ax23.plot(1e-6 * self.alpha_arr, mult * np.real(data))
                ax24.plot(1e-6 * self.alpha_arr, mult * np.imag(data))
                if popt is not None:
                    ax23.plot(
                        1e-6 * self.alpha_arr,
                        mult * _gaussian(self.alpha_arr, *popt),
                        "--",
                    )

                ax21.set_ylabel(f"Amplitude [{unit:s}FS]")
                ax22.set_ylabel("Phase [rad]")
                ax23.set_ylabel(f"I [{unit:s}FS]")
                ax24.set_ylabel(f"Q [{unit:s}FS]")
                ax2[-1].set_xlabel("Anharmonicity [MHz]")
                fig2.show()
                ret_fig.append(fig2)

            # bigger plot just for I quadrature
            fig3, ax3 = plt.subplots(tight_layout=True)
            ax3.plot(1e-6 * self.alpha_arr, mult * np.real(data), ".")
            if popt is not None:
                ax3.plot(1e-6 * self.alpha_arr, mult * _gaussian(self.alpha_arr, *popt), "--")
            ax3.set_ylabel(f"I quadrature [{unit:s}FS]")
            ax3.set_xlabel(r"Anharmonicity α / 2π [MHz]")
            ax3.grid()
            fig3.show()
            ret_fig.append(fig3)

            return ret_fig

        else:
            assert popt is not None
            f0 = popt[0]
            return float(f0)


def _gaussian(x, x0, s, a, o):
    return a * np.exp(-0.5 * ((x - x0) / s) ** 2) + o
