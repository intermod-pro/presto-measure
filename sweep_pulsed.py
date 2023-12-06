# -*- coding: utf-8 -*-
"""Pulsed frequency sweep on the resonator."""
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt

from presto.hardware import AdcMode, DacMode
from presto import pulsed
from presto.utils import untwist_downconversion

from _base import Base

DAC_CURRENT = 32_000  # uA
IDX_LOW = 0
IDX_HIGH = -1


class SweepPulsed(Base):
    def __init__(
        self,
        readout_freq_center: float,
        readout_freq_span: float,
        readout_freq_nr: int,
        readout_amp: float,
        readout_duration: float,
        sample_duration: float,
        readout_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
    ) -> None:
        self.readout_freq_center = readout_freq_center
        self.readout_freq_span = readout_freq_span
        self.readout_freq_nr = readout_freq_nr
        self.readout_amp = readout_amp
        self.readout_duration = readout_duration
        self.sample_duration = sample_duration
        self.readout_port = readout_port
        self.sample_port = sample_port
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages

        self.readout_freq_arr = None  # replaced by run
        self.readout_if_arr = None  # replaced by run
        self.readout_nco = None  # replaced by run
        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run

    def run(
        self,
        presto_address: str,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
    ) -> str:
        # Instantiate interface class
        with pulsed.Pulsed(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            adc_mode=AdcMode.Mixed,
            dac_mode=DacMode.Mixed,
        ) as pls:
            # figure out frequencies
            assert self.readout_freq_center > (self.readout_freq_span / 2)
            assert self.readout_freq_span < pls.get_fs("dac") / 2  # fits in HSB
            readout_if_center = pls.get_fs("dac") / 4  # middle of HSB
            readout_if_start = readout_if_center - self.readout_freq_span / 2
            readout_if_stop = readout_if_center + self.readout_freq_span / 2
            self.readout_if_arr = np.linspace(
                readout_if_start, readout_if_stop, self.readout_freq_nr
            )
            self.readout_nco = self.readout_freq_center - readout_if_center
            self.readout_freq_arr = self.readout_nco + self.readout_if_arr

            pls.hardware.set_adc_attenuation(self.sample_port, 0.0)
            pls.hardware.set_dac_current(self.readout_port, DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.readout_port, 0)

            pls.hardware.configure_mixer(
                freq=self.readout_nco,
                in_ports=self.sample_port,
                out_ports=self.readout_port,
                sync=True,
            )  # sync here

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************
            # Setup lookup tables for frequencies we sweep
            pls.setup_freq_lut(
                self.readout_port,
                group=0,
                frequencies=self.readout_if_arr,
                phases=np.full_like(self.readout_if_arr, 0.0),
                phases_q=np.full_like(self.readout_if_arr, -np.pi / 2),
            )  # HSB

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(self.readout_port, group=0, scales=self.readout_amp)

            # Setup readout
            # use setup_long_drive to create a pulse with square envelope
            # setup_long_drive supports smooth rise and fall transitions for the pulse,
            # but we keep it simple here
            readout_pulse = pls.setup_long_drive(
                self.readout_port,
                group=0,
                duration=self.readout_duration,
                amplitude=1.0 + 1j,
                envelope=True,
            )

            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            pls.reset_phase(T, self.readout_port)
            pls.output_pulse(T, readout_pulse)  # Readout pulse
            pls.store(T + self.readout_sample_delay)
            T += self.readout_duration
            pls.next_frequency(T, self.readout_port)
            T += self.wait_delay  # wait for decay

            # **************************
            # *** Run the experiment ***
            # **************************
            pls.run(period=T, repeat_count=self.readout_freq_nr, num_averages=self.num_averages)
            self.t_arr, self.store_arr = pls.get_store_data()

        return self.save()

    def save(self, save_filename: Optional[str] = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "SweepPulsed":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq_center = float(h5f.attrs["readout_freq_center"])  # type: ignore
            readout_freq_span = float(h5f.attrs["readout_freq_span"])  # type: ignore
            readout_freq_nr = int(h5f.attrs["readout_freq_nr"])  # type: ignore
            readout_amp = float(h5f.attrs["readout_amp"])  # type: ignore
            readout_duration = float(h5f.attrs["readout_duration"])  # type: ignore
            sample_duration = float(h5f.attrs["sample_duration"])  # type: ignore
            readout_port = int(h5f.attrs["readout_port"])  # type: ignore
            sample_port = int(h5f.attrs["sample_port"])  # type: ignore
            wait_delay = float(h5f.attrs["wait_delay"])  # type: ignore
            readout_sample_delay = float(h5f.attrs["readout_sample_delay"])  # type: ignore
            num_averages = int(h5f.attrs["num_averages"])  # type: ignore
            readout_nco = float(h5f.attrs["readout_nco"])  # type: ignore

            readout_freq_arr: npt.NDArray[np.float64] = h5f["readout_freq_arr"][()]  # type: ignore
            readout_if_arr: npt.NDArray[np.float64] = h5f["readout_if_arr"][()]  # type: ignore
            t_arr: npt.NDArray[np.float64] = h5f["t_arr"][()]  # type: ignore
            store_arr: npt.NDArray[np.complex128] = h5f["store_arr"][()]  # type: ignore

        self = cls(
            readout_freq_center=readout_freq_center,
            readout_freq_span=readout_freq_span,
            readout_freq_nr=readout_freq_nr,
            readout_amp=readout_amp,
            readout_duration=readout_duration,
            sample_duration=sample_duration,
            readout_port=readout_port,
            sample_port=sample_port,
            wait_delay=wait_delay,
            readout_sample_delay=readout_sample_delay,
            num_averages=num_averages,
        )
        self.readout_freq_arr = readout_freq_arr
        self.readout_if_arr = readout_if_arr
        self.readout_nco = readout_nco
        self.t_arr = t_arr
        self.store_arr = store_arr

        return self

    def analyze(self, all_plots: bool = False):
        assert self.t_arr is not None
        assert self.store_arr is not None
        assert self.readout_freq_arr is not None
        assert self.readout_if_arr is not None
        assert self.readout_nco is not None
        assert len(self.readout_freq_arr) == self.readout_freq_nr
        assert len(self.readout_if_arr) == self.readout_freq_nr

        import matplotlib.pyplot as plt

        try:
            from resonator_tools import circuit

            _has_resonator_tools = True
        except ImportError:
            _has_resonator_tools = False

        ret_fig = []

        idx = np.arange(IDX_LOW, IDX_HIGH)
        t_low = self.t_arr[IDX_LOW]
        t_high = self.t_arr[IDX_HIGH]
        nr_samples = IDX_HIGH - IDX_LOW

        if all_plots:
            # Plot raw store data for first iteration as a check
            fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
            ax11, ax12 = ax1
            ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
            ax12.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
            ax11.plot(
                1e9 * self.t_arr, np.abs(self.store_arr[int(self.readout_freq_nr / 2), 0, :])
            )
            ax12.plot(
                1e9 * self.t_arr, np.angle(self.store_arr[int(self.readout_freq_nr / 2), 0, :])
            )
            ax12.set_xlabel("Time [ns]")
            fig1.show()
            ret_fig.append(fig1)

        # Analyze
        data = self.store_arr[:, 0, idx]
        data.shape = (self.readout_freq_nr, 1, nr_samples)
        resp_I_arr = np.zeros((1, self.readout_freq_nr), np.complex128)
        resp_Q_arr = np.zeros((1, self.readout_freq_nr), np.complex128)
        dt = self.t_arr[1] - self.t_arr[0]
        t = dt * np.arange(nr_samples)
        for ii, readout_if in enumerate(self.readout_if_arr):
            cos = np.cos(2 * np.pi * readout_if * t)
            sin = np.sin(2 * np.pi * readout_if * t)
            for jj in range(1):
                data_slice = data[ii, jj, :]
                # TODO: low-pass filter the demodulated signal?
                I_real = np.sum(data_slice.real * cos) / nr_samples
                I_imag = -np.sum(data_slice.real * sin) / nr_samples
                resp_I_arr[jj, ii] = I_real + 1j * I_imag
                Q_real = np.sum(data_slice.imag * cos) / nr_samples
                Q_imag = -np.sum(data_slice.imag * sin) / nr_samples
                resp_Q_arr[jj, ii] = Q_real + 1j * Q_imag

        _, resp_H_arr = untwist_downconversion(resp_I_arr, resp_Q_arr)
        resp_dB = 20 * np.log10(np.abs(resp_H_arr))
        resp_phase = np.angle(resp_H_arr)
        # resp_phase *= -1
        resp_phase = np.unwrap(resp_phase, axis=-1)
        N = self.readout_freq_nr // 4
        idx = np.zeros(self.readout_freq_nr, bool)
        idx[:N] = True
        idx[-N:] = True
        pfit_g = np.polyfit(self.readout_freq_arr[idx], resp_phase[0, idx], 1)
        background = np.polyval(pfit_g, self.readout_freq_arr)
        resp_phase[0, :] -= background

        print("----------------")
        if _has_resonator_tools:
            # port_g = circuit.reflection_port(
            #    self.readout_freq_arr, resp_H_arr[0, :] * np.exp(-1j * background)
            # )
            port_g = circuit.notch_port(  # pyright: ignore [reportUnboundVariable]
                self.readout_freq_arr, resp_H_arr[0, :] * np.exp(-1j * background)
            )
            port_g.autofit(electric_delay=-6.1e-9)

            f_g = port_g.fitresults["fr"]

            print(f"ω_g / 2π = {f_g * 1e-9:.6f} GHz")
        print("----------------")

        fig2, ax2 = plt.subplots(2, 1, sharex=True, tight_layout=True, figsize=(6.4, 6.4))
        ax21, ax22 = ax2

        for ax_ in ax2:
            if _has_resonator_tools:
                ax_.axvline(1e-9 * f_g, ls="--", c="tab:red", alpha=0.5)  # pyright: ignore [reportUnboundVariable]

        ax21.plot(1e-9 * self.readout_freq_arr, resp_dB[0, :], c="tab:blue", label="|g>")
        ax22.plot(1e-9 * self.readout_freq_arr, resp_phase[0, :], c="tab:blue")

        if _has_resonator_tools:
            ax21.plot(
                1e-9 * port_g.f_data,  # pyright: ignore
                20 * np.log10(np.abs(port_g.z_data_sim)),  # pyright: ignore [reportUnboundVariable]
                c="tab:red",
                ls="--",
            )
            ax22.plot(1e-9 * port_g.f_data, np.angle(port_g.z_data_sim), c="tab:red", ls="--")  # pyright: ignore

        ax21.set_ylabel("Amplitude [dBFS]")
        ax22.set_ylabel("Phase [rad]")
        ax2[-1].set_xlabel("Readout frequency [GHz]")
        ax21.legend(ncol=2, loc="lower right")
        fig2.show()
        ret_fig.append(fig2)

        return ret_fig
