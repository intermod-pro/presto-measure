# -*- coding: utf-8 -*-
"""AC-Stark shift and measurement-induced dephasing.

Measure Ramsey oscillations while driving the resonator with variable power.
"""
import ast
from typing import List

import h5py
import numpy as np

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
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


class AcStarkShift(Base):
    def __init__(
        self,
        readout_freq: float,
        control_freq: float,
        readout_amp: float,
        control_amp: float,
        readout_duration: float,
        control_duration: float,
        sample_duration: float,
        ringup_duration: float,
        delay_arr: List[float],
        ringup_amp_arr: List[float],
        readout_port: int,
        control_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
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
        self.ringup_duration = ringup_duration
        self.delay_arr = np.atleast_1d(delay_arr).astype(np.float64)
        self.ringup_amp_arr = np.atleast_1d(ringup_amp_arr).astype(np.float64)
        self.readout_port = readout_port
        self.control_port = control_port
        self.sample_port = sample_port
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages
        self.jpa_params = jpa_params
        self.drag = drag

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run

    def run(
        self,
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

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for frequencies
            # we only need to use carrier 1
            pls.setup_freq_lut(
                output_ports=self.readout_port,
                group=0,
                frequencies=0.0,
                phases=0.0,
                phases_q=0.0,
            )
            pls.setup_freq_lut(
                output_ports=self.readout_port,
                group=1,  # for ringup
                frequencies=0.0,
                phases=0.0,
                phases_q=0.0,
            )
            pls.setup_freq_lut(
                output_ports=self.control_port,
                group=0,
                frequencies=0.0,
                phases=0.0,
                phases_q=0.0,
            )

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(
                output_ports=self.readout_port,
                group=0,
                scales=self.readout_amp,
            )
            pls.setup_scale_lut(
                output_ports=self.readout_port,
                group=1,
                scales=self.ringup_amp_arr,
            )
            pls.setup_scale_lut(
                output_ports=self.control_port,
                group=0,
                scales=self.control_amp,
            )

            # Setup readout and control pulses
            readout_pulse = pls.setup_long_drive(
                output_port=self.readout_port,
                group=0,
                duration=self.readout_duration,
            )
            ringup_pulse = pls.setup_long_drive(
                output_port=self.readout_port,
                group=1,
                duration=self.ringup_duration,  # will be extended during sequence
            )
            control_ns = int(
                round(self.control_duration * pls.get_fs("dac"))
            )  # number of samples in the control template
            control_envelope = sin2(control_ns, drag=self.drag)
            control_pulse = pls.setup_template(
                output_port=self.control_port,
                group=0,
                template=control_envelope,
                template_q=control_envelope if self.drag == 0.0 else None,
                envelope=True,
            )

            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            for delay in self.delay_arr:
                # ringup cavity
                # NOTE: adjust phase here so that it matches the reset phase below?
                #       actually since we use zero IF reset phase is a no-op...
                #       so it doesn't really matter!
                new_duration = self.ringup_duration + 2 * self.control_duration + delay
                ringup_pulse.set_total_duration(new_duration)
                pls.output_pulse(T, ringup_pulse)
                T += self.ringup_duration
                # first pi/2 pulse
                pls.reset_phase(T, self.control_port)
                pls.output_pulse(T, control_pulse)
                T += self.control_duration
                # Ramsey delay
                T += delay
                # second pi/2 pulse
                pls.output_pulse(T, control_pulse)
                T += self.control_duration
                # Readout
                # NOTE: reset phase for readout pulse here
                pls.reset_phase(T, self.readout_port)
                pls.output_pulse(T, readout_pulse)
                pls.store(T + self.readout_sample_delay)
                T += self.readout_duration
                # Move to next iteration, waiting for decay
                T += self.wait_delay
            pls.next_scale(T, self.readout_port, 1)
            T += self.wait_delay

            if self.jpa_params is not None:
                # adjust period to minimize effect of JPA idler
                idler_freq = self.jpa_params["pump_freq"] - self.readout_freq
                idler_if = abs(idler_freq - self.readout_freq)  # NCO at readout_freq
                idler_period = 1 / idler_if
                T_clk = int(round(T * pls.get_clk_f()))
                idler_period_clk = int(round(idler_period * pls.get_clk_f()))
                # first make T a multiple of idler period
                if T_clk % idler_period_clk > 0:
                    T_clk += idler_period_clk - (T_clk % idler_period_clk)
                # then make it off by one clock cycle
                T_clk += 1
                T = T_clk * pls.get_clk_T()

            # **************************
            # *** Run the experiment ***
            # **************************
            nr_amps = len(self.ringup_amp_arr)
            pls.run(
                period=T,
                repeat_count=nr_amps,
                num_averages=self.num_averages,
                print_time=True,
            )
            self.t_arr, self.store_arr = pls.get_store_data()

            if self.jpa_params is not None:
                pls.hardware.set_lmx(0.0, 0.0, self.jpa_params["pump_port"])
                pls.hardware.set_dc_bias(0.0, self.jpa_params["bias_port"])

        return self.save()

    def save(self, save_filename: str = None) -> str:
        return super().save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "AcStarkShift":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = h5f.attrs["readout_freq"]
            control_freq = h5f.attrs["control_freq"]
            readout_amp = h5f.attrs["readout_amp"]
            control_amp = h5f.attrs["control_amp"]
            readout_duration = h5f.attrs["readout_duration"]
            control_duration = h5f.attrs["control_duration"]
            sample_duration = h5f.attrs["sample_duration"]
            ringup_duration = h5f.attrs["ringup_duration"]
            delay_arr = h5f["delay_arr"][()]
            ringup_amp_arr = h5f["ringup_amp_arr"][()]
            readout_port = h5f.attrs["readout_port"]
            control_port = h5f.attrs["control_port"]
            sample_port = h5f.attrs["sample_port"]
            wait_delay = h5f.attrs["wait_delay"]
            readout_sample_delay = h5f.attrs["readout_sample_delay"]
            num_averages = h5f.attrs["num_averages"]
            drag = h5f.attrs["drag"]

            jpa_params = ast.literal_eval(h5f.attrs["jpa_params"])

            t_arr = h5f["t_arr"][()]
            store_arr = h5f["store_arr"][()]

        self = cls(
            readout_freq=readout_freq,
            control_freq=control_freq,
            readout_amp=readout_amp,
            control_amp=control_amp,
            readout_duration=readout_duration,
            control_duration=control_duration,
            sample_duration=sample_duration,
            ringup_duration=ringup_duration,
            delay_arr=delay_arr,
            ringup_amp_arr=ringup_amp_arr,
            readout_port=readout_port,
            control_port=control_port,
            sample_port=sample_port,
            wait_delay=wait_delay,
            readout_sample_delay=readout_sample_delay,
            num_averages=num_averages,
            jpa_params=jpa_params,
            drag=drag,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr

        return self

    def analyze(self, all_plots: bool = False):
        if self.t_arr is None:
            raise RuntimeError
        if self.store_arr is None:
            raise RuntimeError

        import matplotlib.pyplot as plt

        ret_fig = []

        idx = np.arange(IDX_LOW, IDX_HIGH)
        t_low = self.t_arr[IDX_LOW]
        t_high = self.t_arr[IDX_HIGH]

        if all_plots:
            # Plot raw store data for first iteration as a check
            fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
            ax11, ax12 = ax1
            ax11.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
            ax12.axvspan(1e9 * t_low, 1e9 * t_high, facecolor="#dfdfdf")
            ax11.plot(1e9 * self.t_arr, np.abs(self.store_arr[0, 0, :]))
            ax12.plot(1e9 * self.t_arr, np.angle(self.store_arr[0, 0, :]))
            ax12.set_xlabel("Time [ns]")
            fig1.show()
            ret_fig.append(fig1)

        # analyze and reshape data
        resp_arr = np.mean(self.store_arr[:, 0, idx], axis=-1)
        data = rotate_opt(resp_arr).real
        nr_amps = len(self.ringup_amp_arr)
        nr_delays = len(self.delay_arr)
        data.shape = (nr_amps, nr_delays)

        fig2, ax2 = plt.subplots(tight_layout=True)
        for ii, amp in enumerate(self.ringup_amp_arr):
            ax2.plot(1e6 * self.delay_arr, data[ii, :], label=str(amp))
        ax2.set_ylabel("I [FS]")
        ax2.set_xlabel("Ramsey delay [μs]")
        ax2.legend()
        fig2.show()
        ret_fig.append(fig2)

        # Ramsey fits
        popt_all = np.zeros((nr_amps, 5))
        perr_all = np.zeros((nr_amps, 5))
        for ii, amp in enumerate(self.ringup_amp_arr):
            try:
                popt, perr = _fit_simple(self.delay_arr, data[ii, :])
                popt_all[ii, :] = popt
                perr_all[ii, :] = perr
            except Exception as err:
                print(f"Unable to fit data for amp nr {ii}: {amp}!")
                print(err)
                popt_all[ii, :] = np.nan
                perr_all[ii, :] = np.nan

        ringup_pwr_arr = self.ringup_amp_arr**2
        freq_arr = popt_all[:, 3]
        gamma_arr = popt_all[:, 2]
        freq_err_arr = perr_all[:, 3]
        gamma_err_arr = perr_all[:, 2]

        # linear fit
        pfit_freq = np.polyfit(ringup_pwr_arr, freq_arr, 1)
        pfit_gamma = np.polyfit(ringup_pwr_arr, gamma_arr, 1)

        chi_div_kappa = pfit_gamma[0] / (2 * np.pi * pfit_freq[0]) / 4
        print(f"χ / κ = {chi_div_kappa}")

        kappa = 2 * np.pi * 455.13e3
        chi = -2 * np.pi * 302.25e3
        eta = kappa**2 / (kappa**2 + 4 * chi**2)
        print((2 * np.pi * pfit_freq[0]) / (2 * eta * chi))
        print(pfit_gamma[0] / (4 * eta * chi**2 / kappa))

        # kappa = 2. * np.pi * 904e3  # from circle fit
        # chi = chi_kappa * kappa

        # ph_n = (np.polyval(pfit_omegas, power) - pfit_omegas[1]) / 2 / chi

        fig3, ax3 = plt.subplots(2, 1, sharex=True, tight_layout=True)
        ax31, ax32 = ax3
        ax31.errorbar(
            1e6 * ringup_pwr_arr, 1e-3 * freq_arr, yerr=1e-3 * freq_err_arr, fmt=".", capsize=1
        )
        ax32.errorbar(
            1e6 * ringup_pwr_arr, 1e-3 * gamma_arr, yerr=1e-3 * gamma_err_arr, fmt=".", capsize=1
        )
        ax31.plot(1e6 * ringup_pwr_arr, 1e-3 * np.polyval(pfit_freq, ringup_pwr_arr), ls="--")
        ax32.plot(1e6 * ringup_pwr_arr, 1e-3 * np.polyval(pfit_gamma, ringup_pwr_arr), ls="--")
        ax31.set_ylabel("ω / 2π [kHz]")
        ax32.set_ylabel("Γ2 [kHz]")
        ax32.set_xlabel("Ringup power [μFS]")
        fig3.show()
        ret_fig.append(fig3)

        return ret_fig


def _func(t, offset, amplitude, gamma, frequency, phase):
    return offset + amplitude * np.exp(-gamma * t) * np.cos(2.0 * np.pi * frequency * t + phase)


def _fit_simple(x, y):
    from scipy.optimize import curve_fit

    pkpk = np.max(y) - np.min(y)
    offset = np.min(y) + pkpk / 2
    amplitude = 0.5 * pkpk
    T2 = 0.5 * (np.max(x) - np.min(x))
    gamma = 1 / T2
    freqs = np.fft.rfftfreq(len(x), x[1] - x[0])
    fft = np.fft.rfft(y)
    fft[0] = 0
    idx_max = np.argmax(np.abs(fft))
    frequency = freqs[idx_max]
    phase = np.angle(fft[idx_max])
    p0 = (
        offset,
        amplitude,
        gamma,
        frequency,
        phase,
    )
    popt, pcov = curve_fit(
        _func,
        x,
        y,
        p0=p0,
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr
