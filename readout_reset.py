# -*- coding: utf-8 -*-
"""Active qubit reset.

Perform pulsed readout starting from ground and excited state using template matching with known
reference traces. Use feedback to correct the state of the qubit.
"""
import ast
from typing import List

import h5py
import numpy as np

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import pulsed
from presto.utils import sin2

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


# WHICH_QUBIT = 2  # 1 (higher resonator) or 2 (lower resonator)
# USE_JPA = False
#
# # Presto's IP address or hostname
# ADDRESS = "130.237.35.90"
# PORT = 42874
# EXT_REF_CLK = False  # set to True to lock to an external reference clock
# jpa_bias_port = 1
#
# if WHICH_QUBIT == 1:
#     readout_freq = 6.166_600 * 1e9  # Hz, frequency for resonator readout
#     control_freq = 3.557_866 * 1e9  # Hz
#     control_amp = 0.1026  # FS <-- pi pulse
#     control_port = 3
#     jpa_pump_freq = 2 * 6.169e9  # Hz
#     jpa_pump_pwr = 11  # lmx units
#     jpa_bias = +0.437  # V
#     template_filename = None
# elif WHICH_QUBIT == 2:
#     readout_freq = 6.028_448 * 1e9  # Hz, frequency for resonator readout
#     control_freq = 4.091_777 * 1e9  # Hz
#     control_amp = 0.1537  # FS <-- pi pulse
#     control_port = 4
#     jpa_pump_freq = 2 * 6.031e9  # Hz
#     jpa_pump_pwr = 9  # lmx units
#     jpa_bias = +0.449  # V
#     template_filename = "data/readout_avg_20210610_140856.h5"
# else:
#     raise ValueError
#
# # cavity drive: readout
# readout_amp = 0.3  # FS
# readout_duration = 2 * 1e-6  # s, duration of the readout pulse
# readout_port = 1
#
# # qubit drive: control
# control_duration = 100 * 1e-9  # s, duration of the control pulse
#
# # cavity readout: sample
# sample_duration = 4 * 1e-6  # s, duration of the sampling window
# sample_port = 1
# readout_sample_delay = 290 * 1e-9 + 0 * 1e-6  # s, delay between readout pulse and sample window to account for latency
#
# # IQ readout experiment
# num_averages = 10_000
# wait_delay = 500e-6  # s, delay between repetitions to allow the qubit to decay
# readout_decay = 2e-6  # s, delay between the two readouts to allow photons to decay
#
# # template matching
# with h5py.File(template_filename, "r") as h5f:
#     match_t_in_store = h5f.attrs["match_t_in_store"]
#     template_g = h5f["template_g"][()]
#     template_e = h5f["template_e"][()]
#     threshold = 0.5 * (np.sum(np.abs(template_e)**2) -
#                        np.sum(np.abs(template_g)**2))


class ReadoutReset(Base):
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
        readout_match_delay: float,
        ref_g: List[complex],
        ref_e: List[complex],
        num_averages: int,
        extra_wait: float = 0.0,
        jpa_params: dict = None,
        drag: float = 0.0,
        clear: dict = None,
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
        self.readout_match_delay = readout_match_delay
        self.ref_g = np.atleast_1d(ref_g).astype(np.complex128)
        self.ref_e = np.atleast_1d(ref_e).astype(np.complex128)
        self.num_averages = num_averages
        self.extra_wait = extra_wait
        self.drag = drag
        self.clear = clear

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run
        self.match_g_arr = None  # replaced by run
        self.match_e_arr = None  # replaced by run

        self.jpa_params = jpa_params

        assert self.ref_g.shape == self.ref_e.shape

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
            pls.setup_freq_lut(
                output_ports=self.readout_port,
                group=0,
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
                scales=1.0,  # set it in pulse
            )
            pls.setup_scale_lut(
                output_ports=self.control_port,
                group=0,
                scales=self.control_amp,
            )

            # Setup readout and control pulses
            # use setup_long_drive to create a pulse with square envelope
            # setup_long_drive supports smooth rise and fall transitions for the pulse,
            # but we keep it simple here
            if self.clear is None:
                readout_pulse = pls.setup_long_drive(
                    output_port=self.readout_port,
                    group=0,
                    duration=self.readout_duration,
                    amplitude=self.readout_amp,
                    amplitude_q=self.readout_amp,
                    rise_time=0e-9,
                    fall_time=0e-9,
                )
            else:
                from presto._clear import clear

                lens, amps = clear(self.readout_duration * 1e9, **self.clear)
                lens = [int(round(l * pls.get_fs("dac"))) for l in lens]

                readout_ns = int(
                    round(self.readout_duration * pls.get_fs("dac"))
                )  # number of samples in the control template
                readout_envelope = np.zeros(readout_ns)
                start = 0
                for l, a in zip(lens, amps):
                    stop = start + l
                    readout_envelope[start:stop] = a
                    start += l
                readout_envelope *= self.readout_amp

                readout_pulse = pls.setup_template(
                    output_port=self.readout_port,
                    group=0,
                    template=readout_envelope,
                    template_q=readout_envelope,
                    envelope=True,
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
            control_pulse_to_g = pls.setup_template(
                output_port=self.control_port,
                group=0,
                template=control_envelope,
                template_q=control_envelope if self.drag == 0.0 else None,
                envelope=True,
            )
            control_pulse_to_e = pls.setup_template(
                output_port=self.control_port,
                group=0,
                template=control_envelope,
                template_q=control_envelope if self.drag == 0.0 else None,
                envelope=True,
            )

            # Setup sampling window
            pls.set_store_ports(self.sample_port)
            pls.set_store_duration(self.sample_duration)

            # Setup template matching
            # threshold = 0.5 * (_norm(self.ref_e)**2 - _norm(self.ref_g)**2)
            threshold = _threshold(self.ref_g, self.ref_e)
            # templ_g = np.zeros(len(self.ref_g) * 2, np.float64)
            # templ_g[0::2] = np.real(self.ref_g)
            # templ_g[1::2] = np.imag(self.ref_g)
            # templ_e = np.zeros(len(self.ref_e) * 2, np.float64)
            # templ_e[0::2] = np.real(self.ref_e)
            # templ_e[1::2] = np.imag(self.ref_e)
            match_g, match_e = pls.setup_template_matching_pair(
                input_port=self.sample_port,
                template1=-self.ref_g,  # NOTE minus sign
                template2=self.ref_e,
                threshold=threshold,
            )  # success when match_e + match_g - threshold > 0

            # Setup feedback
            # make the control pulse conditional
            pls.setup_condition([match_g, match_e], control_pulse_to_g, control_pulse_to_e)

            # ******************************
            # *** Program pulse sequence ***
            # ******************************
            T = 0.0  # s, start at time zero ...
            for ii in range(2):
                pls.reset_phase(T, self.control_port)
                if ii == 0:
                    # init in |g>
                    pass
                else:
                    # init in |e>
                    pls.output_pulse(T, control_pulse)
                T += self.control_duration

                # First readout, store and match
                pls.reset_phase(T, self.readout_port)
                pls.output_pulse(T, readout_pulse)
                pls.store(T + self.readout_sample_delay)
                pls.match(T + self.readout_match_delay, [match_g, match_e])
                # T += self.readout_duration

                # End of match window and of readout pulse
                end_of_match = self.readout_match_delay + match_g.get_duration()
                T += max(end_of_match, self.readout_duration)

                # extra wait
                T += self.extra_wait

                # Conditional pi pulse
                if ii == 0:
                    # reset to |g>
                    pls.output_pulse(T, control_pulse_to_g)
                else:
                    # reset to |e>
                    pls.output_pulse(T, control_pulse_to_e)
                T += self.control_duration

                # # extra wait
                # T += self.extra_wait

                # Second readout, store and match
                # right after conditional pulse
                pls.reset_phase(T, self.readout_port)
                pls.output_pulse(T, readout_pulse)
                # pls.store(T + self.readout_sample_delay)
                pls.match(T + self.readout_match_delay, [match_g, match_e])
                T += self.readout_duration

                # Move to next iteration
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
            pls.run(
                period=T,
                repeat_count=1,
                num_averages=self.num_averages,
                print_time=True,
            )
            self.t_arr, self.store_arr = pls.get_store_data()
            (self.match_g_arr, self.match_e_arr) = pls.get_template_matching_data(
                [match_g, match_e]
            )

            if self.jpa_params is not None:
                pls.hardware.set_lmx(0.0, 0.0, self.jpa_params["pump_port"])
                pls.hardware.set_dc_bias(0.0, self.jpa_params["bias_port"])

        return self.save()

    def save(self, save_filename: str = None) -> str:
        return super().save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "ReadoutReset":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = h5f.attrs["readout_freq"]
            control_freq = h5f.attrs["control_freq"]
            readout_amp = h5f.attrs["readout_amp"]
            control_amp = h5f.attrs["control_amp"]
            readout_duration = h5f.attrs["readout_duration"]
            control_duration = h5f.attrs["control_duration"]
            sample_duration = h5f.attrs["sample_duration"]
            readout_port = h5f.attrs["readout_port"]
            control_port = h5f.attrs["control_port"]
            sample_port = h5f.attrs["sample_port"]
            wait_delay = h5f.attrs["wait_delay"]
            readout_sample_delay = h5f.attrs["readout_sample_delay"]
            readout_match_delay = h5f.attrs["readout_match_delay"]
            num_averages = h5f.attrs["num_averages"]
            extra_wait = h5f.attrs["extra_wait"]
            drag = h5f.attrs["drag"]

            jpa_params = ast.literal_eval(h5f.attrs["jpa_params"])
            clear = ast.literal_eval(h5f.attrs["clear"])

            t_arr = h5f["t_arr"][()]
            store_arr = h5f["store_arr"][()]
            ref_g = h5f["ref_g"][()]
            ref_e = h5f["ref_e"][()]
            match_g_arr = h5f["match_g_arr"][()]
            match_e_arr = h5f["match_e_arr"][()]

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
            readout_match_delay=readout_match_delay,
            ref_g=ref_g,
            ref_e=ref_e,
            num_averages=num_averages,
            extra_wait=extra_wait,
            jpa_params=jpa_params,
            drag=drag,
            clear=clear,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr
        self.match_g_arr = match_g_arr
        self.match_e_arr = match_e_arr

        return self

    def analyze(self, fix_sum: bool = True, logscale: bool = False):
        assert self.t_arr is not None
        assert self.store_arr is not None
        assert self.match_g_arr is not None
        assert self.match_e_arr is not None

        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit

        ret_fig = []

        # plot average trace
        fig1, ax1 = plt.subplots(2, 1, sharex=True, sharey=True, tight_layout=True)
        ax11, ax12 = ax1
        ax11.plot(1e9 * self.t_arr, np.real(self.store_arr[0, 0, :]), label="|g>")
        ax11.plot(1e9 * self.t_arr, np.real(self.store_arr[1, 0, :]), label="|e>")
        ax12.plot(1e9 * self.t_arr, np.imag(self.store_arr[0, 0, :]), label="|g>")
        ax12.plot(1e9 * self.t_arr, np.imag(self.store_arr[1, 0, :]), label="|e>")
        ax11.set_ylabel("Real")
        ax12.set_ylabel("Imaginary")
        ax12.set_xlabel("Time [ns]")
        ax12.legend()
        fig1.show()
        ret_fig.append(fig1)

        threshold = _threshold(self.ref_g, self.ref_e)
        match_diff = (
            self.match_e_arr + self.match_g_arr - threshold
        )  # does |e> match better than |g>?
        match_diff.shape = (
            -1,
            2,
            2,
        )  # -1: repetitions; 2: prepared in |g> or |e>; 2: 1st and 2nd readout
        for idx_excited in [0, 1]:
            match_diff_1 = match_diff[
                :, idx_excited, 0
            ]  # prepared in |g/e>, first readout (before reset)
            match_diff_2 = match_diff[
                :, idx_excited, 1
            ]  # prepared in |g/e>, second readout (after reset)

            idx_low_1 = match_diff_1 < 0
            idx_high_1 = np.logical_not(idx_low_1)
            idx_low_2 = match_diff_2 < 0
            idx_high_2 = np.logical_not(idx_low_2)
            mean_low_1 = match_diff_1[idx_low_1].mean()
            mean_high_1 = match_diff_1[idx_high_1].mean()
            mean_low_2 = match_diff_2[idx_low_2].mean()
            mean_high_2 = match_diff_2[idx_high_2].mean()
            std_low_1 = match_diff_1[idx_low_1].std()
            std_high_1 = match_diff_1[idx_high_1].std()
            std_low_2 = match_diff_2[idx_low_2].std()
            std_high_2 = match_diff_2[idx_high_2].std()
            weight_low_1 = np.sum(idx_low_1) / len(idx_low_1)
            weight_high_1 = 1.0 - weight_low_1
            weight_low_2 = np.sum(idx_low_2) / len(idx_low_2)
            weight_high_2 = 1.0 - weight_low_2
            std = max(std_low_1, std_high_1, std_low_2, std_high_2)
            x_min = min(mean_low_1, mean_low_2) - 5 * std
            x_max = max(mean_high_1, mean_high_2) + 5 * std
            nr_bins = int(round(np.sqrt(match_diff.shape[0])))
            H_1, xedges = np.histogram(
                match_diff_1, bins=nr_bins, range=(x_min, x_max), density=True
            )
            H_2, xedges = np.histogram(
                match_diff_2, bins=nr_bins, range=(x_min, x_max), density=True
            )
            xdata = 0.5 * (xedges[1:] + xedges[:-1])
            dx = xdata[1] - xdata[0]
            ntot = match_diff.shape[0]
            min_dens = 1.0 / ntot / dx

            init_1 = np.array(
                [mean_low_1, std_low_1, weight_low_1, mean_high_1, std_high_1, weight_high_1]
            )
            init_2 = np.array(
                [mean_low_2, std_low_2, weight_low_2, mean_high_2, std_high_2, weight_high_2]
            )
            if fix_sum:
                # skip second weight
                popt_1, pcov_1 = curve_fit(double_gaussian_fixed, xdata, H_1, p0=init_1[:-1])

                def double_gaussian_fixed_no_ms(x, w0):
                    w1 = 1.0 - w0
                    return double_gaussian(x, popt_1[0], popt_1[1], w0, popt_1[3], popt_1[4], w1)

                # popt_2, pcov_2 = curve_fit(double_gaussian_fixed, xdata, H_2, p0=init_2[:-1])
                _popt_2, pcov_2 = curve_fit(double_gaussian_fixed_no_ms, xdata, H_2, p0=init_2[2])
                # add back second weight for ease of use
                popt_1 = np.r_[popt_1, 1.0 - popt_1[2]]
                # popt_2 = np.r_[popt_2, 1.0 - popt_2[2]]
                popt_2 = popt_1.copy()
                popt_2[2] = _popt_2[0]
                popt_2[5] = 1 - _popt_2[0]
            else:
                popt_1, pcov_1 = curve_fit(double_gaussian, xdata, H_1, p0=init_1)
                popt_2, pcov_2 = curve_fit(double_gaussian, xdata, H_2, p0=init_2)

            # *** Effective temperature ***
            # Teff_1 = Planck * control_freq / (Boltzmann * np.log(1 / popt_1[5] - 1))
            # Teff_2 = Planck * control_freq / (Boltzmann * np.log(1 / popt_2[5] - 1))
            Teff_1 = t_eff(popt_1[5], self.control_freq)
            Teff_2 = t_eff(popt_2[5], self.control_freq)

            print("Before reset:")
            print(f"  |e>: {popt_1[5]:5.1%}")
            print(f"  T_e: {1e3 * Teff_1:.1f}mK")
            print("After reset:")
            print(f"  |e>: {popt_2[5]:5.1%}")
            print(f"  T_e: {1e3 * Teff_2:.1f}mK")

            # *** Readout fidelity ***
            err_eg = error(popt_1[0], popt_1[1])  # measure |e> but it was |g>
            err_ge = error(popt_1[3], popt_1[4])  # measure |g> but it was |e>
            fidelity = 1.0 - 0.5 * (err_eg + err_ge)
            print(f"{err_eg = }")
            print(f"{err_ge = }")
            print(f"{fidelity = }")
            # fidelity_g = 0.5 * (1 + erf((0.0 - popt_g[0]) / np.sqrt(2 * popt_g[1]**2)))
            # fidelity_e = 1.0 - 0.5 * (1 + erf(
            #     (0.0 - popt_e[3]) / np.sqrt(2 * popt_e[4]**2)))

            fig2, ax2 = plt.subplots(2, 1, sharex=True, sharey=True)
            ax21, ax22 = ax2
            for ax_ in ax2:
                ax_.axvline(0.0, c="0.75")
                # ax_.axhline(0.0, c="0.5")

            # hist_color = transparent(0x1f77b4, 0.5)
            # hist_color = f"#{hist_color:06x}"
            hist_color = "0.75"
            hist_plot(ax21, H_1, xedges, lw=1, c=hist_color)
            # ax21.plot(xdata, double_gaussian(xdata, *popt_1), c="k")
            ax21.plot(
                xdata,
                single_gaussian(xdata, *popt_1[:3]),
                ls="-",
                c="tab:blue",
                label=f"$\\left|\\mathrm{{g}}\\right>$: {popt_1[2]:.1%}",
            )
            ax21.plot(
                xdata,
                single_gaussian(xdata, *popt_1[3:]),
                ls="-",
                c="tab:orange",
                label=f"$\\left|\\mathrm{{e}}\\right>$:   {popt_1[5]:.1%}",
            )

            hist_plot(ax22, H_2, xedges, lw=1, c=hist_color)
            # ax22.plot(xdata, double_gaussian(xdata, *popt_2), c="k")
            ax22.plot(
                xdata,
                single_gaussian(xdata, *popt_2[:3]),
                ls="-",
                c="tab:blue",
                label=f"$\\left|\\mathrm{{g}}\\right>$: {popt_2[2]:.1%}",
            )
            ax22.plot(
                xdata,
                single_gaussian(xdata, *popt_2[3:]),
                ls="-",
                c="tab:orange",
                label=f"$\\left|\\mathrm{{e}}\\right>$:   {popt_2[5]:.1%}",
            )

            # ax21.set_title(f"Before reset: $T_\\mathrm{{eff}}$ = {1e3*Teff_1:.0f} mK")
            ax21.set_ylabel("Norm. counts")
            legend_loc = "upper left" if idx_excited else "upper right"
            ax21.legend(title="before reset", ncol=1, loc=legend_loc)
            # ax22.set_xlabel("Comparator result")
            ax22.set_xlabel(
                r"$\left< s, \tau_\mathrm{e} \right> - \left< s, \tau_{g} \right> - \theta_{eg}$"
            )
            ax22.set_ylabel("Norm. counts")
            # ax22.set_title(f"After reset: $T_\\mathrm{{eff}}$ = {1e3*Teff_2:.0f} mK")
            ax22.legend(title="after reset", ncol=1, loc=legend_loc)

            if logscale:
                ax21.set_yscale("log")
                ymin = np.log10(min_dens)
                ymax = np.log10(max(H_1.max(), H_2.max()))
                yrng = ymax - ymin
                ax21.set_ylim(10 ** (ymin - 0.05 * yrng), 10 ** (ymax + 0.05 * yrng))

            if idx_excited:
                ax21.text(
                    0.98, 0.95, "(a)", fontsize=10, va="top", ha="right", transform=ax21.transAxes
                )
                ax22.text(
                    0.98, 0.95, "(b)", fontsize=10, va="top", ha="right", transform=ax22.transAxes
                )
            else:
                ax21.text(0.02, 0.95, "(a)", fontsize=10, va="top", transform=ax21.transAxes)
                ax22.text(0.02, 0.95, "(b)", fontsize=10, va="top", transform=ax22.transAxes)

            # fig2.savefig("reset")
            fig2.savefig("reset_e.png")
            fig2.show()
            ret_fig.append(fig2)

        return ret_fig


# def _inprod(f, g, t=None, dt=None):
#     if t is not None:
#         dt = t[1] - t[0]
#         ns = len(t)
#         T = ns * dt
#     elif dt is not None:
#         ns = len(f)
#         T = ns * dt
#     else:
#         T = 1.
#     return np.trapz(f * np.conj(g), x=t) / T


# def _norm(x, t=None, dt=None):
#     return np.sqrt(np.real(_inprod(x, x, t=t, dt=dt)))


def _threshold(ref1, ref2):
    return 0.5 * (np.sum(np.abs(ref2) ** 2) - np.sum(np.abs(ref1) ** 2))


def single_gaussian(x, m, s, w):
    return w * np.exp(-((x - m) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


def double_gaussian(x, m0, s0, w0, m1, s1, w1):
    return single_gaussian(x, m0, s0, w0) + single_gaussian(x, m1, s1, w1)


def double_gaussian_fixed(x, m0, s0, w0, m1, s1):
    w1 = 1.0 - w0
    return double_gaussian(x, m0, s0, w0, m1, s1, w1)


def transparent(rgb, alpha):
    r = (rgb >> 16) & 0xFF
    g = (rgb >> 8) & 0xFF
    b = rgb & 0xFF
    new_r = int(round(0xFF * alpha + r * (1 - alpha)))
    new_g = int(round(0xFF * alpha + g * (1 - alpha)))
    new_b = int(round(0xFF * alpha + b * (1 - alpha)))
    new_rgb = new_b
    new_rgb |= new_g << 8
    new_rgb |= new_r << 16
    return new_rgb


def error(m, s):
    from scipy.special import erf

    x = abs(m) / (np.sqrt(2) * s)
    # if m < 0.0:
    #     return 0.5 * (1 + erf(x))
    # else:
    return 0.5 * (1 - erf(x))


def t_eff(p, fq):
    from scipy.constants import Boltzmann, Planck

    return Planck * fq / (Boltzmann * np.log(1 / p - 1))


def hist_plot(ax, spec, bin_ar, **kwargs):
    x_quad = np.zeros((len(bin_ar) - 1) * 4)  # length 4*N
    # bin_ar[0:-1] takes all but the rightmost element of bin_ar -> length N
    x_quad[::4] = bin_ar[0:-1]  # for lower left,  (x,0)
    x_quad[1::4] = bin_ar[0:-1]  # for upper left,  (x,y)
    x_quad[2::4] = bin_ar[1:]  # for upper right, (x+1,y)
    x_quad[3::4] = bin_ar[1:]  # for lower right, (x+1,0)

    y_quad = np.zeros(len(spec) * 4)
    y_quad[1::4] = spec  # for upper left,  (x,y)
    y_quad[2::4] = spec  # for upper right, (x+1,y)

    return ax.plot(x_quad, y_quad, **kwargs)
