# -*- coding: utf-8 -*-
"""Single-shot readout.

Perform pulsed readout starting from ground and excited state using template matching with known
reference traces. Compare with standard IQ readout.
"""

import ast
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt

from presto import pulsed
from presto.utils import sin2

from _base import PlsBase


class SingleShot(PlsBase):
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
        ref_g: npt.NDArray[np.complex128],
        ref_e: npt.NDArray[np.complex128],
        num_averages: int,
        jpa_params: Optional[dict] = None,
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
        self.readout_match_delay = readout_match_delay
        self.ref_g = np.atleast_1d(ref_g).astype(np.complex128)
        self.ref_e = np.atleast_1d(ref_e).astype(np.complex128)
        self.num_averages = num_averages
        self.drag = drag

        self.t_arr = None  # replaced by run
        self.store_arr = None  # replaced by run
        self.match_g_arr = None  # replaced by run
        self.match_e_arr = None  # replaced by run
        self.match_i_arr = None  # replaced by run
        self.match_q_arr = None  # replaced by run

        self.jpa_params = jpa_params

        assert self.ref_g.shape == self.ref_e.shape

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
            **self.DC_PARAMS,
        ) as pls:
            pls.hardware.set_adc_attenuation(self.sample_port, self.ADC_ATTENUATION)
            pls.hardware.set_dac_current(self.readout_port, self.DAC_CURRENT)
            pls.hardware.set_dac_current(self.control_port, self.DAC_CURRENT)
            pls.hardware.set_inv_sinc(self.readout_port, 0)
            pls.hardware.set_inv_sinc(self.control_port, 0)
            pls.hardware.configure_mixer(
                freq=self.readout_freq,
                in_ports=self.sample_port,
                out_ports=self.readout_port,
            )
            pls.hardware.configure_mixer(
                freq=self.control_freq,
                out_ports=self.control_port,
            )

            self._jpa_setup(pls)

            # ************************************
            # *** Setup measurement parameters ***
            # ************************************

            # Setup lookup tables for amplitudes
            pls.setup_scale_lut(self.readout_port, group=0, scales=self.readout_amp)
            pls.setup_scale_lut(self.control_port, group=0, scales=self.control_amp)

            # Setup readout and control pulses
            # use setup_long_drive to create a pulse with square envelope
            # setup_long_drive supports smooth rise and fall transitions for the pulse,
            # but we keep it simple here
            readout_pulse = pls.setup_long_drive(
                self.readout_port,
                group=0,
                duration=self.readout_duration,
                amplitude=1.0 + 1j,
                envelope=False,
            )

            control_ns = int(
                round(self.control_duration * pls.get_fs("dac"))
            )  # number of samples in the control template
            control_envelope = sin2(control_ns, drag=self.drag)
            control_pulse = pls.setup_template(
                self.control_port,
                group=0,
                template=control_envelope + 1j * control_envelope,
                envelope=False,
            )

            # Setup sampling window
            pls.setup_store(self.sample_port, self.sample_duration)

            # Setup template matching
            # threshold = _threshold(self.ref_g, self.ref_e)
            match_g, match_e = pls.setup_template_matching_pair(
                input_port=self.sample_port,
                # template1=-self.ref_g,  # NOTE: minus sign
                template1=self.ref_g,
                template2=self.ref_e,
                # threshold=threshold,
            )  # success when match_e + match_g - threshold > 0

            len_ref = len(self.ref_g)
            match_i, match_q = pls.setup_template_matching_pair(
                input_port=self.sample_port,
                template1=np.full(len_ref, 1.0 + 0.0j, np.complex128),
                template2=np.full(len_ref, 0.0 + 1.0j, np.complex128),
                # threshold=threshold,
            )  # success when match_e + match_g - threshold > 0

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
                pls.match(T + self.readout_match_delay, [match_g, match_e, match_i, match_q])
                # T += self.readout_duration

                # End of match window and of readout pulse
                end_of_match = self.readout_match_delay + match_g.get_duration()
                T += max(end_of_match, self.readout_duration)

                # Move to next iteration
                T += self.wait_delay

            T = self._jpa_tweak(T, pls)

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
            (self.match_i_arr, self.match_q_arr) = pls.get_template_matching_data(
                [match_i, match_q]
            )

            self._jpa_stop(pls)

        return self.save()

    def save(self, save_filename: Optional[str] = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "SingleShot":
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
            readout_match_delay = float(h5f.attrs["readout_match_delay"])  # type: ignore
            num_averages = int(h5f.attrs["num_averages"])  # type: ignore
            drag = float(h5f.attrs["drag"])  # type: ignore

            jpa_params: dict = ast.literal_eval(h5f.attrs["jpa_params"])  # type: ignore

            t_arr: npt.NDArray[np.float64] = h5f["t_arr"][()]  # type: ignore
            store_arr: npt.NDArray[np.complex128] = h5f["store_arr"][()]  # type: ignore
            ref_g: npt.NDArray[np.complex128] = h5f["ref_g"][()]  # type: ignore
            ref_e: npt.NDArray[np.complex128] = h5f["ref_e"][()]  # type: ignore
            match_g_arr: npt.NDArray[np.float64] = h5f["match_g_arr"][()]  # type: ignore
            match_e_arr: npt.NDArray[np.float64] = h5f["match_e_arr"][()]  # type: ignore
            match_i_arr: npt.NDArray[np.float64] = h5f["match_i_arr"][()]  # type: ignore
            match_q_arr: npt.NDArray[np.float64] = h5f["match_q_arr"][()]  # type: ignore

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
            jpa_params=jpa_params,
            drag=drag,
        )
        self.t_arr = t_arr
        self.store_arr = store_arr
        self.match_g_arr = match_g_arr
        self.match_e_arr = match_e_arr
        self.match_i_arr = match_i_arr
        self.match_q_arr = match_q_arr

        return self

    def analyze(self, fix_sum: bool = True, logscale: bool = False, rotate: bool = False):
        assert self.t_arr is not None
        assert self.store_arr is not None
        assert self.match_g_arr is not None
        assert self.match_e_arr is not None
        assert self.match_i_arr is not None
        assert self.match_q_arr is not None

        import matplotlib.pyplot as plt

        ret_fig = []

        if rotate:
            ref_g, ref_e, x = _rotate_opt(self.ref_g, self.ref_e)
        else:
            ref_g = self.ref_g
            ref_e = self.ref_e
            x = 0.0

        t0_ref = self.readout_match_delay - self.readout_sample_delay
        dt = self.t_arr[1] - self.t_arr[0]
        t_arr_ref = t0_ref + dt * np.arange(len(self.ref_g))

        # plot average trace
        fig1, ax1 = plt.subplots(2, 1, sharex=True, sharey=True, constrained_layout=True)
        ax11, ax12 = ax1
        ax11.plot(1e9 * self.t_arr, np.real(np.exp(1j * x) * self.store_arr[0, 0, :]), label="|g>")
        ax11.plot(1e9 * self.t_arr, np.real(np.exp(1j * x) * self.store_arr[1, 0, :]), label="|e>")
        ax11.plot(1e9 * t_arr_ref, np.real(ref_g), label="ref |g>")
        ax11.plot(1e9 * t_arr_ref, np.real(ref_e), label="ref |e>")
        ax12.plot(1e9 * self.t_arr, np.imag(np.exp(1j * x) * self.store_arr[0, 0, :]), label="|g>")
        ax12.plot(1e9 * self.t_arr, np.imag(np.exp(1j * x) * self.store_arr[1, 0, :]), label="|e>")
        ax12.plot(1e9 * t_arr_ref, np.imag(ref_g), label="ref |g>")
        ax12.plot(1e9 * t_arr_ref, np.imag(ref_e), label="ref |e>")
        ax11.set_ylabel("Real")
        ax12.set_ylabel("Imaginary")
        ax12.set_xlabel("Time [ns]")
        ax11.grid()
        ax12.grid()
        ax12.legend()
        fig1.show()
        ret_fig.append(fig1)

        threshold = _threshold(self.ref_g, self.ref_e)
        match_diff = (
            self.match_e_arr - self.match_g_arr - threshold
        )  # does |e> match better than |g>?

        fig2, ax2 = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True)
        _analyze_hist(ax2[0], match_diff[0::2], 0.0, fix_sum=fix_sum, logscale=logscale)
        _analyze_hist(ax2[1], match_diff[1::2], 0.0, fix_sum=fix_sum, logscale=logscale)
        ax2[0].set_title("Prepared |g>")
        ax2[1].set_title("Prepared |e>")
        fig2.show()
        ret_fig.append(fig2)

        match_iq = self.match_i_arr + 1j * self.match_q_arr
        match_iq *= np.exp(1j * x)

        fig3, ax3 = plt.subplots(1, 2, constrained_layout=True)
        ax3[0].hist2d(match_iq[0::2].real, match_iq[0::2].imag, bins=200)
        ax3[1].hist2d(match_iq[1::2].real, match_iq[1::2].imag, bins=200)
        fig3.show()
        ret_fig.append(fig3)

        fig4, ax4 = plt.subplots(1, 2, constrained_layout=True)
        _analyze_hist(ax4[0], match_iq.real, fix_sum=fix_sum, logscale=logscale)
        _analyze_hist(ax4[1], match_diff, 0.0, fix_sum=fix_sum, logscale=logscale)
        fig4.show()
        ret_fig.append(fig4)

        return ret_fig


def _analyze_hist(
    ax,
    data: npt.NDArray[np.float64],
    threshold: Optional[float] = None,
    fix_sum: bool = True,
    logscale: bool = False,
):
    from scipy.optimize import curve_fit

    if threshold is None:
        threshold = float(np.mean(data))

    idx_low = data < threshold
    idx_high = np.logical_not(idx_low)
    mean_low = data[idx_low].mean()
    mean_high = data[idx_high].mean()
    std_low = data[idx_low].std()
    std_high = data[idx_high].std()
    weight_low = np.sum(idx_low) / len(idx_low)
    weight_high = 1.0 - weight_low
    std = max(std_low, std_high)
    x_min = mean_low - 5 * std
    x_max = mean_high + 5 * std
    nr_bins = int(round(np.sqrt(data.shape[0])))
    H, xedges = np.histogram(data, bins=nr_bins, range=(x_min, x_max), density=True)
    xdata = 0.5 * (xedges[1:] + xedges[:-1])
    dx = xdata[1] - xdata[0]
    ntot = data.shape[0]
    min_dens = 1.0 / ntot / dx

    init = np.array([mean_low, std_low, weight_low, mean_high, std_high, weight_high])
    if fix_sum:
        # skip second weight
        popt, _ = curve_fit(_double_gaussian_fixed, xdata, H, p0=init[:-1])

        # add back second weight for ease of use
        popt = np.r_[popt, 1.0 - popt[2]]
        # popt_2 = np.r_[popt_2, 1.0 - popt_2[2]]
    else:
        popt, _ = curve_fit(_double_gaussian, xdata, H, p0=init)

    contrast = np.sqrt((popt[0] - popt[3]) ** 2 / (popt[1] ** 2 + popt[4] ** 2))
    print(f"{contrast = }")

    _hist_plot(ax, H, xedges, lw=1, c="0.75")
    ax.plot(
        xdata,
        _single_gaussian(xdata, *popt[:3]),
        ls="-",
        c="tab:blue",
        label=f"{popt[2]:.1%}",
    )
    ax.plot(
        xdata,
        _single_gaussian(xdata, *popt[3:]),
        ls="-",
        c="tab:orange",
        label=f"{popt[5]:.1%}",
    )

    if logscale:
        ax.set_yscale("log")
        ymin = np.log10(min_dens)
        ymax = np.log10(H.max())
        yrng = ymax - ymin
        ax.set_ylim(10 ** (ymin - 0.05 * yrng), 10 ** (ymax + 0.05 * yrng))

    legend_loc = "upper left" if popt[2] < popt[5] else "upper right"
    ax.legend(loc=legend_loc)

    ax.grid()


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


def _single_gaussian(x, m, s, w):
    return w * np.exp(-((x - m) ** 2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)


def _double_gaussian(x, m0, s0, w0, m1, s1, w1):
    return _single_gaussian(x, m0, s0, w0) + _single_gaussian(x, m1, s1, w1)


def _double_gaussian_fixed(x, m0, s0, w0, m1, s1):
    w1 = 1.0 - w0
    return _double_gaussian(x, m0, s0, w0, m1, s1, w1)


def _hist_plot(ax, spec, bin_ar, **kwargs):
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


def _rotate_opt(trace_g, trace_e):
    data = trace_e - trace_g  # complex distance

    # calculate the mean of data.imag**2 in steps of 1 deg
    N = 360
    _mean = np.zeros(N)
    for ii in range(N):
        _data = data * np.exp(1j * 2 * np.pi / N * ii)
        _mean[ii] = np.mean(_data.imag**2)

    # the mean goes like cos(x)**2
    # FFT and find the phase at frequency "2"
    fft = np.fft.rfft(_mean) / N
    # first solution
    x_fft1 = -np.angle(fft[2])  # compensate for measured phase
    x_fft1 -= np.pi  # we want to be at the zero of cos(2x)
    x_fft1 /= 2  # move from frequency "2" to "1"
    # there's a second solution np.pi away (a minus sign)
    x_fft2 = x_fft1 + np.pi

    # convert to +/- interval
    x_fft1 = _to_pm_pi(x_fft1)
    x_fft2 = _to_pm_pi(x_fft2)
    # choose the closest to zero
    if np.abs(x_fft1) < np.abs(x_fft2):
        x_fft = x_fft1
    else:
        x_fft = x_fft2

    # rotate the data and return a copy
    trace_g = trace_g * np.exp(1j * x_fft)
    trace_e = trace_e * np.exp(1j * x_fft)

    return trace_g, trace_e, x_fft


def _to_pm_pi(phase: float) -> float:
    """Converts a phase in radians into the [-π, +π) interval.

    Args:
        phase
    """
    return (phase + np.pi) % (2 * np.pi) - np.pi
