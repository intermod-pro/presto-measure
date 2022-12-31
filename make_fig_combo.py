import matplotlib.pyplot as plt
import numpy as np
from resonator_tools import circuit
from scipy.optimize import curve_fit

from presto.utils import rotate_opt, untwist_downconversion

import excited_sweep
import rabi_amp
import ramsey_chevron
import ramsey_echo
import readout_ref
import t1

plt.rcParams["figure.autolayout"] = False  # tight_layout
plt.rcParams["figure.constrained_layout.use"] = True  # constrained_layout

plt.rcParams["figure.figsize"] = [6.69, 2.77]  # double column, 1/3 page
# plt.rcParams['figure.figsize'] = [6.69, 5.02]  # double column, 4:3
# plt.rcParams['figure.figsize'] = [3.37, 2.53]  # single column, 4:3

plt.rcParams["legend.borderaxespad"] = 0.5 / 2
plt.rcParams["legend.borderpad"] = 0.4 / 2
plt.rcParams["legend.fontsize"] = 8.0
plt.rcParams["legend.handleheight"] = 0.7 / 2
plt.rcParams["legend.handlelength"] = 2.0 / 1
plt.rcParams["legend.handletextpad"] = 0.8 / 2
plt.rcParams["legend.labelspacing"] = 0.5 / 4

plt.rcParams["axes.labelsize"] = 10.0
plt.rcParams["axes.labelpad"] = 4.0 / 4
plt.rcParams["xtick.labelsize"] = 8.0
plt.rcParams["ytick.labelsize"] = 8.0
plt.rcParams["ytick.major.pad"] = 3.5 / 4

plt.rcParams["figure.dpi"] = 108.8  # my screen
plt.rcParams["savefig.dpi"] = 300.0  # RSI
plt.rcParams["savefig.format"] = "eps"


def main():
    # m_ref = readout_ref.ReadoutRef.load('data/readout_ref_20220405_141238.h5')
    m_ref = readout_ref.ReadoutRef.load("data/readout_ref_20220413_091546.h5")
    global ref_g, ref_e
    ref_g = m_ref.store_arr[0, 0, :]
    ref_e = m_ref.store_arr[1, 0, :]

    # m_rabi = rabi_amp.RabiAmp.load('data/rabi_amp_20220404_102615.h5')
    m_rabi = rabi_amp.RabiAmp.load("data/rabi_amp_20220413_082434.h5")
    # m_chevron = ramsey_chevron.RamseyChevron.load('data/ramsey_chevron_20220407_merged.h5')
    m_chevron = ramsey_chevron.RamseyChevron.load("data/ramsey_chevron_20220413_013628.h5")
    m_t1 = t1.T1.load("data/t1_20220404_130724.h5")
    m_t2 = ramsey_echo.RamseyEcho.load("data/ramsey_echo_20220405_113517.h5")
    m_sweep = excited_sweep.ExcitedSweep.load("data/excited_sweep_20220402_201551.h5")

    fig = plt.figure()
    gs0 = fig.add_gridspec(1, 3)

    gs1 = gs0[0].subgridspec(2, 1)
    ax_rabi = fig.add_subplot(gs1[0])
    plot_rabi(m_rabi, ax_rabi)

    ax_t1 = fig.add_subplot(gs1[1])
    plot_t1(m_t1, ax_t1)
    plot_t2(m_t2, ax_t1)

    gs2 = gs0[1].subgridspec(1, 1)
    ax_chevron = fig.add_subplot(gs2[0])
    plot_chevron(m_chevron, ax_chevron, fig)

    gs3 = gs0[2].subgridspec(3, 1)
    ax_sweep = gs3.subplots(sharex=True)
    plot_sweep(m_sweep, ax_sweep)

    ax_rabi.text(-0.18, 1.0, "(a)", fontsize=10, transform=ax_rabi.transAxes)
    ax_t1.text(-0.18, 1.0, "(b)", fontsize=10, transform=ax_t1.transAxes)
    ax_chevron.text(-0.18, 1.0, "(c)", fontsize=10, transform=ax_chevron.transAxes)

    ax_sweep[0].text(0.98, 0.15, "(d)", fontsize=10, ha="right", transform=ax_sweep[0].transAxes)
    ax_sweep[1].text(0.98, 0.15, "(e)", fontsize=10, ha="right", transform=ax_sweep[1].transAxes)
    ax_sweep[2].text(
        0.98, 0.95, "(f)", fontsize=10, ha="right", va="top", transform=ax_sweep[2].transAxes
    )

    # fig.savefig("measurement_combo")
    fig.show()


def project(resp_arr):
    conj_g = ref_g.conj()
    conj_e = ref_e.conj()
    norm_g = np.sum(ref_g * conj_g).real
    norm_e = np.sum(ref_e * conj_e).real
    overlap = np.sum(ref_g * conj_e).real
    proj_g = np.zeros(resp_arr.shape[0])
    proj_e = np.zeros(resp_arr.shape[0])
    for i in range(resp_arr.shape[0]):
        proj_g[i] = np.sum(conj_g * resp_arr[i, :]).real
        proj_e[i] = np.sum(conj_e * resp_arr[i, :]).real
    res = proj_e - proj_g
    res_g = overlap - norm_g
    res_e = norm_e - overlap
    res_min = res_g
    res_rng = res_e - res_g
    data = (res - res_min) / res_rng
    return data


def plot_rabi(self: rabi_amp.RabiAmp, ax) -> None:
    print_header("rabi amp")
    assert self.t_arr is not None
    assert self.store_arr is not None

    # idx = np.arange(rabi_amp.IDX_LOW, rabi_amp.IDX_HIGH)
    # resp_arr = np.mean(self.store_arr[:, 0, idx], axis=-1)
    # data = rotate_opt(resp_arr).real

    resp_arr = self.store_arr[:, 0, :]
    data = project(resp_arr)

    # Fit data
    popt_x, perr_x = rabi_amp._fit_period(self.control_amp_arr, data)
    period = popt_x[3]
    period_err = perr_x[3]
    pi_amp = period / 2
    pi_2_amp = period / 4
    if self.num_pulses > 1:
        print(f"{self.num_pulses} pulses")
    print(
        "Tau pulse amplitude: {} +- {} FS".format(
            period * self.num_pulses, period_err * self.num_pulses
        )
    )
    print(
        "Pi pulse amplitude: {} +- {} FS".format(
            pi_amp * self.num_pulses, period_err / 2 * self.num_pulses
        )
    )
    print(
        "Pi/2 pulse amplitude: {} +- {} FS".format(
            pi_2_amp * self.num_pulses, period_err / 4 * self.num_pulses
        )
    )

    # x_min = popt_x[0] - popt_x[1]
    # x_max = popt_x[0] + popt_x[1]
    # ax.plot(100 * self.control_amp_arr, rescale(data, x_min, x_max), '.', c="0.75")
    # ax.plot(100 * self.control_amp_arr, rescale(rabi_amp._func(self.control_amp_arr, *popt_x), x_min, x_max), '-')

    ax.plot(100 * self.control_amp_arr, data, ".", c="0.75")
    ax.plot(100 * self.control_amp_arr, rabi_amp._func(self.control_amp_arr, *popt_x), "-")

    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.set_yticks([0.5], minor=True)
    ax.set_ylabel("$P_\\mathrm{e}$")
    ax.set_xticks([0, 30, 60])
    ax.set_xlabel("Output amplitude [%]")

    ax.tick_params(axis="x", which="both", direction="in")


def plot_t1(self: t1.T1, ax) -> None:
    print_header("t1")
    assert self.t_arr is not None
    assert self.store_arr is not None

    # idx = np.arange(t1.IDX_LOW, t1.IDX_HIGH)
    # resp_arr = np.mean(self.store_arr[:, 0, idx], axis=-1)
    # data = rotate_opt(resp_arr).real

    resp_arr = self.store_arr[:, 0, :]
    data = project(resp_arr)

    popt, perr = t1._fit_simple(self.delay_arr, data)
    T1, x0, x1 = popt
    T1_err = perr[0]
    print("T1 time I: {} +- {} us".format(1e6 * T1, 1e6 * T1_err))
    label = f"$T_1 = {1e6*T1:.1f} \\mathrm{{\\mu s}}$"

    # xg = x1
    # xe = x0
    # ax.plot(1e6 * self.delay_arr, rescale(data, xg, xe), '.', c="0.75")
    # ax.plot(1e6 * self.delay_arr, rescale(t1._decay(self.delay_arr, *popt), xg, xe), '-', label=label)

    ax.plot(1e6 * self.delay_arr, data, ".", c="0.75")
    ax.plot(1e6 * self.delay_arr, t1._decay(self.delay_arr, *popt), "-", label=label)

    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.set_yticks([0.5], minor=True)
    ax.set_ylabel("$P_\\mathrm{e}$")
    ax.set_xlabel("Time [μs]")

    ax.tick_params(axis="x", which="both", direction="in")


def plot_t2(self: ramsey_echo.RamseyEcho, ax) -> None:
    print_header("t2")
    assert self.t_arr is not None
    assert self.store_arr is not None

    # idx = np.arange(ramsey_echo.IDX_LOW, ramsey_echo.IDX_HIGH)
    # resp_arr = np.mean(self.store_arr[:, 0, idx], axis=-1)
    # data = rotate_opt(resp_arr).real

    resp_arr = self.store_arr[:, 0, :]
    data = project(resp_arr)

    popt, perr = ramsey_echo._fit_simple(self.delay_arr, np.real(data))

    T2, x0, x1 = popt
    T2_err = perr[0]
    print(f"T2_echo time: {1e6*T2} ± {1e6*T2_err} μs")
    label = f"$T_2 = {1e6*T2:.1f} \\mathrm{{\\mu s}}$"

    # xg = x0
    # xe = x0 + 2 * (x1 - x0)
    # ax.plot(1e6 * self.delay_arr, rescale(data, xg, xe), '.', c="0.75")
    # ax.plot(1e6 * self.delay_arr, rescale(ramsey_echo._decay(self.delay_arr, *popt), xg, xe), '-', label=label)

    ax.plot(1e6 * self.delay_arr, data, ".", c="0.75")
    ax.plot(1e6 * self.delay_arr, ramsey_echo._decay(self.delay_arr, *popt), "-", label=label)

    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.set_yticks([0.5], minor=True)
    ax.set_ylabel("$P_\\mathrm{e}$")
    ax.set_xlabel("Time [μs]")

    ax.tick_params(axis="x", which="both", direction="in")


def plot_chevron(self: ramsey_chevron.RamseyChevron, ax, fig) -> None:
    print_header("chevron")
    assert self.t_arr is not None
    assert self.store_arr is not None
    assert self.control_freq_arr is not None
    assert len(self.control_freq_arr) == self.control_freq_nr

    # idx = np.arange(ramsey_chevron.IDX_LOW, ramsey_chevron.IDX_HIGH)
    # resp_arr = np.mean(self.store_arr[:, 0, idx], axis=-1)
    # resp_arr.shape = (self.control_freq_nr, len(self.delay_arr))
    # data = rotate_opt(resp_arr).real

    resp_arr = self.store_arr[:, 0, :]
    data = project(resp_arr)
    data.shape = (self.control_freq_nr, len(self.delay_arr))

    # choose limits for colorbar
    # cutoff = 0.0  # %
    # lowlim = np.percentile(data, cutoff)
    # highlim = np.percentile(data, 100. - cutoff)
    # lowlim, highlim = 0, 1
    lowlim, highlim = -0.01, 1.01

    # extent
    x_min = 1e6 * self.delay_arr[0]
    x_max = 1e6 * self.delay_arr[-1]
    dx = 1e6 * (self.delay_arr[1] - self.delay_arr[0])
    y_min = 1e-6 * self.control_freq_arr[0]
    y_max = 1e-6 * self.control_freq_arr[-1]
    dy = 1e-6 * (self.control_freq_arr[1] - self.control_freq_arr[0])

    y_min -= 1e-6 * self.control_freq_center
    y_max -= 1e-6 * self.control_freq_center

    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        interpolation="none",
        extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
        vmin=lowlim,
        vmax=highlim,
    )
    ax.set_xlabel("Time [μs]")
    ax.set_ylabel("$\\omega - \\omega_{{01}}$ [MHz]")
    cb = fig.colorbar(im, ax=ax, location="bottom", aspect=8, pad=0.0)
    cb.set_label("$P_\\mathrm{e}$", labelpad=-10)
    cb.set_ticks([0.0, 1.0])


def plot_sweep(self: excited_sweep.ExcitedSweep, ax: list) -> None:
    print_header("excited sweep")
    assert self.t_arr is not None
    assert self.store_arr is not None
    assert self.readout_freq_arr is not None
    assert self.readout_if_arr is not None
    assert self.readout_nco is not None
    assert len(self.readout_freq_arr) == self.readout_freq_nr
    assert len(self.readout_if_arr) == self.readout_freq_nr

    idx = np.arange(excited_sweep.IDX_LOW, excited_sweep.IDX_HIGH)
    nr_samples = excited_sweep.IDX_HIGH - excited_sweep.IDX_LOW

    data = self.store_arr[:, 0, idx]
    data.shape = (self.readout_freq_nr, 2, nr_samples)
    resp_I_arr = np.zeros((2, self.readout_freq_nr), np.complex128)
    resp_Q_arr = np.zeros((2, self.readout_freq_nr), np.complex128)
    dt = self.t_arr[1] - self.t_arr[0]
    t = dt * np.arange(nr_samples)
    for ii, readout_if in enumerate(self.readout_if_arr):
        cos = np.cos(2 * np.pi * readout_if * t)
        sin = np.sin(2 * np.pi * readout_if * t)
        for jj in range(2):
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
    pfit_e = np.polyfit(self.readout_freq_arr[idx], resp_phase[1, idx], 1)
    pfit = 0.5 * (pfit_g + pfit_e)
    background = np.polyval(pfit, self.readout_freq_arr)
    resp_phase[0, :] -= background
    resp_phase[1, :] -= background
    separation = np.abs(resp_H_arr[1, :] - resp_H_arr[0, :])

    p0 = [
        self.readout_freq_arr[np.argmax(separation)],
        1 / self.readout_duration,
        np.max(separation),
        0.0,
    ]
    popt, pcov = curve_fit(excited_sweep._gaussian, self.readout_freq_arr, separation, p0)

    port_g = circuit.notch_port(self.readout_freq_arr, resp_H_arr[0, :] * np.exp(-1j * background))
    port_e = circuit.notch_port(self.readout_freq_arr, resp_H_arr[1, :] * np.exp(-1j * background))
    port_g.autofit()
    port_e.autofit()

    f_g = port_g.fitresults["fr"]
    f_e = port_e.fitresults["fr"]
    chi_hz = (f_e - f_g) / 2
    print(f"ω_g / 2π = {f_g * 1e-9:.6f} GHz")
    print(f"ω_e / 2π = {f_e * 1e-9:.6f} GHz")
    print(f"χ / 2π = {chi_hz * 1e-3:.2f} kHz")
    print(f"ω_opt / 2π = {popt[0] * 1e-9:.6f} GHz")

    """
    fig9, ax9 = plt.subplots(figsize=[3.37, 2.53])  # single column, 4:3
    # ax9.plot(resp_H_arr[0, :].real, resp_H_arr[0, :].imag)
    # ax9.plot(resp_H_arr[1, :].real, resp_H_arr[1, :].imag)
    # ax9.plot(port_g.z_data_sim_norm.real, port_g.z_data_sim_norm.imag)
    # ax9.plot(port_e.z_data_sim_norm.real, port_e.z_data_sim_norm.imag)
    ax9.axhline(0.0, c='0.85')
    ax9.axvline(0.0, c='0.85')
    # ax9.plot(port_g.z_data.real, port_g.z_data.imag, c="tab:blue")
    # ax9.plot(port_e.z_data.real, port_e.z_data.imag, c="tab:orange")
    ax9.scatter(port_g.z_data.real, port_g.z_data.imag, s=12, c=np.linspace(0, 1, len(port_g.z_data)), cmap="viridis")
    ax9.scatter(port_e.z_data.real, port_e.z_data.imag, s=12, c=np.linspace(0, 1, len(port_e.z_data)), cmap="magma")
    max_idx = np.argmin(np.abs(port_g.f_data - popt[0]))
    ax9.plot(port_g.z_data[max_idx].real, port_g.z_data[max_idx].imag, '.', ms=12, c="tab:blue")
    ax9.plot(port_e.z_data[max_idx].real, port_e.z_data[max_idx].imag, '.', ms=12, c="tab:orange")
    ax9.annotate(
        "",
        xy=(port_g.z_data[max_idx].real, port_g.z_data[max_idx].imag),
        xytext=(port_e.z_data[max_idx].real, port_e.z_data[max_idx].imag),
        arrowprops=dict(arrowstyle="<->", color="tab:green", lw=1.5),
    )
    ax9.set_xlabel("Normalized I quadrature [arb. units]")
    ax9.set_ylabel("Normalized Q quadrature")
    fig9.show()

    from scipy.signal import butter, lfilter

    readout_if = self.readout_if_arr[max_idx]
    def lp(data):
        b, a = butter(5, readout_if / 2, fs=1e9, btype='low', analog=False)
        return lfilter(b, a, data)
    cos = np.cos(2 * np.pi * readout_if * t)
    sin = np.sin(2 * np.pi * readout_if * t)
    data_slice = data[max_idx, 0, :]
    # TODO: low-pass filter the demodulated signal?
    I_real = lp(data_slice.real * cos)
    I_imag = -lp(data_slice.real * sin)
    resp_I = I_real + 1j * I_imag
    Q_real = lp(data_slice.imag * cos)
    Q_imag = -lp(data_slice.imag * sin)
    resp_Q = Q_real + 1j * Q_imag
    _, resp_H = untwist_downconversion(resp_I, resp_Q)

    fig10, ax10 = plt.subplots(figsize=[3.37, 2.53])  # single column, 4:3
    ax10.plot(resp_H.real)
    ax10.plot(resp_H.imag)
    fig10.show()
    """

    ax1, ax2, ax3 = ax

    amp_bg = 20 * np.log10(np.abs(port_g.z_data_sim[0]))
    sep_max = excited_sweep._gaussian(self.readout_freq_arr, *popt).max()

    ax1.plot(1e-9 * self.readout_freq_arr, resp_dB[0, :] - amp_bg, ".", c="0.75")
    ax1.plot(1e-9 * self.readout_freq_arr, resp_dB[1, :] - amp_bg, ".", c="0.75")
    ax2.plot(1e-9 * self.readout_freq_arr, resp_phase[0, :], ".", c="0.75")
    ax2.plot(1e-9 * self.readout_freq_arr, resp_phase[1, :], ".", c="0.75")
    # ax3.plot(1e-9 * self.readout_freq_arr, 1e3 * separation, '.', c="0.75")
    ax3.plot(1e-9 * self.readout_freq_arr, separation / sep_max, ".", c="0.75")

    ax1.plot(
        1e-9 * port_g.f_data,
        20 * np.log10(np.abs(port_g.z_data_sim)) - amp_bg,
        c="tab:blue",
        ls="-",
        label="|g>",
    )
    ax1.plot(
        1e-9 * port_e.f_data,
        20 * np.log10(np.abs(port_e.z_data_sim)) - amp_bg,
        c="tab:orange",
        ls="-",
        label="|e>",
    )
    ax2.plot(1e-9 * port_g.f_data, np.angle(port_g.z_data_sim), c="tab:blue", ls="-")
    ax2.plot(1e-9 * port_e.f_data, np.angle(port_e.z_data_sim), c="tab:orange", ls="-")

    ax3.plot(
        1e-9 * self.readout_freq_arr,
        # 1e3 * excited_sweep._gaussian(self.readout_freq_arr, *popt),
        excited_sweep._gaussian(self.readout_freq_arr, *popt) / sep_max,
        c="tab:green",
        ls="-",
    )

    # ax1.set_ylabel("Amplitude")
    # ax2.set_ylabel("Phase")
    # ax3.set_ylabel("Separation")
    ax1.set_ylabel("$A$ [dB]")
    ax2.set_ylabel("$\phi$ [rad]")
    ax3.set_ylabel(r"$S / S_\mathrm{max}$")
    ax[-1].set_xlabel("Readout frequency [GHz]")
    ax1.legend(loc="lower left")
    # ax3.legend(loc="upper right")

    # ax2.text(0.0, 0.0, "1 rad/div.", fontsize=10, rotation="vertical", transform=ax2.transAxes)

    ax1.set_yticks([-20, 0])
    ax1.set_yticks([-10], minor=True)
    ax3.set_yticks([0, 1])
    ax3.set_yticks([0.5], minor=True)

    ax1.tick_params(axis="x", which="both", direction="in")
    ax2.tick_params(axis="x", which="both", direction="in")
    ax3.tick_params(axis="x", which="both", direction="inout")
    for ax_ in ax:
        # ax_.axvline(1e-9 * f_g, ls='--', c='tab:red', alpha=0.5)
        # ax_.axvline(1e-9 * f_e, ls='--', c='tab:purple', alpha=0.5)
        # ax_.axvline(1e-9 * popt[0], ls='--', c='tab:brown', alpha=0.5)
        # ax_.set_yticks([])

        # for tick in ax_.get_yticklabels():
        #     tick.set_visible(False)

        # labels = [" " for item in ax_.get_yticklabels()]
        # ax_.set_yticklabels(labels)

        ax_.yaxis.tick_right()
        ax_.tick_params("y", labelrotation=90)
        # ax_.tick_params(axis='y', which="both", direction="in", labelrotation=90)
        for tick in ax_.yaxis.get_majorticklabels():
            tick.set_verticalalignment("center")


def print_header(msg: str) -> None:
    msg = f" {msg} "
    print(f"\n{msg:-^80}")


def rescale(data, min_, max_):
    rng = max_ - min_
    data = data - min_  # make copy
    data /= rng
    return data


if __name__ == "__main__":
    main()
