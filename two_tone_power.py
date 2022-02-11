# -*- coding: utf-8 -*-
"""
Two-tone spectroscopy in Lockin mode: 2D sweep of pump power and frequency, with fixed probe.
"""
import time

import h5py
import numpy as np
from numpy.typing import ArrayLike

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import lockin
from presto.utils import format_sec, ProgressBar

from _base import Base

DAC_CURRENT = 32_000  # uA
CONVERTER_CONFIGURATION = {
    "adc_mode": AdcMode.Mixed,
    "adc_fsample": AdcFSample.G4,
    "dac_mode": [DacMode.Mixed42, DacMode.Mixed02, DacMode.Mixed02, DacMode.Mixed02],
    "dac_fsample": [DacFSample.G10, DacFSample.G6, DacFSample.G6, DacFSample.G6],
}

# WHICH_QUBIT = 1  # 1 or 2

# # Presto's IP address or hostname
# ADDRESS = "130.237.35.90"
# PORT = 42874
# EXT_REF_CLK = False  # set to True to lock to an external reference clock
# USE_JPA = True

# if WHICH_QUBIT == 1:
#     center_freq = 3.437 * 1e9  # Hz, center frequency for qubit sweep, qubit 1
#     cavity_freq = 6.166_600 * 1e9  # Hz, frequency for cavity, resonator 1
#     qubit_port = 3  # qubit 1
# elif WHICH_QUBIT == 2:
#     center_freq = 3.975 * 1e9  # Hz, center frequency for qubit sweep, qubit 2
#     cavity_freq = 6.028_448 * 1e9  # Hz, frequency for cavity, resonator 2
#     qubit_port = 4  # qubit 2
# else:
#     raise ValueError

# span = 350 * 1e6  # Hz, span for qubit frequency sweep
# df = 1e6  # Hz, measurement bandwidth for each point in sweep

# cavity_amp = 10**(-20.0 / 20)  # FS

# nr_amps = 61
# self.control_amp_arr = np.logspace(-3, 0, nr_amps)

# cavity_port = 1
# input_port = 1
# dither = True
# extra = 500
# Navg = 1_000

# if USE_JPA:
#     jpa_bias_port = 1
#     if WHICH_QUBIT == 1:
#         jpa_bias = +0.437  # V
#         jpa_pump_pwr = 11
#         jpa_pump_freq = 2 * 6.169 * 1e9  # 2.5 MHz away from resonator 1
#     elif WHICH_QUBIT == 2:
#         jpa_bias = +0.449  # V
#         jpa_pump_pwr = 9
#         jpa_pump_freq = 2 * 6.031 * 1e9  # 2.5 MHz away from resonator 2
#     else:
#         raise ValueError

#     import sys
#     if '/home/riccardo/IntermodulatorSuite' not in sys.path:
#         sys.path.append('/home/riccardo/IntermodulatorSuite')
#     from mlaapi import mla_api, mla_globals

#     settings = mla_globals.read_config()
#     mla = mla_api.MLA(settings)
#     mla.connect()
#     mla.lockin.set_dc_offset(jpa_bias_port, jpa_bias)
#     time.sleep(1.0)


class TwoTonePower(Base):
    def __init__(
        self,
        readout_freq: float,
        control_freq_center: float,
        control_freq_span: float,
        df: float,
        readout_amp: float,
        control_amp_arr: ArrayLike,
        readout_port: int,
        control_port: int,
        input_port: int,
        num_averages: int,
        dither=True,
        num_skip=1,
    ) -> None:
        self.readout_freq = readout_freq
        self.control_freq_center = control_freq_center
        self.control_freq_span = control_freq_span
        self.df = df
        self.readout_amp = readout_amp
        self.control_amp_arr = np.atleast_1d(control_amp_arr).astype(np.float64)
        self.readout_port = readout_port
        self.control_port = control_port
        self.input_port = input_port
        self.num_averages = num_averages
        self.dither = dither
        self.num_skip = num_skip

        self.control_freq_arr = None  # replaced by run
        self.resp_arr = None  # replaced by run

    def run(
        self,
        presto_address: str,
        presto_port: int = None,
        ext_ref_clk: bool = False,
    ):
        with lockin.Lockin(
                address=presto_address,
                port=presto_port,
                ext_ref_clk=ext_ref_clk,
                **CONVERTER_CONFIGURATION,
        ) as lck:
            assert lck.hardware is not None

            lck.hardware.set_adc_attenuation(self.input_port, 0.0)
            lck.hardware.set_dac_current(self.readout_port, DAC_CURRENT)
            lck.hardware.set_dac_current(self.control_port, DAC_CURRENT)
            lck.hardware.set_inv_sinc(self.readout_port, 0)
            lck.hardware.set_inv_sinc(self.control_port, 0)
            # if USE_JPA:
            #     lck.hardware.set_lmx(jpa_pump_freq, jpa_pump_pwr)

            nr_amps = len(self.control_amp_arr)

            # tune frequencies
            _, self.df = lck.tune(0.0, self.df)
            f_start = self.control_freq_center - self.control_freq_span / 2
            f_stop = self.control_freq_center + self.control_freq_span / 2
            n_start = int(round(f_start / self.df))
            n_stop = int(round(f_stop / self.df))
            n_arr = np.arange(n_start, n_stop + 1)
            nr_freq = len(n_arr)
            self.control_freq_arr = self.df * n_arr
            self.resp_arr = np.zeros((nr_amps, nr_freq), np.complex128)

            lck.hardware.configure_mixer(
                freq=self.readout_freq,
                in_ports=self.input_port,
                out_ports=self.readout_port,
            )
            lck.hardware.configure_mixer(
                freq=self.control_freq_arr[0],
                out_ports=self.control_port,
            )
            lck.set_df(self.df)
            ogr = lck.add_output_group(self.readout_port, 1)
            ogr.set_frequencies(0.0)
            ogr.set_amplitudes(self.readout_amp)
            ogr.set_phases(0.0, 0.0)
            ogc = lck.add_output_group(self.control_port, 1)
            ogc.set_frequencies(0.0)
            ogc.set_amplitudes(self.control_amp_arr[0])
            ogc.set_phases(0.0, 0.0)

            lck.set_dither(self.dither, [self.readout_port, self.control_port])
            ig = lck.add_input_group(self.input_port, 1)
            ig.set_frequencies(0.0)

            lck.apply_settings()

            pb = ProgressBar(nr_amps * nr_freq)
            pb.start()
            for jj, control_amp in enumerate(self.control_amp_arr):
                ogc.set_amplitudes(control_amp)
                lck.apply_settings()

                for ii, control_freq in enumerate(self.control_freq_arr):
                    lck.hardware.configure_mixer(
                        freq=control_freq,
                        out_ports=self.control_port,
                    )
                    lck.hardware.sleep(1e-3, False)

                    _d = lck.get_pixels(self.num_skip + self.num_averages)
                    data_i = _d[self.input_port][1][:, 0]
                    data_q = _d[self.input_port][2][:, 0]
                    data = data_i.real + 1j * data_q.real  # using zero IF

                    self.resp_arr[jj, ii] = np.mean(data[-self.num_averages:])

                    pb.increment()

            pb.done()

            # Mute outputs at the end of the sweep
            ogr.set_amplitudes(0.0)
            ogc.set_amplitudes(0.0)
            lck.apply_settings()
            # if USE_JPA:
            #     lck.hardware.set_lmx(0.0, 0)
        # if USE_JPA:
        #     mla.lockin.set_dc_offset(jpa_bias_port, 0.0)
        #     mla.disconnect()

        return self.save()

    @classmethod
    def load(cls, load_filename):
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = h5f.attrs["readout_freq"]
            control_freq_center = h5f.attrs["control_freq_center"]
            control_freq_span = h5f.attrs["control_freq_span"]
            df = h5f.attrs["df"]
            readout_amp = h5f.attrs["readout_amp"]
            readout_port = h5f.attrs["readout_port"]
            control_port = h5f.attrs["control_port"]
            input_port = h5f.attrs["input_port"]
            num_averages = h5f.attrs["num_averages"]
            dither = h5f.attrs["dither"]
            num_skip = h5f.attrs["num_skip"]

            control_amp_arr = h5f["control_amp_arr"][()]
            control_freq_arr = h5f["control_freq_arr"][()]
            resp_arr = h5f["resp_arr"][()]

        self = cls(
            readout_freq=readout_freq,
            control_freq_center=control_freq_center,
            control_freq_span=control_freq_span,
            df=df,
            readout_amp=readout_amp,
            control_amp_arr=control_amp_arr,
            readout_port=readout_port,
            control_port=control_port,
            input_port=input_port,
            num_averages=num_averages,
            dither=dither,
            num_skip=num_skip,
        )
        self.control_freq_arr = control_freq_arr
        self.resp_arr = resp_arr

        return self

    def analyze(self, logscale=False, linecut=False, blit=True):
        if self.control_freq_arr is None:
            raise RuntimeError
        if self.resp_arr is None:
            raise RuntimeError

        import matplotlib.pyplot as plt
        nr_amps = len(self.control_amp_arr)

        self._AMP_IDX = nr_amps // 2

        if logscale:
            data = 20. * np.log10(np.abs(self.resp_arr))
            unit = "dB"
        else:
            data = np.abs(self.resp_arr)
            data_max = data.max()
            unit = ""
            if data_max < 1e-6:
                unit = "n"
                data *= 1e9
            elif data_max < 1e-3:
                unit = "μ"
                data *= 1e6
            elif data_max < 1e0:
                unit = "m"
                data *= 1e3
        amp_dBFS = 20 * np.log10(self.control_amp_arr / 1.0)

        # choose limits for colorbar
        cutoff = 1.  # %
        lowlim = np.percentile(data, cutoff)
        highlim = np.percentile(data, 100. - cutoff)

        # extent
        x_min = 1e-9 * self.control_freq_arr[0]
        x_max = 1e-9 * self.control_freq_arr[-1]
        dx = 1e-9 * (self.control_freq_arr[1] - self.control_freq_arr[0])
        y_min = amp_dBFS[0]
        y_max = amp_dBFS[-1]
        dy = amp_dBFS[1] - amp_dBFS[0]

        if linecut:
            fig1 = plt.figure(tight_layout=True, figsize=(6.4, 9.6))
            ax1 = fig1.add_subplot(2, 1, 1)
        else:
            fig1 = plt.figure(tight_layout=True, figsize=(6.4, 4.8))
            ax1 = fig1.add_subplot(1, 1, 1)
        im = ax1.imshow(
            data,
            origin='lower',
            aspect='auto',
            interpolation='none',
            extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
            vmin=lowlim,
            vmax=highlim,
        )
        if linecut:
            line_sel = ax1.axhline(amp_dBFS[self._AMP_IDX], ls="--", c="k", lw=3, animated=blit)
        ax1.set_title(f"Probe frequency: {self.readout_freq/1e9:.2f} GHz")
        ax1.set_xlabel("Pump frequency [GHz]")
        ax1.set_ylabel("Pump amplitude [dBFS]")
        cb = fig1.colorbar(im)
        cb.set_label(f"Response amplitude [{unit:s}FS]")

        if linecut:
            ax2 = fig1.add_subplot(4, 1, 3)
            ax3 = fig1.add_subplot(4, 1, 4, sharex=ax2)

            line_a, = ax2.plot(1e-9 * self.control_freq_arr, data[self._AMP_IDX], animated=blit)
            line_fit_a, = ax2.plot(1e-9 * self.control_freq_arr,
                                   np.full_like(self.control_freq_arr, np.nan),
                                   ls="--",
                                   animated=blit)
            line_p, = ax3.plot(1e-9 * self.control_freq_arr, np.angle(self.resp_arr[self._AMP_IDX]), animated=blit)
            line_fit_p, = ax3.plot(1e-9 * self.control_freq_arr,
                                   np.full_like(self.control_freq_arr, np.nan),
                                   ls="--",
                                   animated=blit)

            f_min = 1e-9 * self.control_freq_arr.min()
            f_max = 1e-9 * self.control_freq_arr.max()
            f_rng = f_max - f_min
            a_min = data.min()
            a_max = data.max()
            a_rng = a_max - a_min
            p_min = -np.pi
            p_max = np.pi
            p_rng = p_max - p_min
            ax2.set_xlim(f_min - 0.05 * f_rng, f_max + 0.05 * f_rng)
            ax2.set_ylim(a_min - 0.05 * a_rng, a_max + 0.05 * a_rng)
            ax3.set_xlim(f_min - 0.05 * f_rng, f_max + 0.05 * f_rng)
            ax3.set_ylim(p_min - 0.05 * p_rng, p_max + 0.05 * p_rng)

            ax3.set_xlabel("Frequency [GHz]")
            if logscale:
                ax2.set_ylabel("Response amplitude [dB]")
            else:
                ax2.set_ylabel(f"Response amplitude [{unit:s}FS]")
            ax3.set_ylabel("Response phase [rad]")

            def onbuttonpress(event):
                if event.inaxes == ax1:
                    self._AMP_IDX = np.argmin(np.abs(amp_dBFS - event.ydata))
                    update()

            def onkeypress(event):
                if event.inaxes == ax1:
                    if event.key == "up":
                        self._AMP_IDX += 1
                        if self._AMP_IDX >= len(amp_dBFS):
                            self._AMP_IDX = len(amp_dBFS) - 1
                        update()
                    elif event.key == "down":
                        self._AMP_IDX -= 1
                        if self._AMP_IDX < 0:
                            self._AMP_IDX = 0
                        update()

            def update():
                line_sel.set_ydata([amp_dBFS[self._AMP_IDX], amp_dBFS[self._AMP_IDX]])
                # ax1.set_title(f"amp = {amp_arr[self._AMP_IDX]:.2e}")
                print(
                    f"drive amp {self._AMP_IDX:d}: {self.control_amp_arr[self._AMP_IDX]:.2e} FS = {amp_dBFS[self._AMP_IDX]:.1f} dBFS"
                )
                line_a.set_ydata(data[self._AMP_IDX])
                line_p.set_ydata(np.angle(self.resp_arr[self._AMP_IDX]))
                line_fit_a.set_ydata(np.full_like(self.control_freq_arr, np.nan))
                line_fit_p.set_ydata(np.full_like(self.control_freq_arr, np.nan))
                # ax2.set_title("")
                if blit:
                    fig1.canvas.restore_region(self._bg)
                    ax1.draw_artist(line_sel)
                    ax2.draw_artist(line_a)
                    ax3.draw_artist(line_p)
                    fig1.canvas.blit(fig1.bbox)
                    # fig1.canvas.flush_events()
                else:
                    fig1.canvas.draw()

            fig1.canvas.mpl_connect('button_press_event', onbuttonpress)
            fig1.canvas.mpl_connect('key_press_event', onkeypress)

        fig1.show()
        if linecut and blit:
            fig1.canvas.draw()
            # fig1.canvas.flush_events()
            self._bg = fig1.canvas.copy_from_bbox(fig1.bbox)
            ax1.draw_artist(line_sel)
            ax2.draw_artist(line_a)
            ax3.draw_artist(line_p)
            fig1.canvas.blit(fig1.bbox)

        return fig1


# # *************************
# # *** Save data to HDF5 ***
# # *************************
# script_path = os.path.realpath(__file__)  # full path of current script
# current_dir, script_basename = os.path.split(script_path)
# script_filename = os.path.splitext(script_basename)[0]  # name of current script
# timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # current date and time
# save_basename = f"{script_filename:s}_{timestamp:s}.h5"  # name of save file
# save_path = os.path.join(current_dir, "data", save_basename)  # full path of save file
# source_code = get_sourcecode(script_path)  # save also the sourcecode of the script for future reference
# with h5py.File(save_path, "w") as h5f:
#     dt = h5py.string_dtype(encoding='utf-8')
#     ds = h5f.create_dataset("source_code", (len(source_code), ), dt)
#     for ii, line in enumerate(source_code):
#         ds[ii] = line
#     h5f.attrs["df"] = df
#     h5f.attrs["dither"] = dither
#     h5f.attrs["input_port"] = input_port
#     h5f.attrs["cavity_port"] = cavity_port
#     h5f.attrs["qubit_port"] = qubit_port
#     h5f.attrs["cavity_amp"] = cavity_amp
#     h5f.attrs["cavity_freq"] = cavity_freq
#     h5f.create_dataset("self.control_freq_arr", data=self.control_freq_arr)
#     h5f.create_dataset("self.control_amp_arr", data=self.control_amp_arr)
#     h5f.create_dataset("resp_arr", data=resp_arr)
# print(f"Data saved to: {save_path}")

# ********************
# *** Plot results ***
# ********************
# fig1 = load_two_tone_power.load(save_path)
