# -*- coding: utf-8 -*-
"""
Two-tone spectroscopy in Lockin mode: 2D sweep of pump power and frequency, with fixed probe.
"""
from typing import List

import h5py
import numpy as np
import warnings

from presto.hardware import AdcFSample, AdcMode, DacFSample, DacMode
from presto import lockin
from presto.utils import ProgressBar, rotate_opt, recommended_dac_config

from _base import Base

DAC_CURRENT = 32_000  # uA
CONVERTER_CONFIGURATION = {
    "adc_mode": AdcMode.Mixed,
    "adc_fsample": AdcFSample.G2,
}


class TwoTonePower(Base):
    def __init__(
        self,
        readout_freq: float,
        control_freq_center: float,
        control_freq_span: float,
        df: float,
        readout_amp: float,
        control_amp_arr: List[float],
        readout_port: int,
        control_port: int,
        input_port: int,
        num_averages: int,
        dither: bool = True,
        num_skip: int = 0,
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
    ) -> str:
        with lockin.Lockin(address=presto_address, ext_ref_clk=ext_ref_clk) as lck:
            control_tile = lck.hardware._port_to_tile(self.control_port, "dac")
            readout_tile = lck.hardware._port_to_tile(self.readout_port, "dac")
        dac_mode_r, dac_fsample_r = recommended_dac_config(self.readout_freq)
        dac_mode_c, dac_fsample_c = recommended_dac_config(self.control_freq_center)
        if dac_mode_c == dac_mode_r and dac_fsample_c == dac_fsample_r:
            dac_mode = dac_mode_c
            dac_fsample = dac_fsample_c
        elif control_tile != readout_tile:
            dac_mode = [dac_mode_r, dac_mode_r, dac_mode_r, dac_mode_r]
            dac_fsample = [dac_fsample_r, dac_fsample_r, dac_fsample_r, dac_fsample_r]
            dac_mode[control_tile] = dac_mode_c
            dac_fsample[control_tile] = dac_fsample_c
        else:
            warnings.warn(
                "Warning: The qubit and readout frequency might not be able to be output on the same tile. Consider outputting qubit tone on a different tile or manually choose the dac_mode and dac_fsample. See presto.utils.recommended_dac_config for help."
            )
            dac_mode = dac_mode_c
            dac_fsample = dac_fsample_c
        with lockin.Lockin(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            dac_fsample=dac_fsample,
            dac_mode=dac_mode,
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
            lck.set_df(self.df)

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

            ogr = lck.add_output_group(self.readout_port, nr_freq=1)
            ogr.set_frequencies(0.0)
            ogr.set_amplitudes(self.readout_amp)
            ogr.set_phases(0.0, 0.0)
            ogc = lck.add_output_group(self.control_port, nr_freq=1)
            ogc.set_frequencies(0.0)
            ogc.set_amplitudes(self.control_amp_arr[0])
            ogc.set_phases(0.0, 0.0)

            lck.set_dither(self.dither, [self.readout_port, self.control_port])

            ig = lck.add_input_group(self.input_port, nr_freq=1)
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

                    _d = lck.get_pixels(self.num_skip + self.num_averages, quiet=True)
                    data_i = _d[self.input_port][1][:, 0]
                    data_q = _d[self.input_port][2][:, 0]
                    data = data_i.real + 1j * data_q.real  # using zero IF

                    self.resp_arr[jj, ii] = np.mean(data[-self.num_averages :])

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

    def save(self, save_filename: str = None) -> str:
        return super().save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "TwoTonePower":
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

    def analyze(self, quantity: str = "quadrature", linecut: bool = False, blit: bool = False):
        if self.control_freq_arr is None:
            raise RuntimeError
        if self.resp_arr is None:
            raise RuntimeError
        if quantity not in ["quadrature", "amplitude", "phase", "dB"]:
            raise ValueError

        import matplotlib.pyplot as plt

        nr_amps = len(self.control_amp_arr)

        self._AMP_IDX = nr_amps // 2

        if quantity == "dB":
            data = 20.0 * np.log10(np.abs(self.resp_arr))
            unit = "dBFS"
            title = "Response amplitude"
        elif quantity == "phase":
            data = np.angle(self.resp_arr)
            unit = "rad"
            title = "Response phase"
        else:
            if quantity == "amplitude":
                data = np.abs(self.resp_arr)
                title = "Response amplitude"
            else:  # "quadrature"
                data = rotate_opt(self.resp_arr).real
                title = "Response quadrature"
            data_max = np.abs(data).max()
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
            unit += "FS"

        amp_dBFS = 20 * np.log10(self.control_amp_arr / 1.0)

        # choose limits for colorbar
        cutoff = 1.0  # %
        lowlim = np.percentile(data, cutoff)
        highlim = np.percentile(data, 100.0 - cutoff)

        # extent
        x_min = 1e-9 * self.control_freq_arr[0]
        x_max = 1e-9 * self.control_freq_arr[-1]
        dx = 1e-9 * (self.control_freq_arr[1] - self.control_freq_arr[0])
        y_min = amp_dBFS[0]
        y_max = amp_dBFS[-1]
        dy = amp_dBFS[1] - amp_dBFS[0]

        if linecut:
            fig1 = plt.figure(tight_layout=True, figsize=(6.4, 7.2))
            gs = fig1.add_gridspec(3, 1)
            ax1 = fig1.add_subplot(gs[:-1, 0])
        else:
            fig1 = plt.figure(tight_layout=True, figsize=(6.4, 4.8))
            # gs = fig1.add_gridspec(1, 1)
            ax1 = fig1.add_subplot(1, 1, 1)
        im = ax1.imshow(
            data,
            origin="lower",
            aspect="auto",
            interpolation="none",
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
        cb.set_label(f"{title:s} [{unit:s}]")

        if linecut:
            ax2 = fig1.add_subplot(gs[-1, 0])

            (line_a,) = ax2.plot(1e-9 * self.control_freq_arr, data[self._AMP_IDX], animated=blit)

            f_min = 1e-9 * self.control_freq_arr.min()
            f_max = 1e-9 * self.control_freq_arr.max()
            f_rng = f_max - f_min
            a_min = data.min()
            a_max = data.max()
            a_rng = a_max - a_min
            ax2.set_xlim(f_min - 0.05 * f_rng, f_max + 0.05 * f_rng)
            ax2.set_ylim(a_min - 0.05 * a_rng, a_max + 0.05 * a_rng)

            ax2.set_xlabel("Frequency [GHz]")
            ax2.set_ylabel(f"{title:s} [{unit:s}]")

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
                # ax2.set_title("")
                if blit:
                    fig1.canvas.restore_region(self._bg)
                    ax1.draw_artist(line_sel)
                    ax2.draw_artist(line_a)
                    fig1.canvas.blit(fig1.bbox)
                    # fig1.canvas.flush_events()
                else:
                    fig1.canvas.draw()

            fig1.canvas.mpl_connect("button_press_event", onbuttonpress)
            fig1.canvas.mpl_connect("key_press_event", onkeypress)

        fig1.show()
        if linecut and blit:
            fig1.canvas.draw()
            # fig1.canvas.flush_events()
            self._bg = fig1.canvas.copy_from_bbox(fig1.bbox)
            ax1.draw_artist(line_sel)
            ax2.draw_artist(line_a)
            fig1.canvas.blit(fig1.bbox)

        return fig1
