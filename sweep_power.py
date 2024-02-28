# -*- coding: utf-8 -*-
"""
2D sweep of drive power and frequency in Lockin mode.
"""
from typing import List, Optional, Union

import h5py
import numpy as np
import numpy.typing as npt

from presto.hardware import AdcMode, DacMode
from presto import lockin
from presto.utils import ProgressBar

from _base import Base

DAC_CURRENT = 32_000  # uA


class SweepPower(Base):
    def __init__(
        self,
        freq_center: float,
        freq_span: float,
        df: float,
        num_averages: int,
        amp_arr: Union[List[float], npt.NDArray[np.float64]],
        output_port: int,
        input_port: int,
        dither: bool = True,
        num_skip: int = 0,
    ) -> None:
        self.freq_center = freq_center
        self.freq_span = freq_span
        self.df = df  # modified after tuning
        self.num_averages = num_averages
        self.amp_arr = np.atleast_1d(amp_arr).astype(np.float64)
        self.output_port = output_port
        self.input_port = input_port
        self.dither = dither
        self.num_skip = num_skip

        self.freq_arr = None  # replaced by run
        self.resp_arr = None  # replaced by run

    def run(
        self,
        presto_address: str,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
    ) -> str:
        with lockin.Lockin(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            adc_mode=AdcMode.Mixed,
            dac_mode=DacMode.Mixed,
        ) as lck:
            lck.hardware.set_adc_attenuation(self.input_port, 0.0)
            lck.hardware.set_dac_current(self.output_port, DAC_CURRENT)
            lck.hardware.set_inv_sinc(self.output_port, 0)

            nr_amps = len(self.amp_arr)

            # tune frequencies
            _, self.df = lck.tune(0.0, self.df)
            lck.set_df(self.df)

            f_start = self.freq_center - self.freq_span / 2
            f_stop = self.freq_center + self.freq_span / 2
            n_start = int(round(f_start / self.df))
            n_stop = int(round(f_stop / self.df))
            n_arr = np.arange(n_start, n_stop + 1)
            nr_freq = len(n_arr)
            self.freq_arr = self.df * n_arr
            self.resp_arr = np.zeros((nr_amps, nr_freq), np.complex128)

            lck.hardware.configure_mixer(
                freq=self.freq_arr[0], in_ports=self.input_port, out_ports=self.output_port
            )

            og = lck.add_output_group(self.output_port, 1)
            og.set_frequencies(0.0)
            og.set_amplitudes(self.amp_arr[0])
            og.set_phases(0.0, 0.0)

            lck.set_dither(self.dither, self.output_port)
            ig = lck.add_input_group(self.input_port, 1)
            ig.set_frequencies(0.0)

            lck.apply_settings()

            pb = ProgressBar(nr_amps * nr_freq)
            pb.start()
            for jj, amp in enumerate(self.amp_arr):
                og.set_amplitudes(amp)
                lck.apply_settings()

                for ii, freq in enumerate(self.freq_arr):
                    lck.hardware.configure_mixer(
                        freq=freq, in_ports=self.input_port, out_ports=self.output_port
                    )
                    lck.apply_settings()

                    _d = lck.get_pixels(self.num_skip + self.num_averages, quiet=True)
                    data_i = _d[self.input_port][1][:, 0]
                    data_q = _d[self.input_port][2][:, 0]
                    data = data_i.real + 1j * data_q.real  # using zero IF

                    self.resp_arr[jj, ii] = np.mean(data[-self.num_averages :])

                    pb.increment()

            pb.done()

            # Mute outputs at the end of the sweep
            og.set_amplitudes(0.0)
            lck.apply_settings()

        return self.save()

    def save(self, save_filename: Optional[str] = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "SweepPower":
        with h5py.File(load_filename, "r") as h5f:
            freq_center = float(h5f.attrs["freq_center"])  # type: ignore
            freq_span = float(h5f.attrs["freq_span"])  # type: ignore
            df = float(h5f.attrs["df"])  # type: ignore
            num_averages = int(h5f.attrs["num_averages"])  # type: ignore
            output_port = int(h5f.attrs["output_port"])  # type: ignore
            input_port = int(h5f.attrs["input_port"])  # type: ignore
            dither = bool(h5f.attrs["dither"])  # type: ignore
            num_skip = int(h5f.attrs["num_skip"])  # type: ignore

            amp_arr: npt.NDArray[np.float64] = h5f["amp_arr"][()]  # type: ignore
            freq_arr: npt.NDArray[np.float64] = h5f["freq_arr"][()]  # type: ignore
            resp_arr: npt.NDArray[np.complex128] = h5f["resp_arr"][()]  # type: ignore

        self = cls(
            freq_center=freq_center,
            freq_span=freq_span,
            df=df,
            num_averages=num_averages,
            amp_arr=amp_arr,
            output_port=output_port,
            input_port=input_port,
            dither=dither,
            num_skip=num_skip,
        )
        self.freq_arr = freq_arr
        self.resp_arr = resp_arr

        return self

    def analyze(self, norm: bool = True, portrait: bool = True, blit: bool = False):
        if self.freq_arr is None:
            raise RuntimeError
        if self.resp_arr is None:
            raise RuntimeError

        import matplotlib.pyplot as plt

        try:
            from resonator_tools import circuit
            import matplotlib.widgets as mwidgets

            _do_fit = True
        except ImportError:
            _do_fit = False

        nr_amps = len(self.amp_arr)
        self._AMP_IDX = nr_amps // 2

        if norm:
            resp_scaled = np.zeros_like(self.resp_arr)
            for jj in range(nr_amps):
                resp_scaled[jj] = self.resp_arr[jj] / self.amp_arr[jj]
        else:
            resp_scaled = self.resp_arr

        resp_dB = 20.0 * np.log10(np.abs(resp_scaled))
        amp_dBFS = 20 * np.log10(self.amp_arr / 1.0)

        # choose limits for colorbar
        cutoff = 1.0  # %
        lowlim = np.percentile(resp_dB, cutoff)
        highlim = np.percentile(resp_dB, 100.0 - cutoff)

        # extent
        x_min = 1e-9 * self.freq_arr[0]
        x_max = 1e-9 * self.freq_arr[-1]
        dx = 1e-9 * (self.freq_arr[1] - self.freq_arr[0])
        y_min = amp_dBFS[0]
        y_max = amp_dBFS[-1]
        dy = amp_dBFS[1] - amp_dBFS[0]

        if portrait:
            fig1 = plt.figure(tight_layout=True, figsize=(6.4, 9.6))
            ax1 = fig1.add_subplot(2, 1, 1)
        else:
            fig1 = plt.figure(tight_layout=True, figsize=(12.8, 4.8))
            ax1 = fig1.add_subplot(1, 2, 1)
        im = ax1.imshow(
            resp_dB,
            origin="lower",
            aspect="auto",
            interpolation="none",
            extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
            vmin=lowlim,  # type: ignore
            vmax=highlim,  # type: ignore
        )
        line_sel = ax1.axhline(amp_dBFS[self._AMP_IDX], ls="--", c="k", lw=3, animated=blit)
        # ax1.set_title(f"amp = {amp_arr[AMP_IDX]:.2e}")
        ax1.set_xlabel("Frequency [GHz]")
        ax1.set_ylabel("Drive amplitude [dBFS]")
        cb = fig1.colorbar(im)
        if portrait:
            cb.set_label("Response amplitude [dB]")
        else:
            ax1.set_title("Response amplitude [dB]")
        fig1.show()
        # return fig1

        if portrait:
            ax2 = fig1.add_subplot(4, 1, 3)
            ax3 = fig1.add_subplot(4, 1, 4, sharex=ax2)
        else:
            ax2 = fig1.add_subplot(2, 2, 2)
            ax3 = fig1.add_subplot(2, 2, 4, sharex=ax2)
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            ax3.yaxis.set_label_position("right")
            ax3.yaxis.tick_right()

        (line_a,) = ax2.plot(
            1e-9 * self.freq_arr, resp_dB[self._AMP_IDX], ".", label="measured", animated=blit
        )
        (line_p,) = ax3.plot(
            1e-9 * self.freq_arr, np.angle(self.resp_arr[self._AMP_IDX]), ".", animated=blit
        )
        if _do_fit:
            (line_fit_a,) = ax2.plot(
                1e-9 * self.freq_arr,
                np.full_like(self.freq_arr, np.nan),
                ls="--",
                label="fit",
                animated=blit,
            )
            (line_fit_p,) = ax3.plot(
                1e-9 * self.freq_arr, np.full_like(self.freq_arr, np.nan), ls="--", animated=blit
            )

        f_min = 1e-9 * self.freq_arr.min()
        f_max = 1e-9 * self.freq_arr.max()
        f_rng = f_max - f_min
        a_min = resp_dB.min()
        a_max = resp_dB.max()
        a_rng = a_max - a_min
        p_min = -np.pi
        p_max = np.pi
        p_rng = p_max - p_min
        ax2.set_xlim(f_min - 0.05 * f_rng, f_max + 0.05 * f_rng)
        ax2.set_ylim(a_min - 0.05 * a_rng, a_max + 0.05 * a_rng)
        ax3.set_xlim(f_min - 0.05 * f_rng, f_max + 0.05 * f_rng)
        ax3.set_ylim(p_min - 0.05 * p_rng, p_max + 0.05 * p_rng)

        ax3.set_xlabel("Frequency [GHz]")
        ax2.set_ylabel("Response amplitude [dB]")
        ax3.set_ylabel("Response phase [rad]")
        ax2.xaxis.set_tick_params(labelbottom=False)

        ax2.legend(loc="lower right")

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
            assert self.resp_arr is not None
            line_sel.set_ydata([amp_dBFS[self._AMP_IDX], amp_dBFS[self._AMP_IDX]])
            # ax1.set_title(f"amp = {amp_arr[AMP_IDX]:.2e}")
            print(
                f"drive amp {self._AMP_IDX:d}: {self.amp_arr[self._AMP_IDX]:.2e} FS = {amp_dBFS[self._AMP_IDX]:.1f} dBFS"
            )
            line_a.set_ydata(resp_dB[self._AMP_IDX])
            line_p.set_ydata(np.angle(self.resp_arr[self._AMP_IDX]))
            if _do_fit:
                line_fit_a.set_ydata(np.full_like(self.freq_arr, np.nan))  # pyright: ignore [reportUnboundVariable]
                line_fit_p.set_ydata(np.full_like(self.freq_arr, np.nan))  # pyright: ignore [reportUnboundVariable]
            # ax2.set_title("")
            if blit:
                fig1.canvas.restore_region(self._bg)  # type: ignore
                ax1.draw_artist(line_sel)
                ax2.draw_artist(line_a)
                ax3.draw_artist(line_p)
                fig1.canvas.blit(fig1.bbox)
                fig1.canvas.flush_events()
            else:
                fig1.canvas.draw()

        if _do_fit:

            def onselect(xmin, xmax):
                assert self.resp_arr is not None
                port = circuit.notch_port(self.freq_arr, self.resp_arr[self._AMP_IDX])  # pyright: ignore [reportUnboundVariable]
                port.autofit(fcrop=(xmin * 1e9, xmax * 1e9))
                if norm:
                    line_fit_a.set_data(  # pyright: ignore [reportUnboundVariable]
                        1e-9 * port.f_data,  # type: ignore
                        20 * np.log10(np.abs(port.z_data_sim / self.amp_arr[self._AMP_IDX])),
                    )
                else:
                    line_fit_a.set_data(1e-9 * port.f_data, 20 * np.log10(np.abs(port.z_data_sim)))  # pyright: ignore
                line_fit_p.set_data(1e-9 * port.f_data, np.angle(port.z_data_sim))  # pyright: ignore
                # print(port.fitresults)
                print("----------------")
                print(f"fr = {port.fitresults['fr']}")
                print(f"Qi = {port.fitresults['Qi_dia_corr']}")
                print(f"Qc = {port.fitresults['Qc_dia_corr']}")
                print(f"Ql = {port.fitresults['Ql']}")
                print(f"kappa = {port.fitresults['fr'] / port.fitresults['Qc_dia_corr']}")
                print("----------------")
                # ax2.set_title(
                #     f"fr = {1e-6*fr:.0f} MHz, Ql = {Ql:.0f}, Qi = {Qi:.0f}, Qc = {Qc:.0f}, kappa = {1e-3*kappa:.0f} kHz")
                if blit:
                    fig1.canvas.restore_region(self._bg)  # type: ignore
                    ax1.draw_artist(line_sel)
                    ax2.draw_artist(line_a)
                    ax2.draw_artist(line_fit_a)  # pyright: ignore [reportUnboundVariable]
                    ax3.draw_artist(line_p)
                    ax3.draw_artist(line_fit_p)  # pyright: ignore [reportUnboundVariable]
                    fig1.canvas.blit(fig1.bbox)
                    fig1.canvas.flush_events()
                else:
                    fig1.canvas.draw()

            rectprops = dict(facecolor="tab:gray", alpha=0.5)
            fig1._span_a = mwidgets.SpanSelector(  # type: ignore
                ax2, onselect, "horizontal", props=rectprops, useblit=blit
            )
            fig1._span_p = mwidgets.SpanSelector(  # type: ignore
                ax3, onselect, "horizontal", props=rectprops, useblit=blit
            )

        fig1.canvas.mpl_connect("button_press_event", onbuttonpress)
        fig1.canvas.mpl_connect("key_press_event", onkeypress)
        fig1.show()
        if blit:
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            self._bg = fig1.canvas.copy_from_bbox(fig1.bbox)  # type: ignore
            ax1.draw_artist(line_sel)
            ax2.draw_artist(line_a)
            ax3.draw_artist(line_p)
            fig1.canvas.blit(fig1.bbox)

        return fig1
