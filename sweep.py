# -*- coding: utf-8 -*-
"""
Simple frequency sweep using the Lockin mode.
"""
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt

from presto.hardware import AdcMode, DacMode
from presto import lockin
from presto.utils import ProgressBar

from _base import Base

DAC_CURRENT = 32_000  # uA


class Sweep(Base):
    def __init__(
        self,
        freq_center: float,
        freq_span: float,
        df: float,
        num_averages: int,
        amp: float,
        output_port: int,
        input_port: int,
        dither: bool = True,
        num_skip: int = 0,
    ) -> None:
        self.freq_center = freq_center
        self.freq_span = freq_span
        self.df = df  # modified after tuning
        self.num_averages = num_averages
        self.amp = amp
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

            # tune frequencies
            _, self.df = lck.tune(0.0, self.df)
            f_start = self.freq_center - self.freq_span / 2
            f_stop = self.freq_center + self.freq_span / 2
            n_start = int(round(f_start / self.df))
            n_stop = int(round(f_stop / self.df))
            n_arr = np.arange(n_start, n_stop + 1)
            nr_freq = len(n_arr)
            self.freq_arr = self.df * n_arr
            self.resp_arr = np.zeros(nr_freq, np.complex128)

            lck.hardware.configure_mixer(
                freq=self.freq_arr[0], in_ports=self.input_port, out_ports=self.output_port
            )

            lck.set_df(self.df)
            og = lck.add_output_group(self.output_port, 1)
            og.set_frequencies(0.0)
            og.set_amplitudes(self.amp)
            og.set_phases(0.0, 0.0)

            lck.set_dither(self.dither, self.output_port)

            ig = lck.add_input_group(self.input_port, 1)
            ig.set_frequencies(0.0)

            lck.apply_settings()

            pb = ProgressBar(nr_freq)
            pb.start()
            for ii in range(len(n_arr)):
                f = self.freq_arr[ii]
                lck.hardware.configure_mixer(
                    freq=f, in_ports=self.input_port, out_ports=self.output_port
                )
                lck.apply_settings()

                _d = lck.get_pixels(self.num_skip + self.num_averages, quiet=True)
                data_i = _d[self.input_port][1][:, 0]
                data_q = _d[self.input_port][2][:, 0]
                data = data_i.real + 1j * data_q.real  # using zero IF

                self.resp_arr[ii] = np.mean(data[-self.num_averages :])

                pb.increment()

            pb.done()

            # Mute outputs at the end of the sweep
            og.set_amplitudes(0.0)
            lck.apply_settings()

        return self.save()

    def save(self, save_filename: Optional[str] = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "Sweep":
        with h5py.File(load_filename, "r") as h5f:
            freq_center = float(h5f.attrs["freq_center"])  # type: ignore
            freq_span = float(h5f.attrs["freq_span"])  # type: ignore
            df = float(h5f.attrs["df"])  # type: ignore
            num_averages = int(h5f.attrs["num_averages"])  # type: ignore
            amp = float(h5f.attrs["amp"])  # type: ignore
            output_port = int(h5f.attrs["output_port"])  # type: ignore
            input_port = int(h5f.attrs["input_port"])  # type: ignore
            dither = bool(h5f.attrs["dither"])  # type: ignore
            num_skip = int(h5f.attrs["num_skip"])  # type: ignore

            freq_arr: npt.NDArray[np.float64] = h5f["freq_arr"][()]  # type: ignore
            resp_arr: npt.NDArray[np.complex128] = h5f["resp_arr"][()]  # type: ignore

        self = cls(
            freq_center=freq_center,
            freq_span=freq_span,
            df=df,
            num_averages=num_averages,
            amp=amp,
            output_port=output_port,
            input_port=input_port,
            dither=dither,
            num_skip=num_skip,
        )
        self.freq_arr = freq_arr
        self.resp_arr = resp_arr

        return self

    def analyze(self):
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

        resp_dB = 20.0 * np.log10(np.abs(self.resp_arr))

        fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
        ax11, ax12 = ax1
        ax11.plot(1e-9 * self.freq_arr, resp_dB, ".")
        # ax11.plot(1e-9 * freq_arr, np.abs(resp_arr))
        (line_fit_a,) = ax11.plot(
            1e-9 * self.freq_arr, np.full_like(self.freq_arr, np.nan), ls="--"
        )
        ax12.plot(1e-9 * self.freq_arr, np.angle(self.resp_arr), ".", label="data")
        (line_fit_p,) = ax12.plot(
            1e-9 * self.freq_arr, np.full_like(self.freq_arr, np.nan), ls="--", label="fit"
        )
        ax12.set_xlabel("Frequency [GHz]")
        ax11.set_ylabel("Amplitude [dB]")
        ax12.set_ylabel("Phase [rad]")
        ax12.legend()

        if _do_fit:

            def onselect(xmin, xmax):
                port = circuit.notch_port(self.freq_arr, self.resp_arr)  # pyright: ignore[reportUnboundVariable]
                port.autofit(fcrop=(xmin * 1e9, xmax * 1e9))
                sim_db = 20 * np.log10(np.abs(port.z_data_sim))
                line_fit_a.set_data(1e-9 * port.f_data, sim_db)  # type: ignore
                line_fit_p.set_data(1e-9 * port.f_data, np.angle(port.z_data_sim))  # type: ignore
                f_min = port.f_data[np.argmin(sim_db)]  # type: ignore
                print("----------------")
                print(f"fr = {port.fitresults['fr']}")
                print(f"Qi = {port.fitresults['Qi_dia_corr']}")
                print(f"Qc = {port.fitresults['Qc_dia_corr']}")
                print(f"Ql = {port.fitresults['Ql']}")
                print(f"kappa = {port.fitresults['fr'] / port.fitresults['Qc_dia_corr']}")
                print(f"f_min = {f_min}")
                print("----------------")
                fig1.canvas.draw()

            rectprops = dict(facecolor="tab:gray", alpha=0.5)
            span_a = mwidgets.SpanSelector(ax11, onselect, "horizontal", props=rectprops)  # pyright: ignore[reportUnboundVariable]
            span_p = mwidgets.SpanSelector(ax12, onselect, "horizontal", props=rectprops)  # pyright: ignore[reportUnboundVariable]
            # keep references to span selectors
            fig1._span_a = span_a  # type: ignore
            fig1._span_p = span_p  # type: ignore
        fig1.show()

        return fig1
