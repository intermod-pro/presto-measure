# -*- coding: utf-8 -*-
"""
2D sweep of DC bias and frequency of probe to find the modulation curve of the JPA.
"""
from typing import List, Optional, Union

import h5py
import numpy as np
import numpy.typing as npt

from presto.hardware import AdcMode, DacMode, Hardware
from presto import lockin
from presto.utils import ProgressBar

from _base import Base

DAC_CURRENT = 32_000  # uA


class SweepFreqAndDC(Base):
    def __init__(
        self,
        freq_center: float,
        freq_span: float,
        df: float,
        num_averages: int,
        amp: float,
        bias_arr: Union[List[float], npt.NDArray[np.float64]],
        output_port: int,
        input_port: int,
        bias_port: int,
        bias_ramp_rate: float = 0.01,
        dither: bool = True,
        num_skip: int = 0,
    ) -> None:
        self.freq_center = freq_center
        self.freq_span = freq_span
        self.df = df  # modified after tuning
        self.num_averages = num_averages
        self.amp = amp
        self.bias_arr = np.atleast_1d(bias_arr).astype(np.float64)
        self.bias_ramp_rate = bias_ramp_rate
        self.output_port = output_port
        self.input_port = input_port
        self.bias_port = bias_port
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

            nr_bias = len(self.bias_arr)
            _, self.df = lck.tune(0.0, self.df)

            f_start = self.freq_center - self.freq_span / 2
            f_stop = self.freq_center + self.freq_span / 2
            n_start = int(round(f_start / self.df))
            n_stop = int(round(f_stop / self.df))
            n_arr = np.arange(n_start, n_stop + 1)
            nr_freq = len(n_arr)
            self.freq_arr = self.df * n_arr
            self.resp_arr = np.zeros((nr_bias, nr_freq), np.complex128)

            max_bias = max(self.bias_arr)
            min_bias = min(self.bias_arr)
            active_bias, active_range = lck.hardware.get_dc_bias(self.bias_port, get_range=True)
            active_range_max_voltage, active_range_min_voltage = Hardware._dc_max_min(active_range)
            if max_bias > active_range_max_voltage or min_bias < active_range_min_voltage:
                if max_bias > 10 or min_bias < 10:
                    raise ValueError("Value of DC bias has to be between -10 and 10V")
                elif min_bias > 0:
                    if max_bias > 3.33:
                        new_range = 1
                    else:
                        new_range = 0
                elif max_bias > 6.67 or min_bias < -6.67:
                    new_range = 4
                elif max_bias > 3.33 or min_bias < -3.33:
                    new_range = 3
                else:
                    new_range = 2
                if new_range != active_range:
                    lck.hardware.set_dc_bias(active_bias, self.bias_port, new_range)
                    lck.hardware.sleep(1.0, False)
            if active_bias != self.bias_arr[0]:
                lck.hardware.ramp_dc_bias(self.bias_arr[0], self.bias_port, self.bias_ramp_rate)

            lck.hardware.configure_mixer(
                freq=self.freq_arr[0],
                in_ports=self.input_port,
                out_ports=self.output_port,
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

            pb = ProgressBar(nr_bias)
            pb.start()
            for jj, bias in enumerate(self.bias_arr):
                lck.hardware.ramp_dc_bias(bias, self.bias_port, self.bias_ramp_rate)

                for ii, freq in enumerate(self.freq_arr):
                    lck.hardware.configure_mixer(
                        freq=freq,
                        in_ports=self.input_port,
                        out_ports=self.output_port,
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
            # lck.hardware.ramp_dc_bias(0.0, self.bias_port,self.bias_ramp_rate)

        return self.save()

    def save(self, save_filename: Optional[str] = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "SweepFreqAndDC":
        with h5py.File(load_filename, "r") as h5f:
            freq_center = float(h5f.attrs["freq_center"])  # type: ignore
            freq_span = float(h5f.attrs["freq_span"])  # type: ignore
            df = float(h5f.attrs["df"])  # type: ignore
            num_averages = int(h5f.attrs["num_averages"])  # type: ignore
            amp = float(h5f.attrs["amp"])  # type: ignore
            output_port = int(h5f.attrs["output_port"])  # type: ignore
            input_port = int(h5f.attrs["input_port"])  # type: ignore
            bias_port = int(h5f.attrs["bias_port"])  # type: ignore
            bias_ramp_rate = float(h5f.attrs["bias_ramp_rate"])  # type: ignore
            dither = bool(h5f.attrs["dither"])  # type: ignore
            num_skip = int(h5f.attrs["num_skip"])  # type: ignore

            bias_arr: npt.NDArray[np.float64] = h5f["bias_arr"][()]  # type: ignore
            freq_arr: npt.NDArray[np.float64] = h5f["freq_arr"][()]  # type: ignore
            resp_arr: npt.NDArray[np.complex128] = h5f["resp_arr"][()]  # type: ignore

        self = cls(
            freq_center=freq_center,
            freq_span=freq_span,
            df=df,
            num_averages=num_averages,
            amp=amp,
            bias_arr=bias_arr,
            output_port=output_port,
            input_port=input_port,
            bias_port=bias_port,
            bias_ramp_rate=bias_ramp_rate,
            dither=dither,
            num_skip=num_skip,
        )
        self.bias_arr = bias_arr
        self.freq_arr = freq_arr
        self.resp_arr = resp_arr

        return self

    def analyze(self, quantity: str):
        assert self.freq_arr is not None
        assert self.resp_arr is not None

        import matplotlib.pyplot as plt

        # fig, ax = plt.subplots()
        # x = self.freq_arr
        # y = np.unwrap(np.angle(self.resp_arr[0, :]))
        # pfit = np.polyfit(x, y, 1)
        # y -= np.polyval(pfit, x)
        # ax.plot(self.freq_arr, y)
        # fig.show()

        if quantity == "amplitude":
            data = np.abs(self.resp_arr)
            label = "Amplitude [FS]"
        elif quantity == "phase":
            data = np.unwrap(np.unwrap(np.angle(self.resp_arr), axis=1), axis=0)
            # data = np.unwrap(np.angle(self.resp_arr), axis=1)
            label = "Phase [rad]"
        elif quantity == "dB":
            data = 20 * np.log10(np.abs(self.resp_arr))
            label = "Amplitude [dBFS]"
        elif quantity == "group delay":
            _phase = np.unwrap(np.unwrap(np.angle(self.resp_arr), axis=1), axis=0)
            # _phase = np.unwrap(np.angle(self.resp_arr), axis=1)
            dw = 2 * np.pi * self.df
            data = -np.gradient(_phase, axis=1) / dw
            data *= 1e9
            label = "Group delay [ns]"
        elif quantity == "dpdb":
            _phase = np.unwrap(np.unwrap(np.angle(self.resp_arr), axis=1), axis=0)
            # _phase = np.unwrap(np.angle(self.resp_arr), axis=1)
            db = self.bias_arr[1] - self.bias_arr[0]
            data = np.gradient(_phase, axis=0) / db
            label = r"$\mathrm{d}\phi / \mathrm{d} V$ [rad / V]"
        else:
            raise ValueError

        # choose limits for colorbar
        cutoff = 10.0  # %
        lowlim = np.percentile(data, cutoff)
        highlim = np.percentile(data, 100.0 - cutoff)

        # extent
        x_min = 1e-9 * self.freq_arr[0]
        x_max = 1e-9 * self.freq_arr[-1]
        dx = 1e-9 * (self.freq_arr[1] - self.freq_arr[0])
        y_min = self.bias_arr[0]
        y_max = self.bias_arr[-1]
        dy = self.bias_arr[1] - self.bias_arr[0]

        fig1, ax1 = plt.subplots(tight_layout=True)
        im = ax1.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
            vmin=lowlim,
            vmax=highlim,
        )
        ax1.set_xlabel("Frequency [GHz]")
        ax1.set_ylabel("Bias [V]")
        cb = fig1.colorbar(im)
        cb.set_label(label)
        plt.show()

        return fig1
