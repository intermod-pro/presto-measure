# -*- coding: utf-8 -*-
import ast
import os
import signal
import time
from typing import List, Optional, Union

import h5py
from matplotlib import _pylab_helpers
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
import numpy.typing as npt

from presto.utils import format_precision

from _base import Base
from ramsey_echo import RamseyEcho
from t1 import T1 as T1Class

KEEP_GOING = True


class CycleTs(Base):
    def __init__(
        self,
        readout_freq: float,
        control_freq: float,
        readout_amp: float,
        control_amp_90: float,
        control_amp_180: float,
        readout_duration: float,
        control_duration: float,
        sample_duration: float,
        delay_arr: Union[List[float], npt.NDArray[np.float64]],
        readout_port: int,
        control_port: int,
        sample_port: int,
        wait_delay: float,
        readout_sample_delay: float,
        num_averages: int,
        jpa_params: Optional[dict] = None,
        drag: float = 0.0,
        ref_g: Optional[List[complex]] = None,
        ref_e: Optional[List[complex]] = None,
    ) -> None:
        self.readout_freq = readout_freq
        self.control_freq = control_freq
        self.readout_amp = readout_amp
        self.control_amp_90 = control_amp_90
        self.control_amp_180 = control_amp_180
        self.readout_duration = readout_duration
        self.control_duration = control_duration
        self.sample_duration = sample_duration
        self.delay_arr = np.atleast_1d(delay_arr).astype(np.float64)
        self.readout_port = readout_port
        self.control_port = control_port
        self.sample_port = sample_port
        self.wait_delay = wait_delay
        self.readout_sample_delay = readout_sample_delay
        self.num_averages = num_averages
        self.jpa_params = jpa_params
        self.drag = drag
        # self.ref_g = ref_g
        # self.ref_e = ref_e

        self.time_start: float = 0.0  # replaced by run
        self._nr_delays = len(self.delay_arr)

        if ref_g is None or ref_e is None:
            self._ref_templates = None
        else:
            self._ref_templates = (ref_g, ref_e)

        self._save_filename: str = ""  # replaced by save
        self._time1_arr = np.zeros(0, np.float64)
        self._time2_arr = np.zeros(0, np.float64)
        self._t1_arr = np.zeros(0, np.float64)
        self._t2_arr = np.zeros(0, np.float64)
        self._t1_err_arr = np.zeros(0, np.float64)
        self._t2_err_arr = np.zeros(0, np.float64)
        # self._data1 = np.zeros((0, self._nr_delays), np.float64)
        # self._data2 = np.zeros((0, self._nr_delays), np.float64)
        self._data1 = np.zeros(0, np.float64)  # will have shape (self._nr_delays,)
        self._data2 = np.zeros(0, np.float64)  # will have shape (self._nr_delays,)

    def run(
        self,
        presto_address: str,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
    ):
        self.time_start = time.time()
        # save initial parameters
        self.save()

        fig, ax = plt.subplots(constrained_layout=True)

        (line_t1,) = ax.plot(self._time1_arr, self._t1_arr, ".", c="tab:blue", label="T1")
        (line_t2,) = ax.plot(self._time2_arr, self._t2_arr, ".", c="tab:orange", label="T2")

        ax.set_ylabel("μs")
        ax.set_xlabel("Time since start [s]")
        ax.legend()

        fig.show()
        _my_pause()

        signal.signal(signal.SIGINT, _handler)
        count = 0
        global KEEP_GOING
        while KEEP_GOING:
            print("\n\n\n")
            print(f"******* Run number {count+1:d} *******")
            count += 1

            print("\n")
            print("------- measure T1 -------")
            self._data1, t1, t1_err = self.measure_t1(presto_address, presto_port, ext_ref_clk)
            print("T1 = {:s} μs".format(format_precision(1e6 * t1, 1e6 * t1_err)))

            # self._data1 = np.vstack((self._data1, data1))
            self._t1_arr = np.r_[self._t1_arr, t1]
            self._t1_err_arr = np.r_[self._t1_err_arr, t1_err]
            self._time1_arr = np.r_[self._time1_arr, time.time()]
            self.append(1)

            line_t1.set_data(self._time1_arr - self.time_start, 1e6 * self._t1_arr)
            ax.relim()
            ax.autoscale()
            _my_pause(1.0)

            print("\n")
            print("------- measure T2 -------")
            self._data2, t2, t2_err = self.measure_t2(presto_address, presto_port, ext_ref_clk)
            print("T2 = {:s} μs".format(format_precision(1e6 * t2, 1e6 * t2_err)))

            # self._data2 = np.vstack((self._data2, data2))
            self._t2_arr = np.r_[self._t2_arr, t2]
            self._t2_err_arr = np.r_[self._t2_err_arr, t2_err]
            self._time2_arr = np.r_[self._time2_arr, time.time()]
            self.append(2)

            line_t2.set_data(self._time2_arr - self.time_start, 1e6 * self._t2_arr)
            ax.relim()
            ax.autoscale()
            _my_pause(1.0)

        print("\n\n\n")
        print("Done")
        input("___ Press Enter to close ___")

    def save(self, save_filename: Optional[str] = None) -> str:
        # save parameters
        self._save_filename = super()._save(__file__, save_filename=save_filename)
        # add growable arrays
        with h5py.File(self._save_filename, "a") as h5f:
            # h5f.create_dataset('data1', data=self._data1, compression="gzip", chunks=True, maxshape=(None, self._nr_delays))
            # h5f.create_dataset('data2', data=self._data2, compression="gzip", chunks=True, maxshape=(None, self._nr_delays))
            h5f.create_dataset(
                "data1",
                shape=(0, self._nr_delays),
                dtype=np.float64,
                compression="gzip",
                chunks=True,
                maxshape=(None, self._nr_delays),
            )
            h5f.create_dataset(
                "data2",
                shape=(0, self._nr_delays),
                dtype=np.float64,
                compression="gzip",
                chunks=True,
                maxshape=(None, self._nr_delays),
            )
            h5f.create_dataset(
                "time1_arr",
                data=self._time1_arr,
                compression="gzip",
                chunks=True,
                maxshape=(None,),
            )
            h5f.create_dataset(
                "time2_arr",
                data=self._time2_arr,
                compression="gzip",
                chunks=True,
                maxshape=(None,),
            )
            h5f.create_dataset(
                "t1_arr", data=self._t1_arr, compression="gzip", chunks=True, maxshape=(None,)
            )
            h5f.create_dataset(
                "t2_arr", data=self._t2_arr, compression="gzip", chunks=True, maxshape=(None,)
            )
            h5f.create_dataset(
                "t1_err_arr",
                data=self._t1_err_arr,
                compression="gzip",
                chunks=True,
                maxshape=(None,),
            )
            h5f.create_dataset(
                "t2_err_arr",
                data=self._t2_err_arr,
                compression="gzip",
                chunks=True,
                maxshape=(None,),
            )
        return self._save_filename

    @staticmethod
    def _append(h5f, ds, data):
        data = np.atleast_1d(data)
        h5f[ds].resize((h5f[ds].shape[0] + data.shape[0]), axis=0)
        h5f[ds][-data.shape[0] :] = data

    def append(self, which: int = 3):
        with h5py.File(self._save_filename, "a") as h5f:
            if which & 0b01 > 0:
                self._append(h5f, "data1", self._data1)
                self._append(h5f, "time1_arr", self._time1_arr[-1])
                self._append(h5f, "t1_arr", self._t1_arr[-1])
                self._append(h5f, "t1_err_arr", self._t1_err_arr[-1])
            if which & 0b10 > 0:
                self._append(h5f, "data2", self._data2)
                self._append(h5f, "time2_arr", self._time2_arr[-1])
                self._append(h5f, "t2_arr", self._t2_arr[-1])
                self._append(h5f, "t2_err_arr", self._t2_err_arr[-1])

        print(f"Data appended to: {self._save_filename}")

    @classmethod
    def load(cls, load_filename: str) -> "CycleTs":
        with h5py.File(load_filename, "r") as h5f:
            readout_freq = float(h5f.attrs["readout_freq"])  # type: ignore
            control_freq = float(h5f.attrs["control_freq"])  # type: ignore
            readout_amp = float(h5f.attrs["readout_amp"])  # type: ignore
            control_amp_90 = float(h5f.attrs["control_amp_90"])  # type: ignore
            control_amp_180 = float(h5f.attrs["control_amp_180"])  # type: ignore
            readout_duration = float(h5f.attrs["readout_duration"])  # type: ignore
            control_duration = float(h5f.attrs["control_duration"])  # type: ignore
            sample_duration = float(h5f.attrs["sample_duration"])  # type: ignore
            delay_arr = np.array(h5f["delay_arr"])
            readout_port = int(h5f.attrs["readout_port"])  # type: ignore
            control_port = int(h5f.attrs["control_port"])  # type: ignore
            sample_port = int(h5f.attrs["sample_port"])  # type: ignore
            wait_delay = float(h5f.attrs["wait_delay"])  # type: ignore
            readout_sample_delay = float(h5f.attrs["readout_sample_delay"])  # type: ignore
            num_averages = int(h5f.attrs["num_averages"])  # type: ignore
            jpa_params = ast.literal_eval(h5f.attrs["jpa_params"])  # type: ignore
            drag = float(h5f.attrs["drag"])  # type: ignore

            time_start = float(h5f.attrs["time_start"])  # type: ignore
            time1_arr = np.array(h5f["time1_arr"])
            time2_arr = np.array(h5f["time2_arr"])
            t1_arr = np.array(h5f["t1_arr"])
            t2_arr = np.array(h5f["t2_arr"])
            t1_err_arr = np.array(h5f["t1_err_arr"])
            t2_err_arr = np.array(h5f["t2_err_arr"])

        self = cls(
            readout_freq=readout_freq,
            control_freq=control_freq,
            readout_amp=readout_amp,
            control_amp_90=control_amp_90,
            control_amp_180=control_amp_180,
            readout_duration=readout_duration,
            control_duration=control_duration,
            sample_duration=sample_duration,
            delay_arr=delay_arr,
            readout_port=readout_port,
            control_port=control_port,
            sample_port=sample_port,
            wait_delay=wait_delay,
            readout_sample_delay=readout_sample_delay,
            num_averages=num_averages,
            jpa_params=jpa_params,
            drag=drag,
        )
        self.time_start = time_start
        self._time1_arr = time1_arr
        self._time2_arr = time2_arr
        self._t1_arr = t1_arr
        self._t2_arr = t2_arr
        self._t1_err_arr = t1_err_arr
        self._t2_err_arr = t2_err_arr

        return self

    def analyze(self, selector=True):
        ret_fig = []

        time1_arr = self._time1_arr - self.time_start
        time2_arr = self._time2_arr - self.time_start
        max_time = max(np.max(time1_arr), np.max(time2_arr))
        if max_time > 86_400:
            time1_arr /= 86_400
            time2_arr /= 86_400
            time_unit = "days"
        elif max_time > 3_600:
            time1_arr /= 3_600
            time2_arr /= 3_600
            time_unit = "hours"
        elif max_time > 60:
            time1_arr /= 60
            time2_arr /= 60
            time_unit = "minutes"
        else:
            time_unit = "seconds"

        # plot vs time
        fig1, ax1 = plt.subplots(constrained_layout=True)

        ax1.plot(time1_arr, 1e6 * self._t1_arr, ".", c="tab:blue", label="$T_1$")
        ax1.plot(time2_arr, 1e6 * self._t2_arr, ".", c="tab:orange", label="$T_2^\\mathrm{echo}$")

        ax1.set_ylabel("Fitted Tx [μs]")
        ax1.set_xlabel(f"Time since start [{time_unit:s}]")
        ax1.legend()
        ax1.grid()

        fig1.show()
        ret_fig.append(fig1)

        # statistics
        _print_stats(self._t1_arr, self._t2_arr)

        # histogram
        fig2, ax2 = plt.subplots(2, 1, sharex=True, sharey=True, constrained_layout=True)
        ax21, ax22 = ax2
        _plot_histograms(self._t1_arr, self._t2_arr, ax21, ax22)
        fig2.show()
        ret_fig.append(fig1)

        if selector:

            def onselect(xmin, xmax):
                idx1 = np.logical_and(time1_arr >= xmin, time1_arr < xmax)
                idx2 = np.logical_and(time2_arr >= xmin, time2_arr < xmax)
                _print_stats(self._t1_arr[idx1], self._t2_arr[idx2])
                _plot_histograms(self._t1_arr[idx1], self._t2_arr[idx2], ax21, ax22)
                fig2.canvas.draw()

            rectprops = dict(facecolor="tab:gray", alpha=0.5)
            span = mwidgets.SpanSelector(ax1, onselect, "horizontal", props=rectprops)
            fig1._span = span  # keep references to span selector  # type: ignore

        return ret_fig

    def measure_t1(self, presto_address, presto_port, ext_ref_clk):
        m = T1Class(
            readout_freq=self.readout_freq,
            control_freq=self.control_freq,
            readout_amp=self.readout_amp,
            control_amp=self.control_amp_180,
            readout_duration=self.readout_duration,
            control_duration=self.control_duration,
            sample_duration=self.sample_duration,
            delay_arr=self.delay_arr,
            readout_port=self.readout_port,
            control_port=self.control_port,
            sample_port=self.sample_port,
            wait_delay=self.wait_delay,
            readout_sample_delay=self.readout_sample_delay,
            num_averages=self.num_averages,
            jpa_params=self.jpa_params,
            drag=self.drag,
        )
        m.run(presto_address, presto_port, ext_ref_clk, save=False)
        data, (popt, perr) = m.analyze_batch(self._ref_templates)

        t1 = np.nan if popt is None else popt[0]
        t1_err = np.nan if perr is None else perr[0]

        return data, t1, t1_err

    def measure_t2(self, presto_address, presto_port, ext_ref_clk):
        m = RamseyEcho(
            readout_freq=self.readout_freq,
            control_freq=self.control_freq,
            readout_amp=self.readout_amp,
            control_amp_90=self.control_amp_90,
            control_amp_180=self.control_amp_180,
            readout_duration=self.readout_duration,
            control_duration=self.control_duration,
            sample_duration=self.sample_duration,
            delay_arr=self.delay_arr,
            readout_port=self.readout_port,
            control_port=self.control_port,
            sample_port=self.sample_port,
            wait_delay=self.wait_delay,
            readout_sample_delay=self.readout_sample_delay,
            num_averages=self.num_averages,
            jpa_params=self.jpa_params,
            drag=self.drag,
        )
        m.run(presto_address, presto_port, ext_ref_clk, save=False)
        data, (popt, perr) = m.analyze_batch(self._ref_templates)

        t2 = np.nan if popt is None else popt[0]
        t2_err = np.nan if perr is None else perr[0]

        return data, t2, t2_err


def get_save_filename():
    script_path = os.path.realpath(__file__)  # full path of current script
    current_dir, script_basename = os.path.split(script_path)
    script_filename = os.path.splitext(script_basename)[0]  # name of current script
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # current date and time
    save_basename = f"{script_filename:s}_{timestamp:s}.h5"  # name of save file
    save_path = os.path.join(current_dir, "data", save_basename)  # full path of save file
    return save_path


def _my_pause(interval=0.1):
    manager = _pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
        # plt.show(block=False)
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)


def _handler(signum, frame):
    global KEEP_GOING
    if KEEP_GOING:
        print("\n\n")
        print("Ctrl-C pressed!")
        print("Will finish this run and then stop.")
        print("Press Ctrl-C again to abort.")
        print("\n\n")
        KEEP_GOING = False
    else:
        raise KeyboardInterrupt


def _plot_histograms(data1, data2, ax1, ax2):
    ax1.cla()
    ax2.cla()
    _make_hist(data1, ax1, c="tab:blue", label="$T_1$")
    _make_hist(data2, ax2, c="tab:orange", label="$T_2^\\mathrm{echo}$")
    ax1.legend()
    ax2.legend()
    ax1.set_ylabel("Counts")
    ax2.set_ylabel("Counts")
    ax2.set_xlabel("Time [μs]")


def _make_hist(data, ax, **kwargs):
    nr_bins = int(round(np.sqrt(data.shape[0])))
    hist1, xedges1 = np.histogram(1e6 * data, bins=nr_bins, density=False)  # range=(x_min, x_max)
    _hist_plot(ax, hist1, xedges1, **kwargs)


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


def _irq(data):
    return np.percentile(data, 75) - np.percentile(data, 25)


def _print_stats(data1, data2):
    t1_arr = 1e6 * data1
    t2_arr = 1e6 * data2
    print()
    print(f"Mean:\t{np.mean(t1_arr):.3g}\t{np.mean(t2_arr):.3g}")
    print(f"Std:\t{np.std(t1_arr):.3g}\t{np.std(t2_arr):.3g}")
    print(f"Median:\t{np.median(t1_arr):.3g}\t{np.median(t2_arr):.3g}")
    print(f"IQR:\t{_irq(t1_arr):.3g}\t{_irq(t2_arr):.3g}")
