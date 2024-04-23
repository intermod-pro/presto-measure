# -*- coding: utf-8 -*-
import os
import time
from typing import Optional

import h5py
import numpy as np

from presto.hardware import AdcMode, DacMode
from presto.pulsed import Pulsed
from presto.utils import get_sourcecode


class Base:
    """
    Base class for measurements
    """

    DAC_CURRENT: int = 32_000
    """μA -- Change to increase or decrease DAC analog output range"""
    ADC_ATTENUATION: float = 0.0  # dB
    """dB -- Change to increase or decrease ADC analog input range"""
    DC_PARAMS: dict = {
        "adc_mode": AdcMode.Mixed,
        "dac_mode": DacMode.Mixed,
    }
    """Parameters to configure the data converters (ADC and DAC)"""

    def _save(self, script_path: str, save_filename: Optional[str] = None) -> str:
        script_path = os.path.realpath(script_path)  # full path of current script

        if save_filename is None:
            current_dir, script_basename = os.path.split(script_path)
            script_filename = os.path.splitext(script_basename)[0]  # name of current script
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # current date and time
            save_basename = f"{script_filename:s}_{timestamp:s}.h5"  # name of save file
            save_path = os.path.join(current_dir, "data", save_basename)  # full path of save file
        else:
            save_path = os.path.realpath(save_filename)

        source_code = get_sourcecode(
            script_path
        )  # save also the sourcecode of the script for future reference
        with h5py.File(save_path, "w") as h5f:
            dt = h5py.string_dtype(encoding="utf-8")
            ds = h5f.create_dataset("source_code", (len(source_code),), dt)
            for ii, line in enumerate(source_code):
                ds[ii] = line

            for attribute in self.__dict__:
                if attribute.startswith("_"):
                    # don't save private attributes
                    continue
                if attribute in ["jpa_params", "clear"]:
                    h5f.attrs[attribute] = str(self.__dict__[attribute])
                elif np.isscalar(self.__dict__[attribute]):
                    h5f.attrs[attribute] = self.__dict__[attribute]
                else:
                    h5f.create_dataset(attribute, data=self.__dict__[attribute])
        print(f"Data saved to: {save_path}")
        return save_path

    def _jpa_setup(self, pls: Pulsed):
        self.jpa_params: Optional[dict]
        if self.jpa_params is not None:
            pls.hardware.set_lmx(
                self.jpa_params["pump_freq"],
                self.jpa_params["pump_pwr"],
                self.jpa_params["pump_port"],
            )
            pls.hardware.set_dc_bias(self.jpa_params["bias"], self.jpa_params["bias_port"])
            pls.hardware.sleep(1.0, False)

    def _jpa_stop(self, pls: Pulsed):
        self.jpa_params: Optional[dict]
        if self.jpa_params is not None:
            pls.hardware.set_lmx(0.0, 0, self.jpa_params["pump_port"])
            pls.hardware.set_dc_bias(0.0, self.jpa_params["bias_port"])

    def _jpa_tweak(self, T: float, pls: Pulsed) -> float:
        self.jpa_params: Optional[dict]
        self.readout_freq: float
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
            return T
        else:
            return T


def project(resp_arr, reference_templates):
    ref_g, ref_e = reference_templates
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
