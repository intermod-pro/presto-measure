# -*- coding: utf-8 -*-
import os
import time
from typing import Optional

import h5py
import numpy as np

from presto.utils import get_sourcecode


class Base:
    """
    Base class for measurements
    """

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
