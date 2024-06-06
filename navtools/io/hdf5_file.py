"""
|========================================= hdf5_file.py ===========================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/io/hdf5_file.py                                                              |
|  @brief    hdf5 reader and writer.                                                               |
|  @ref                                                                                            |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
"""

__all__ = ["savehdf5", "loadhdf5", "hdf5Slicer"]

import h5py
import numpy as np
from .common import os, ensure_exist


def savehdf5(filename: str, data: dict, kwargs: dict = {}):
    ensure_exist(os.path.dirname(os.path.realpath(filename)))
    with h5py.File(filename, "w") as hf:
        for k, v in data.items():
            # hf.create_dataset(k, data=v, compression="gzip", compression_opts=1)
            hf.create_dataset(k, data=v, **kwargs)


def loadhdf5(filename: str):
    with h5py.File(filename, "r") as hf:
        data = {k: np.squeeze(np.array(v)) for k, v in hf.items()}
    return data


class hdf5Slicer:
    def __init__(self, filename: str, keys: list = []):
        self.__filename = filename
        self.__keys = keys
        self.__hf = h5py.File(self.__filename, "r")

    def __del__(self):
        self.__hf.close()

    def set_keys(self, keys: list):
        self.__keys = keys

    def load_slice(self, idx: int | list, keys: list = None):
        """only loads a slice of data into memory"""
        if keys is None:
            keys = self.__keys

        data = {}
        for k in keys:
            data[k] = self.__hf[k][idx]
        return data

    def load_scalar(self, key: str):
        return self.__hf[key][()]
