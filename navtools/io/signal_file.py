"""
|======================================== signal_file.py ==========================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/io/signal_file.py                                                            |
|  @brief    Binary signal file parser.                                                            |
|  @ref                                                                                            |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
"""

__all__ = ["SignalFile"]

import numpy as np
from .common import ensure_exist


class SignalFile:
    # === init ===
    # SignalFile constructor
    def __init__(self, filename: str, dtype: str, is_complex: bool):
        ensure_exist(filename)
        self.fid_ = open(filename, "rb+")
        self.dtype_ = np.dtype(dtype)
        self.dsize_ = self.dtype_.itemsize
        self.is_complex_ = is_complex

    # === del ===
    # signal file destructor
    def __del__(self):
        self.fid_.close()

    # === fread ===
    # read data from file
    def fread(self, n_samp: int):
        if self.is_complex_:
            stream = np.fromfile(self.fid_, self.dtype_, 2 * n_samp)
            stream = stream[::2] + 1j * stream[1::2]
            # stream = stream.astype(np.float32).view(np.complex64)
        else:
            stream = np.fromfile(self.fid_, self.dtype_, n_samp)
        return stream

    # === fseek ===
    # seek from beginning of file
    def fseek(self, samp, offset):
        if self.is_complex_:
            self.fid_.seek(2 * self.dsize_ * (samp + offset), 0)
        else:
            self.fid_.seek(samp + offset, 0)

    # === ftell ===
    # retrieve sample from beginning of file
    def ftell(self):
        if self.is_complex_:
            loc = int(self.fid_.tell() / 2 / self.dsize_)
        else:
            loc = int(self.fid_.tell() / self.dsize_)
        return loc


# class SignalFile:  # TODO: correct handling of real-valued data
#     def __init__(self, file_path, dtype: str, is_complex: bool):
#         self._fid = open(file_path, "rb+")
#         self._dtype = np.dtype(dtype)
#         self._byte_depth = self._dtype.itemsize
#         self._is_complex = is_complex

#         MIN_NUMPY_COMPLEX_BYTE_DEPTH = 8
#         if is_complex and self._byte_depth < MIN_NUMPY_COMPLEX_BYTE_DEPTH:
#             self._is_complex_with_invalid_dtype = True
#             self._sample_multiplier = 2
#         else:
#             self._is_complex_with_invalid_dtype = False
#             self._sample_multiplier = 1

#     def __del__(self):
#         self._fid.close()

#     @property
#     def sample_location(self) -> int:
#         byte_location = self._fid.tell()
#         sample_location = int(
#             byte_location / (self._byte_depth * self._sample_multiplier)
#         )
#         return sample_location

#     @property
#     def fid(self):
#         return self._fid

#     def fseek(self, sample_location: int):
#         bytes_per_sample = self._byte_depth * self._sample_multiplier
#         byte_location = int(sample_location * bytes_per_sample)
#         self._fid.seek(byte_location)

#     def fread(self, nsamples: int) -> np.array:
#         nsamples = int(nsamples) * self._sample_multiplier
#         samples = np.fromfile(
#             file=self._fid,
#             dtype=self._dtype,
#             count=nsamples,
#         )

#         if self._is_complex_with_invalid_dtype:
#             samples = samples.astype(np.float32).view(np.complex64)

#         if not self._is_complex:
#             samples = samples.astype(np.complex64)

#         return samples
