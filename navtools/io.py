import numpy as np


class SignalFile:  # TODO: correct handling of real-valued data
    def __init__(self, file_path, dtype: str, is_complex: bool):
        self._fid = open(file_path, "rb+")
        self._dtype = np.dtype(dtype)
        self._byte_depth = self._dtype.itemsize
        self._is_complex = is_complex

        MIN_NUMPY_COMPLEX_BYTE_DEPTH = 8
        if is_complex and self._byte_depth < MIN_NUMPY_COMPLEX_BYTE_DEPTH:
            self._is_complex_with_invalid_dtype = True
            self._sample_multiplier = 2
        else:
            self._is_complex_with_invalid_dtype = False
            self._sample_multiplier = 1

    def __del__(self):
        self._fid.close()

    @property
    def sample_location(self) -> int:
        byte_location = self._fid.tell()
        sample_location = int(
            byte_location / (self._byte_depth * self._sample_multiplier)
        )
        return sample_location

    @property
    def fid(self):
        return self._fid

    def fseek(self, sample_location: int):
        bytes_per_sample = self._byte_depth * self._sample_multiplier
        byte_location = int(sample_location * bytes_per_sample)
        self._fid.seek(byte_location)

    def fread(self, nsamples: int) -> np.array:
        nsamples = int(nsamples) * self._sample_multiplier
        samples = np.fromfile(
            file=self._fid,
            dtype=self._dtype,
            count=nsamples,
        )

        if self._is_complex_with_invalid_dtype:
            samples = samples.astype(np.float32).view(np.complex64)

        if not self._is_complex:
            samples = samples.astype(np.complex64)

        return samples
