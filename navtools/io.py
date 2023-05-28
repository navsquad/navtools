# %%
import re
import numpy as np
import pandas as pd
import pathlib as plib
from numbers import Number
from typing import Tuple, Union
from bagpy import bagreader
from tqdm import tqdm


class SignalFile:
    def __init__(self, file_path, dtype: str, is_complex: bool):
        """Signal file for processing

        Parameters
        ----------
        file_path
            Path to signal file
        dtype : str
            numpy datatype of file as string
        is_complex : bool
            Indicates whether signal is complex data
        """
        self.fid = open(file_path, "rb+")
        self.dtype = np.dtype(dtype)
        self.offset = 0
        self.byte_depth = self.dtype.itemsize

        MIN_NUMPY_COMPLEX_BYTE_DEPTH = 8
        if is_complex and self.byte_depth < MIN_NUMPY_COMPLEX_BYTE_DEPTH:
            self.is_complex_with_invalid_dtype = True
            self.sample_multiplier = 2
        else:
            self.is_complex_with_invalid_dtype = False
            self.sample_multiplier = 1

    def __del__(self):
        self.fid.close()

    def fseek(self, sample_offset: int):
        """Sets desired offset from current sample location in file. It is used on next call of fread().

        Parameters
        ----------
        sample_offset : int
            Number of samples to skip from current location
        """
        bytes_per_sample = self.byte_depth * self.sample_multiplier
        byte_offset = sample_offset * bytes_per_sample
        self.offset = byte_offset

    def fread(self, num_samples: int) -> np.array:
        """Returns requested number of signal samples from current location in file.

        Parameters
        ----------
        num_samples : int
            Number of samples to return

        Returns
        -------
        np.array
            Signal samples
        """
        num_samples = num_samples * self.sample_multiplier
        samples = np.fromfile(
            file=self.fid,
            dtype=self.dtype,
            count=num_samples,
            offset=self.offset,
        )
        self.offset = 0  # prevents unwanted sample skipping

        if self.is_complex_with_invalid_dtype:
            samples = samples.astype(np.float32).view(np.complex64)

        return samples

    def get_sample_location(self) -> int:
        """Returns location of file pointer after most recent fread() call.

        Returns
        -------
        int
            Location of last read sample in number of samples
        """
        byte_location = self.fid.tell()
        sample_location = int(byte_location / self.sample_multiplier)
        return sample_location

    def get_fid(self):
        return self.fid


def compute_byte_properties(
    file_data_type: str, fsamp: Number
) -> Tuple[Number, Number]:
    bit_depth_str = re.findall(r"\d+", file_data_type)
    bit_depth = list(map(int, bit_depth_str))

    byte_depth = int(bit_depth[-1] / 8)
    bytes_per_sec = fsamp * byte_depth

    return byte_depth, bytes_per_sec


def parse_bag_topics(bag_file_path: str, topics: list):
    b = bagreader(bag_file_path)
    output_files = []
    DESCRIPTION = "Parsing Bag File... "

    for topic in tqdm(topics, desc=DESCRIPTION):
        topic_data = b.message_by_topic(topic)
        output_files.append(topic_data)


def get_parsed_topic(bag_dir_file_path: str, topic: str):
    modified_topic = topic.replace("/", "", 1).replace("/", "-")
    csv_file_path = plib.Path(bag_dir_file_path) / modified_topic

    topic_data = pd.read_csv(csv_file_path.with_suffix(".csv"))

    return topic_data


# %%
