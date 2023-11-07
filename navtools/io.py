import numpy as np


class SignalFile:  # TODO: correct handling of real-valued data
    def __init__(self, file_path, dtype: str, is_complex: bool):
        self._fid = open(file_path, "rb+")
        self._dtype = np.dtype(dtype)
        self._byte_depth = self._dtype.itemsize

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

    def fread(self, num_samples: int) -> np.array:
        num_samples = num_samples * self._sample_multiplier
        samples = np.fromfile(
            file=self._fid,
            dtype=self._dtype,
            count=num_samples,
        )

        if self._is_complex_with_invalid_dtype:
            samples = samples.astype(np.float32).view(np.complex64)

        return samples


# def parse_bag_topics(bag_file_path: str, topics: list):
#     bag_file_path = pl.Path(bag_file_path)
#     b = bagreader(bag_file_path)
#     output_files = []
#     DESCRIPTION = "Parsing  %s ... " % (bag_file_path.name)

#     for topic in tqdm(topics, desc=DESCRIPTION):
#         topic_data = b.message_by_topic(topic)
#         output_files.append(topic_data)


# def get_parsed_topic(bag_dir_file_path: str, topic: str):
#     modified_topic = topic.replace("/", "", 1).replace("/", "-")
#     csv_file_path = pl.Path(bag_dir_file_path) / modified_topic

#     topic_data = pd.read_csv(csv_file_path.with_suffix(".csv"))

#     return topic_data
