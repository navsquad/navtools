# %%
import re
import numpy as np
import pandas as pd
import pathlib as plib
from numbers import Number
from typing import Tuple
from bagpy import bagreader
from tqdm import tqdm


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
