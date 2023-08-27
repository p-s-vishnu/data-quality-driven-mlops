from collections import OrderedDict
from dataclasses import dataclass
from typing import List
import tensorflow as tf
import os
from .base import Reader, ReaderConfig, UNKNOWN_LABEL
from ..custom_types import DataType, DataWithInfo

# Configuration class for tabular data
@dataclass(frozen=True)
class TabularFileConfig(ReaderConfig):
    path: str
    header_present: bool
    label_column_number: int
    num_columns: int
    label_values: List[str]
    shuffle_buffer_size: int
    delimiter: chr = ','

    @property
    def data_type(self) -> DataType:
        return DataType.ANY  # Assuming TABULAR is one of the DataType options

# Reader class for tabular data
class TabularFileReader(Reader):
    @staticmethod
    def read_data(config: TabularFileConfig) -> DataWithInfo:
        mapping = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(keys=tf.constant(config.label_values),
                                                            values=tf.constant(list(range(len(config.label_values))))),
            default_value=UNKNOWN_LABEL
        )

        index_label = config.label_column_number - 1

        # Define column names so that label column can be identified
        column_names = [str(i) for i in range(config.num_columns)]
        column_names[index_label] = "label"

        cpu_count = os.cpu_count()
        if not cpu_count:
            cpu_count = 1

        def get_features_and_label(line: OrderedDict):
            label_str = tf.as_string(line["label"][0])
            label = mapping.lookup(label_str)
            features = [tf.cast(value[0], dtype=tf.float32) for key, value in line.items() if key != "label"]
            features_tensor = tf.stack(features)
            return features_tensor, label



        data = tf.data.experimental.make_csv_dataset(
            file_pattern=config.path,
            batch_size=1,
            column_names=column_names,
            select_columns=list(range(config.num_columns)),
            field_delim=config.delimiter,
            header=config.header_present,
            num_epochs=1,
            shuffle=True,
            shuffle_buffer_size=config.shuffle_buffer_size,
            num_parallel_reads=cpu_count,
            sloppy=True
        ).map(get_features_and_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        size = _count_file_rows(config.path, config.header_present)
        return DataWithInfo(data=data, size=size, num_labels=len(config.label_values))

def _count_file_rows(path: str, header_present: bool) -> int:
    cnt = 0
    with open(path) as f:
        for _ in f:
            cnt += 1

    if header_present and cnt > 0:
        cnt -= 1

    return cnt
