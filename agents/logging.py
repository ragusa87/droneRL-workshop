import datetime
import os
import pathlib
from abc import abstractmethod, ABC
from typing import Any

import numpy as np


class Logger(ABC):
    def __init__(self, ) -> None:
        super().__init__()
        self.start = datetime.datetime.now()

    @abstractmethod
    def log_dict(self, global_step: int, values: dict) -> None:
        pass


class TensorBoardLogger(Logger):
    def __init__(self, path: str, name: str) -> None:
        super().__init__()
        import tensorflow as tf
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(os.path.join(path, name))

    def log_histogram(self, global_step: int, name: str, values: Any, bins=1000) -> None:
        import tensorflow as tf
        with self.summary_writer.as_default():
            tf.summary.histogram(name, values, step=global_step, buckets=bins)
        self.summary_writer.flush()

    def log_dict(self, global_step: int, values: dict) -> None:
        import tensorflow as tf
        with self.summary_writer.as_default():
            for name, value in values.items():
                tf.summary.scalar(name, value, step=global_step)
        self.summary_writer.flush()


class NoLogger(Logger):

    def log_dict(self, global_step: int, values: dict) -> None:
        pass
