"""File where data processing Flow is defined"""
import logging
from pathlib import Path
from typing import Union
from metaflow import FlowSpec, step, Parameter
import numpy as np
import numpy.typing as npt
import pandas as pd
from product_classification import logger


class DataProcessingFlow(FlowSpec):
    """Flow used to make some data processing and cleaning\n
    In this flow we will:\n
        - Load inputs data.
        - Clean global dataset
    """

    random_state = Parameter(
        "random_state",
        help="Random state for several application",
        default=42,
    )

    input_file_path = Parameter(
        "input_file_path",
        help="Path to files containing input data",
        default=str(Path(__file__).parents[1] / "data"),
    )

    @step
    def start(self) -> None:
        "Load files and create global dataset"
        from product_classification.data_processing.create_dataset import get_merged_dataframe

        self.dataset = get_merged_dataframe(file_path=self.input_file_path)

        self.next(self.clean_dataset)

    @step
    def clean_dataset(self) -> None:
        """Clean dataset"""
        from product_classification.data_processing.create_dataset import clean_dataset

        self.cleaned_dataset = clean_dataset(dataset=self.dataset)

        self.next(self.end)

    @step
    def end(self) -> None:
        """"""
        pass


if __name__ == "__main__":
    DataProcessingFlow()
