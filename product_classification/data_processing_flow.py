"""File where data processing Flow is defined"""
from pathlib import Path
from metaflow import FlowSpec, step, Parameter
from product_classification import logger


class DataProcessingFlow(FlowSpec):
    """Flow used to make some data processing and cleaning\n
    In this flow we will:\n
        - Load inputs data.
        - Clean global dataset
        - Split dataset
    """

    config_path = Parameter(
        "config_path",
        help="Config file path for training params",
        default=str(Path(__file__).parent / "conf" / "config.yml"),
    )

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
        """Load training config from yaml file.
        This file contains parameters for train test split generation"""
        import yaml

        with open(self.config_path, "r") as stream:
            self.config = yaml.load(stream, Loader=None)
        logger.info(f"Config parsed: {self.config}")
        self.next(self.get_dataframe)

    @step
    def get_dataframe(self) -> None:
        "Load files and create global dataset"
        from product_classification.data_processing.create_dataset import get_merged_dataframe

        self.dataset = get_merged_dataframe(file_path=self.input_file_path)

        self.next(self.clean_dataset)

    @step
    def clean_dataset(self) -> None:
        """Clean dataset"""
        from product_classification.data_processing.create_dataset import clean_dataset

        self.cleaned_dataset = clean_dataset(dataset=self.dataset)

        self.next(self.split_dataset)

    @step
    def split_dataset(self) -> None:
        """Split dataset in train test val"""
        from product_classification.data_processing.train_test_val_split import SimpleSplit

        logger.info(f"Split dataset in train test val")
        simple_split = SimpleSplit(
            min_categories_threshold=self.config.get("min_category_count"),
            max_categories_threshold=self.config.get("max_category_count"),
        )
        self.datasets = simple_split.execute(
            dataset=self.cleaned_dataset, split_size=self.config.get("split_size"), random_state=self.random_state
        )
        for fname, dataframe in self.datasets:
            setattr(self.datasets, fname, dataframe.drop(columns=["product_description"]))
        self.multilabel_binarizer = simple_split.multilabel_binarizer
        self.pos_weight = simple_split.pos_weight
        self.next(self.end)

    @step
    def end(self) -> None:
        """"""
        self.multilabel_binarizer
        self.dataset
        self.pos_weight
        pass


if __name__ == "__main__":
    DataProcessingFlow()
