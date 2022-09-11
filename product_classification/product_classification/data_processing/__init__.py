"""Module containing all data preprocessing"""
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from torchtext import data
from product_classification import logger

PRETRAINED_EMB = {
    "charngram.100d": 100,
    "fasttext.en.300d": 300,
    "fasttext.simple.300d": 300,
    "glove.42B.300d": 300,
    "glove.840B.300d": 300,
    "glove.twitter.27B.25d": 25,
    "glove.twitter.27B.50d": 50,
    "glove.twitter.27B.100d": 100,
    "glove.twitter.27B.200d": 200,
    "glove.6B.50d": 50,
    "glove.6B.100d": 100,
    "glove.6B.200d": 200,
    "glove.6B.300d": 300,
}


@dataclass
class TorchFields:
    """
    Torch fields that are used to represent text and label in a classification problem

    Attributes:
        text (data.Field): text sequence field
        label (data.LabelField): label field
    """

    text: data.Field
    categorical: data.RawField
    label: data.LabelField


@dataclass
class Dataset:
    """
    Split dataset for supervised Machine Learning into training, validation and test

    Attributes:
        training (pd.DataFrame): channels used to train categorization
        validation (pd.DataFrame): channels used to validate categorization
        test (pd.DataFrame): channels used to test categorization
    """

    training: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def __iter__(self) -> Tuple[str, pd.DataFrame]:
        for attr, value in self.__dict__.items():
            yield attr, value

    def asdict(self) -> dict[str, pd.DataFrame]:
        """convert to dictionary"""
        return vars(self)


@dataclass
class TorchDatasets:
    """
    Datasets of PyTorch tensors

    Attributes
        training (data.Dataset): dataset used for training
        validation (data.Dataset): dataset used to tune hyper-parameters
        test (data.Dataset): dataset used to generate performance report
    """

    training: data.Dataset
    validation: data.Dataset
    test: data.Dataset

    def astuple(self) -> Tuple[data.Iterator]:
        """Convert dataclass to tuple"""
        return (self.training, self.validation, self.test)


class DataFrameDataset(data.Dataset):
    """
    Create a torch text dataset from pandas dataframes

    Args:
        input_df (pd.DataFrame): the dataframe must contain only two attributes,
            see __dataset_attributes__
        f_text (data.Field): Field to be used for text, it specifies how to tokenize phrases
        f_label (data.Field): Field to be used for label

    Raises:
        BadDataset: if input dataframe attributes are not __dataset_attributes__
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        f_text: data.Field,
        f_category: data.Field,
        f_label: data.Field,
        txt_col: str,
        cat_cols: list[str],
        lbl_cols: list[str],
        **kwargs: int,
    ) -> None:
        fields = [("text", f_text), ("category", f_category), ("label", f_label)]
        logger.info(f"fields: {fields}")
        examples: list[data.Example] = []

        is_test = False if lbl_cols[0] in input_df.columns else True
        n_labels = len(lbl_cols)

        for idx, row in input_df.iterrows():
            categories = [str(row[c]) for c in cat_cols]
            if not is_test:
                labels = [row[l] for l in lbl_cols]
            else:
                labels = [0.0] * n_labels

            text = str(row[txt_col])
            examples.append(data.Example.fromlist([text, categories, labels], fields))

        super().__init__(examples, fields, **kwargs)

    # pylint: disable=arguments-differ
    @classmethod
    def splits(
        cls,
        f_text: data.Field,
        f_category: data.Field,
        f_label: data.Field,
        datasets: dict[str, pd.DataFrame],
        txt_col: str,
        cat_cols: list[str],
        lbl_cols: list[str],
        **kwargs: int,
    ) -> Tuple[data.Dataset]:
        """
        Create Dataset objects for multiple splits of a dataset.

        Args:
            f_text (data.Field): datafield that contains the text
            f_label (data.Field): datafield that contains the label associated with the text
            datasets (Mapping[str, pd.DataFrame]): dictionary that maps names to pd.DataFrame(s);
                the valid keys are 'training', 'validation', 'test', see input_df for details about
                the dataframe format
            kwargs: see parent method

        Returns:
            Tuple[data.Dataset]: tuple of training, validation and test torch datasets
        """
        train_data, val_data, test_data = (None, None, None)

        if "training" in datasets:
            train_data = cls(
                datasets["training"].copy(), f_text, f_category, f_label, txt_col, cat_cols, lbl_cols, **kwargs
            )
        if "validation" in datasets:
            val_data = cls(
                datasets["validation"].copy(), f_text, f_category, f_label, txt_col, cat_cols, lbl_cols, **kwargs
            )
        if "test" in datasets:
            test_data = cls(datasets["test"].copy(), f_text, f_category, f_label, txt_col, cat_cols, lbl_cols, **kwargs)

        return tuple(item for item in (train_data, val_data, test_data) if item is not None)


@dataclass
class TorchIterators:
    """
    Iterators built from torchtext Datasets

    Attributes
        training (data.Iterator): iterator that iterates on TorchDatasets.training
        validation (data.Iterator): iterator that iterates on TorchDatasets.validation
        test (data.Iterator): iterator that iterates on TorchDatasets.test
    """

    training: data.Iterator
    validation: data.Iterator
    test: data.Iterator
