"""Split dataset into train, test, validation sets"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Union
from dataclasses import dataclass
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from product_classification import logger
from product_classification.utils import log_attribute_per_dataset


@dataclass
class Datasets:
    """Split dataset for supervised Machine Learning into training, validation and test

    Attributes:\
        training: training dataset
        test: test dataset
        validation: validation dataset
    """

    training: pd.DataFrame
    test: pd.DataFrame
    validation: pd.DataFrame

    def __iter__(self) -> Iterator[tuple[str, pd.DataFrame]]:
        """Iterate over Dataset object

        Yield:\
            Iterator[tuple[str, pd.DataFrame]]: Dataframe contained in Dataset object
        """
        for attr, value in self.__dict__.items():
            yield attr, value

    def asdict(self) -> dict[str, pd.DataFrame]:
        """Convert to dictionary

        Returns:\
            dict[str, pd.DataFrame]: Dataset attribute converted to dict
        """
        return vars(self)


class Split(ABC):
    """Abstract Class for train test val split

    Attributes:
        _val_size: Rate of validation from test_val dataset
    """

    _val_size: float = 0.5
    _multilabel_binarizer: MultiLabelBinarizer
    _pos_weight: list[float]

    def __init__(self, min_categories_threshold: int, max_categories_threshold: int) -> None:
        """Class constructor

        Args:
            min_categories_threshold: Minimum of sample to have to use a category
            max_categories_threshold: Threshold use to down sampling categories with too much examples
        """
        self._min_categories_threshold = min_categories_threshold
        self._max_categories_threshold = max_categories_threshold

    @property
    def multilabel_binarizer(self) -> MultiLabelBinarizer:
        return self._multilabel_binarizer

    def _filter_categories(self, dataset: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Keep only categories with nb_examples >= min_categories_threshold.

        Args:
            dataset: Dataset to apply filter
            column_name: Name of column containing categories

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        logger.info("Apply filter to remove categories with few examples")
        nb_product_category = (
            dataset[["id_product", column_name]]
            .groupby(column_name)
            .count()
            .reset_index()
            .rename(columns={"id_product": "product_number"})
        )
        category_to_remove = nb_product_category[nb_product_category.product_number < self._min_categories_threshold][
            column_name
        ].tolist()

        dataset = dataset[~dataset[column_name].isin(category_to_remove)]
        logger.info(f"Removed categories: {' ,'.join(category_to_remove)}")
        logger.info(f"Dataset shape: {dataset.shape}")
        return dataset

    def _downsampling_categories(self, dataset: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Downsampling categories with too much examples

        Args:
            dataset: Dataset to downsample
            column_name: Name of column containing categories

        Returns:
            pd.DataFrame: Downsampled dataframe
        """
        logger.info("Downsampling categories with too much examples")
        dataset = dataset.groupby(column_name).head(self._max_categories_threshold).reset_index(drop=True)
        logger.info(f"Dataset shape: {dataset.shape}")
        return dataset

    def _multilabel_transformation(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Join categories for each product

        Args:
            dataset (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        logger.info("Join categories for each product")
        merged_categories = (
            dataset[["id_product", "category"]]
            .groupby("id_product")
            .category.apply(list)
            .reset_index()
            .rename(columns={"category": "categories"})
        )
        return (
            dataset.drop(columns="category")
            .drop_duplicates(subset=["id_product"])
            .merge(merged_categories, on="id_product")
        )

    @abstractmethod
    def _test_val_split(
        self, dataset: pd.DataFrame, split_size: Union[float, int], **kwargs: int
    ) -> Tuple[list[int], list[int], list[int]]:
        """Split questions into test and val ids

        Args:
            dataset: DataFrame with id_product and category columns
            split_size: Size of first split
            random_state: For train_test_split from sklearn random_state for split reproducibility (optional)
            stratify: For train_test_split from sklearn target on wich a stratified split will applied (optional)

        Returns:
            Tuple[list[int], list[int], list[int]]: Tuple containing training, test and validation ids
        """

    # For **kwargs typing https://peps.python.org/pep-0484/#arbitrary-argument-lists-and-default-argument-values
    @abstractmethod
    def execute(self, dataset: pd.DataFrame, split_size: float, **kwargs: float) -> Datasets:
        """Main function of the class

        Args:
            dataset: Dataset to split
            split_size: Rate of split for test_val datasets
            random_state: random_state for split reproducibility

        Returns:
            Datasets: Object containing training, test and validation dataset
        """


class IterativeSplit(Split):
    """Iterative split using skmultilearn and implementing specific split for multilabel problem:

    - http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf
    - http://proceedings.mlr.press/v74/szyma%C5%84ski17a/szyma%C5%84ski17a.pdf
    """

    def _test_val_split(
        self, dataset: pd.DataFrame, split_size: float, **kwargs: int
    ) -> Tuple[list[int], list[int], list[int]]:
        """See parent class docstring"""

        train, _, X_others, y_others = iterative_train_test_split(
            np.asmatrix(dataset[["id_product"]]),
            np.asmatrix(dataset.drop(columns=["id_product"])),
            test_size=split_size,
        )
        train_ids = pd.DataFrame(train, columns=["id_product"])
        test, _, validation, _ = iterative_train_test_split(X_others, y_others, test_size=self._val_size)
        test_ids = pd.DataFrame(test, columns=["id_product"])
        validation_ids = pd.DataFrame(validation, columns=["id_product"])

        return (train_ids.id_product.tolist(), test_ids.id_product.tolist(), validation_ids.id_product.tolist())

    def execute(self, dataset: pd.DataFrame, split_size: float) -> Datasets:  # type: ignore
        """See parent class docstring"""
        logger.info("Split dataset into train, test, val using iterative_train_test_split")
        mlb = MultiLabelBinarizer()

        dataset_multilabel = self._multilabel_transformation(dataset=dataset)
        dataset_multilabel["categories_str"] = dataset_multilabel.categories.apply(str)
        dataset_multilabel = self._filter_categories(dataset=dataset_multilabel, column_name="categories_str")
        dataset_multilabel = self._downsampling_categories(dataset=dataset_multilabel, column_name="categories_str")
        # dataset_multilabel.drop(columns="categories_str", inplace=True)

        labels = mlb.fit_transform(dataset_multilabel.categories)
        dataset_to_split = pd.concat([dataset_multilabel[["id_product"]], pd.DataFrame(labels)], axis=1)
        dataset_to_split.columns = ["id_product"] + list(mlb.classes_)

        train_ids, test_ids, validation_ids = self._test_val_split(dataset=dataset_to_split, split_size=split_size)
        dataset_multilabel_final = dataset_multilabel.drop(columns=["categories"]).merge(
            dataset_to_split, on="id_product"
        )
        datasets = Datasets(
            training=dataset_multilabel_final[dataset_multilabel_final.id_product.isin(train_ids)],
            test=dataset_multilabel_final[dataset_multilabel_final.id_product.isin(test_ids)],
            validation=dataset_multilabel_final[dataset_multilabel_final.id_product.isin(validation_ids)],
        )
        for fname, dataframe in datasets:
            log_attribute_per_dataset(df_data=dataframe, attribute="categories_str", logger=logger, desc=fname)
            setattr(datasets, fname, dataframe.drop(columns=["categories_str"]))
        self._multilabel_binarizer = mlb
        return datasets


class SimpleSplit(Split):
    """Split using sklearn train_test_split with stratify=category"""
    
    @property
    def pos_weight(self) -> list[float]:
        return self._pos_weight
    
    def _get_pos_weight_loss(training_df: pd.DataFrame) -> list[float]:
        """[summary]

        Args:
            training_df: [description]

        Returns:
            list[float]: [description]
        """
        label_count = training_df.groupby("category").id_product.count().reset_index().rename(columns={"id_product": "label_count"})
        label_count["total_label_count"] = training_df.id_product.count()
        label_count["others_label_count"] = label_count["total_label_count"] - label_count["label_count"]
        label_count["label_weight"] = label_count["others_label_count"] / label_count["label_count"]
        return label_count.label_weight.tolist()

    def _test_val_split(
        self, dataset: pd.DataFrame, split_size: float, **kwargs: int
    ) -> Tuple[list[int], list[int], list[int]]:
        """See parent class docstring"""

        # Split dataset in train, test_val dataset
        training, test_val = train_test_split(
            dataset[["id_product", "category"]], test_size=split_size, stratify=dataset["category"], **kwargs
        )
        # Split test_eval dataset into test and validation
        test, validation = train_test_split(
            test_val[["id_product"]], test_size=self._val_size, stratify=test_val["category"], **kwargs
        )

        return (training.id_product.tolist(), test.id_product.tolist(), validation.id_product.tolist())

    def execute(self, dataset: pd.DataFrame, split_size: float, random_state: int) -> Datasets:  # type: ignore
        """See parent class docstring"""
        logger.info("Split dataset into train, test, val using train_test_split with stratification")
        dataset = self._filter_categories(dataset=dataset, column_name="category")
        dataset = self._downsampling_categories(dataset=dataset, column_name="category")

        train_ids, test_ids, validation_ids = self._test_val_split(
            dataset=dataset[["id_product", "category"]], split_size=split_size, random_state=random_state
        )

        datasets = Datasets(
            training=dataset[dataset.id_product.isin(train_ids)],
            test=dataset[dataset.id_product.isin(test_ids)],
            validation=dataset[dataset.id_product.isin(validation_ids)],
        )
        self._pos_weight = self._get_pos_weight_loss(trainin_df=datasets.training)

        for fname, dataframe in datasets:
            log_attribute_per_dataset(df_data=dataframe, attribute="category", logger=logger, desc=fname)
            setattr(datasets, fname, self._multilabel_transformation(dataset=dataframe))

        mlb = MultiLabelBinarizer()
        # Transform target to one hot encoder for training set
        labels = mlb.fit_transform(datasets.training.categories)
        dataset_one_hot = pd.concat([datasets.training["id_product"], pd.DataFrame(labels)], axis=1)
        dataset_one_hot.columns = ["id_product"] + list(mlb.classes_)
        datasets.training = datasets.training.merge(dataset_one_hot, on="id_product").drop(columns=["categories"])
        # Transform target to one hot encoder for test set
        labels = mlb.transform(datasets.test.categories)
        dataset_one_hot = pd.concat([datasets.test["id_product"], pd.DataFrame(labels)], axis=1)
        dataset_one_hot.columns = ["id_product"] + list(mlb.classes_)
        datasets.test = datasets.test.merge(dataset_one_hot, on="id_product").drop(columns=["categories"])
        # Transform target to one hot encoder for validation set
        labels = mlb.transform(datasets.validation.categories)
        dataset_one_hot = pd.concat([datasets.validation["id_product"], pd.DataFrame(labels)], axis=1)
        dataset_one_hot.columns = ["id_product"] + list(mlb.classes_)
        datasets.validation = datasets.validation.merge(dataset_one_hot, on="id_product").drop(columns=["categories"])
        self._multilabel_binarizer = mlb
        return datasets
