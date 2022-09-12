"""Use tf-idf to process text and extract relevant information for cold start recommendations"""
from __future__ import annotations
from enum import Enum
from functools import partial
import re
from typing import Optional, TypeVar
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from torchtext import data
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from product_classification.data_processing import (
    PRETRAINED_EMB,
    DataFrameDataset,
    Dataset,
    TorchDatasets,
    TorchFields,
    TorchIterators,
)
from product_classification.models import CnnHyperParameters
from product_classification.utils import log_raise
from product_classification import logger

SklearnType = TypeVar("SklearnType")


class EStemTag(Enum):
    """Tags that can be used to choose between stemmer or lemmatizer

    Attributes:\
        STEMMER: Use SnowballStemmer from nltk
        LEMMATIZER: Use WordNetLemmatizer from nltk

    """

    STEMMER = "stemmer"
    LEMMATIZER = "lemmatizer"


class Lemmatizer(BaseEstimator, TransformerMixin):
    """Define a lemmatizer to use into sklearn Pipeline"""

    def __init__(self, stem: EStemTag) -> None:
        """Init nltk Lemmatizer

        Args:
            stem: use stemmer or lemmatizer
        """
        self.stem = stem
        if stem == EStemTag.LEMMATIZER:
            self.lemma = WordNetLemmatizer()
        elif stem == EStemTag.STEMMER:
            self.lemma = SnowballStemmer("english")
        else:
            log_raise(logger=logger, err=ValueError("Bad input tag for lemmatizer"))

    def fit(self, X: SklearnType, y: Optional[SklearnType] = None) -> Lemmatizer:
        """Nothing happens here

        Args:
            X:
            y: Defaults to None.

        Returns:
            Lemmatizer:
        """
        return self

    def _clean_text(self, text: str) -> str:
        """Get a noisy text in input and remove url and numbers

        Args:
            text: Unclean input text

        Returns:
            str: Clean text
        """
        text = self.remove_url_and_number(text)

        return text

    @staticmethod
    def remove_url_and_number(text: str) -> str:
        """Get a text containing url or number and remove it

        Args:
            text: Unclean text containing url or number

        Returns:
            str: Clean text without url and number
        """
        to_return = text
        if isinstance(text, str):
            text_without_ulr = re.sub(r"https?://[A-Za-z0-9./]+", "", text, flags=re.MULTILINE)
            to_return = re.sub("\d+", "", text_without_ulr, flags=re.MULTILINE)
        return to_return

    def _lemmatize(self, text: str) -> str:
        """Take a string and lemmatize it

        Args:
            text: Input string

        Returns:
            str: Lemmatized string
        """
        if isinstance(self.lemma, WordNetLemmatizer):
            to_return = " ".join([self.lemma.lemmatize(word) for word in word_tokenize(text)])
        elif isinstance(self.lemma, SnowballStemmer):
            to_return = " ".join([self.lemma.stem(word) for word in word_tokenize(text)])
        return to_return

    def transform(self, X: list[str], y: Optional[SklearnType] = None) -> list[str]:
        """Take a list of string in input and lemmatized each of them

        Args:
            X: list of string to process
            y: Defaults to None.

        Returns:
            list[str]: list of string lemmatized
        """
        return [self._lemmatize(self._clean_text(text)) for text in X]


class FitCategoricalData:
    """Fit categorical data to transform it to one hot vectors"""

    _categorical_data: list[str] = ["brand_name", "merchant_name"]

    def execute(self, processed_data: Dataset) -> OneHotEncoder:
        """_summary_

        Args:
            processed_data: _description_

        Returns:
            OneHotEncoder: _description_
        """
        one_hot = OneHotEncoder(drop="first", handle_unknown="ignore")
        one_hot.fit(processed_data.training[self._categorical_data])
        return one_hot


class InitFields:
    """Initialize torch fields to be used for text classification"""

    # pylint: disable=arguments-differ
    def execute(self, one_hot_encoder: OneHotEncoder) -> TorchFields:
        """
        Node executation

        Returns:
            TorchFields: torch fields used to represent text samples and labels
        """
        one_hot_encoding = partial(self._one_hot_encoding, one_hot_encoder=one_hot_encoder)
        categorical_field = data.Field(postprocessing=one_hot_encoding, use_vocab=False)
        logger.info("Categorical field created")
        text_field = data.Field(
            tokenize="spacy", batch_first=True, preprocessing=self._lowercase, tokenizer_language="en_core_web_sm"
        )
        logger.info("Text field created")
        label_field = data.LabelField(use_vocab=False)
        logger.info("Label field created")
        return TorchFields(text=text_field, label=label_field, categorical=categorical_field)

    @staticmethod
    def _one_hot_encoding(category: list[str], vocab: Any, one_hot_encoder: OneHotEncoder) -> list[list[float]]: # type: ignore
        """"""
        return one_hot_encoder.transform(category).toarray()  # type: ignore

    @staticmethod
    def _lowercase(tags: list[str]) -> list[str]:
        """Tags are converted to lowercase during preprocessing"""
        return [tag.lower() for tag in tags]


class CreateDatasets:
    """Node that creates torch datasets"""

    # pylint: disable=arguments-differ
    def execute(
        self,
        processed_data: Dataset,
        torch_fields: TorchFields,
        cnn_hparams: CnnHyperParameters,
        txt_col: str,
        cat_cols: list[str],
        lbl_cols: list[str],
    ) -> TorchDatasets:
        """
        Execute node

        Args:
            processed_data (Dataset): see ProcessData.execute return doc
            torch_fields (TorchFields): see InitFields.execute return doc
            cnn_hparams (TorchFields): used to filter text whose nb tokens whose is less than max
                kernel size

        Returns:
            TorchDatasets: training, validation and test datasets to be used to train, validate and
                test text classifiers with PyTorch
        """

        train_ds, val_ds, test_ds = DataFrameDataset.splits(
            torch_fields.text,
            torch_fields.categorical,
            torch_fields.label,
            processed_data.asdict(),
            txt_col,
            cat_cols,
            lbl_cols,
        )

        test_ds = self._filter_sample(test_ds, max(cnn_hparams.kernels))
        logger.info("Torchtext datasets created from dataframes")
        return TorchDatasets(training=train_ds, validation=val_ds, test=test_ds)

    @staticmethod
    def _filter_sample(dataset: data.Dataset, min_size: int) -> data.Dataset:
        filtered_samples = [sample for sample in dataset.examples if len(sample.text) >= min_size]
        dataset.examples = filtered_samples
        return dataset


class BuildTextVocabulary:
    """Node that builds vocabulary on the training corpus"""

    # pylint: disable=arguments-differ
    # pylint: disable=too-many-arguments
    def execute(
        self,
        torch_datasets: TorchDatasets,
        torch_fields: TorchFields,
        vocab_size: int,
        embedding_name: str,
        vectors_cache: str,
    ) -> TorchFields:
        """
        Execute node

        Args:
            torch_datasets (TorchDatasets): see CreateDatasets.execute return doc
            torch_fields (TorchFields): see InitFields.execute return doc
            vocab_size (int): maximum size of the vocabulary
            embedding_name (str): embedding that must match one of the options in PRETRAINED_EMB
            cache_folder (Path): path to cache folder
            vectors_cache (str): folder where pretrained embeddings are stored

        Returns:
            str: path to file, used for serialization

        Raises:
            ValueError: when embedding_nam is not PRETRAINED_EMB
        """

        if embedding_name not in PRETRAINED_EMB:
            raise ValueError(f"{embedding_name} not allowed")

        torch_fields.text.build_vocab(
            torch_datasets.training,
            max_size=vocab_size,
            vectors=embedding_name,
            unk_init=torch.Tensor.normal_,
            vectors_cache=vectors_cache,
        )

        logger.info(f"Text vocabulary created. Corpus_dim: {len(torch_fields.text.vocab)}")
        return torch_fields


class BuildIterators:
    """Build dataloaders to iterate over training, validation and test datasets"""

    # pylint: disable=arguments-differ
    def execute(self, torch_datasets: TorchDatasets, batch_size: int, device: torch.device) -> TorchIterators:
        """
        Execute node

        Args:
            torch_datasets (TorchDatasets): [description]
            batch_size (int): batch size, the same value is used for training, validation and test
            device (torch.device): device could be CPU or a GPU

        Returns:
            TorchIterators: [description]
        """

        # pylint: disable=unbalanced-tuple-unpacking
        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            torch_datasets.astuple(),
            batch_size=batch_size,
            device=device,
            sort_key=lambda x: len(x.text),
        )
        test_iter.sort = False  # we want to keep the order during the evaluation phase
        test_iter.sort_within_batch = False  # we want to keep the order during the evaluation phase
        logger.info("Torchtext iterators created from datasets")
        return TorchIterators(training=train_iter, validation=val_iter, test=test_iter)
