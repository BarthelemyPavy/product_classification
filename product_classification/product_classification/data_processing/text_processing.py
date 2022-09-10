"""Use tf-idf to process text and extract relevant information for cold start recommendations"""
from __future__ import annotations
from enum import Enum
import re
from typing import Optional, TypeVar
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer

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
            log_raise(logger=logger, err=ValueError("Bad input tag for lemmatizer"))  # type: ignore

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
