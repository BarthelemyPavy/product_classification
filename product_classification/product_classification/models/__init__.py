"""Implementation of classification models by using PyTorch"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CnnModelArtifacts:
    """
    Paths to CNN model artifacts

    Attributes:
        path_to_model (Path): path to CNN model weights
        path_to_text_field (Path): path to TEXT field that gives access to vocabulary
            and tokenization technique
        path_to_label_field (Path): path to LABEL field that gives accesss to class dictionary
        path_to_config (Path): path to CNN config
    """

    path_to_model: Path
    path_to_text_field: Path
    path_to_label_field: Path
    path_to_config: Path

    def asdict(self) -> dict[str, Path]:
        """convert to dictionary"""
        return vars(self)


S3_CNN_ARTIFACT_NAMES = CnnModelArtifacts(
    path_to_model=Path("models/cnn_text_cls.pth"),
    path_to_text_field=Path("text_field.pt"),
    path_to_label_field=Path("label_field.pt"),
    path_to_config=Path("cnn_config.json"),
)


@dataclass
class LearningRates:
    """
    Learning rate values used to train CNN text classifier

    Attributes:
        init_phase (float): learning rate used in initial phase, lower layers are frozen
        finetuning_phase (float): learning rate used to finetune all the layers
    """

    init_phase: float
    finetuning_phase: float

    def asdict(self) -> dict[str, float]:
        """convert to dictionary"""
        return vars(self)

    def fromdict(self, attr_to_val: dict[str, Any]):  # type: ignore
        """Parse values from a dictionary"""
        self.init_phase = attr_to_val["init_phase"]
        self.finetuning_phase = attr_to_val["finetuning_phase"]
        return self


@dataclass
class Epochs:
    """
    Epochs used to train CNN text classifier

    Attributes:
        init_phase (int): nb of epochs used while fitting in initial phase, lower layers are frozen
        finetuning_phase (int): nb of epochs used while finetuning all layers
    """

    init_phase: int
    finetuning_phase: int

    def asdict(self) -> dict[str, int]:
        """convert to dictionary"""
        return vars(self)

    def fromdict(self, attr_to_val: dict[str, Any]):  # type: ignore
        """Parse values from a dictionary"""
        self.init_phase = attr_to_val["init_phase"]
        self.finetuning_phase = attr_to_val["finetuning_phase"]
        return self


@dataclass
class CnnHyperParameters:
    """
    Hyper-parameters used to define the architecture of a CNN text classifier

    Attributes:
        nb_filters (int): number of filters per CNN layer
        kernels (int): kernel size per CNN layer, note that the length of text to be processed must
            not be less than max(kernels)
        droptout (float): dropout factor
        lrates (LearningRates): learning rate values to be used during different learning phases

    Note:
        the embedding dimension is also an hyper-parameters, but it is tighted with embedding that
        is chosen for pre-training
    """

    # ? embedding dimension is stricly related to the embedding option
    nb_filters: int
    kernels: list[int]
    dropout: float
    lrates: LearningRates
    epochs: Epochs

    def asdict(self):  # type: ignore
        """Convert to dictionary"""
        return {
            "nb_filters": self.nb_filters,
            "kernels": self.kernels,
            "dropout": self.dropout,
            "lrates": self.lrates.asdict(),
            "epochs": self.epochs.asdict(),
        }

    def fromdict(self, attr_to_val: dict[str, Any]):  # type: ignore
        """Parse values from a dictionary"""
        self.nb_filters = attr_to_val["nb_filters"]
        self.kernels = attr_to_val["kernels"]
        self.dropout = attr_to_val["dropout"]
        self.lrates = LearningRates(0.001, 0.001).fromdict(attr_to_val["lrates"])
        self.epochs = Epochs(1, 1).fromdict(attr_to_val["epochs"])
        return self
