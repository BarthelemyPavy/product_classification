"""Classifcal implementation of multi-layer perceptrons"""
from dataclasses import dataclass

import torch
from torch import nn


class LinearBlock(nn.Module):
    """Linear operations with reguralarization: it contains three elements: a BatchNorm1d, Dropout,

    and a fully connected layer

    Attributes
        in_dim: shape[1] of the input tensor, note that shape[0] is the batch size
        out_dim : output dimension

    """

    def __init__(self, in_dim: int, drop_factor: float, out_dim: int) -> None:
        """Initialize LinearBlock object

        Args:
            in_dim: see in_dim attribute doc
            drop_factor: dropout percentage
            out_dim: see out_dim attribute doc

        """
        super().__init__()
        self.in_dim = in_dim
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Dropout(drop_factor),
            nn.Linear(in_dim, out_dim),
        )

    # pylint: disable=arguments-differ
    def forward(self, in_batch: torch.Tensor) -> torch.Tensor:
        """Forward step of linear block"""
        out = self.block(in_batch)
        return out


@dataclass
class LinearParameters:
    """Parameter of a single linear block

    Attribute
        out_dim: output dimension
        drop_factor: dropout percentage

    """

    out_dim: int
    drop_factor: float


class MultiLayerPerceptron(nn.Module):
    """Set of linear blocks with reLU activations. This is often used at the end of a network when

    unstructured information, such as image and text, is already encoded.
    """

    def __init__(
        self,
        in_dim: int,
        block_params: list[LinearParameters],
    ) -> None:
        """Initialize MultiLayerPerceptron object

        Args:
            in_dim: input dimension of the mlp
            block_params: Parameters of each block of mlp. see LinearParameters doc.

        Raises
            RuntimeError: Raised if the network doesn't contains at least a block

        """
        super().__init__()

        if len(block_params) < 2:
            raise RuntimeError("The network must contain at least a block")

        self.model = nn.Sequential()

        module_in = in_dim
        for block_id, params in enumerate(block_params[:-1]):
            self.model.add_module(
                f"hidden_{block_id}",
                LinearBlock(module_in, params.drop_factor, params.out_dim),
            )
            self.model.add_module(f"relu_{block_id}", nn.ReLU(inplace=True))
            module_in = params.out_dim
        self.model.add_module(
            "final_block",
            LinearBlock(module_in, block_params[-1].drop_factor, block_params[-1].out_dim),
        )

    # pylint: disable=arguments-differ
    def forward(self, in_batch: torch.Tensor) -> torch.Tensor:
        """Forward step of mlp

        Args:
            in_batch:

        Returns
            torch.Tensor:

        """
        out = self.model(in_batch)
        return out
