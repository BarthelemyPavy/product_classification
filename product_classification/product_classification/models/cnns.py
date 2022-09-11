"""CNN-based text classifiers"""
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from product_classification.models.mlp import MultiLayerPerceptron, LinearParameters


class BaseCNN(nn.Module):
    """
    Standard one-layer CNN for sentence cls (Y. Kim. CNN for Sentence Classification, 2014)

    Args:
        voc_size (int): number of words in the vocabulary
        embedding_size (int): dimension of the embedding layer
        kernels (Sequence[int]): kernels of convolutional layers, the length of this list is the
            number of convolutional layers
        nb_filters (int): number of filters per convolutional layers
        output_dim (int): number of output classes
        drop_pct (float): percentage of connections that are turned off in the dropout layer
        pad_idx (Optional[int]): token to be used for padding, use 0s if None

    Attributes:
        embedding (nn.Embedding): embedding layer that is used to obtain dense vector representation
        conv_layers (nn.ModuleList): set of Conv1d layers
        mlp (MultiLayerPerceptron): highest part of the network used to classify the encoded text
            representation
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        voc_size: int,
        embedding_size: int,
        kernels: list[int],
        categorical_dim: int,
        nb_filters: int,
        output_dim: int,
        drop_pct: float,
        pad_idx: Optional[int] = None,
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=embedding_size, padding_idx=pad_idx)
        self.conv_layes = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_size, out_channels=nb_filters, kernel_size=fs) for fs in kernels]
        )

        self.mlp = MultiLayerPerceptron(
            len(kernels) * nb_filters + categorical_dim,
            [
                LinearParameters(out_dim=nb_filters, drop_factor=drop_pct),
                LinearParameters(out_dim=output_dim, drop_factor=drop_pct / 2),
            ],
        )

    # pylint: disable=arguments-differ
    def forward(self, in_batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Implementation of CNN forward step

        Args:
            in_batch (torch.Tensor): 2D tensor of dimension (b, s) where b is the batch size and
                s is the length of padded sentences (note, sentences could be padded so that all
                the samples in the batch have the same dimension)

        Returns:
            torch.Tensor: 2D tensor of dimension (b, output_dim), each row contains the confidence
                scores of samples belonging to a given class. Note, use a sigmoid if you want this
                confidence to be expressed as a probability.
        """
        in_text, category_ohe = in_batch
        category_ohe = category_ohe.permute(1, 0)
        embedded = self.embedding(in_text)
        embedded = embedded.permute(0, 2, 1)  # permute to have batch_size, emb_dim, sentence_len

        conved = [F.relu(conv(embedded)) for conv in self.conv_layes]

        # stride equal to sentence length to take max on the sentene representation
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        pooled.append(category_ohe)
        concat = torch.cat(pooled, dim=1)

        out = self.mlp(concat)

        return out
