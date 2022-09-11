"""Mimic tf.keras.Model for PyTorch

https://www.tensorflow.org/api_docs/python/tf/keras/Model#class_model_2
"""
from typing import Iterator, Optional, Union
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import numpy.typing as npt

from product_classification.learner.metrics import IMetric

PathOrStr = Union[Path, str]


class Learner:
    """A learner contains a model to be trained, an optimizer that is used at the backpropagation step,

    and a loss function. Metrics can also be optionally defined to log evolution trend on.

    Attributes
        _model: model to be trained
        _optimizer: optimization function. The client should construct the
            optimizer by using functools.partial if they want to use custom
            optimization hyper-parameters
        _loss_func: loss function, e.g. cross-entropy loss for multiclass classification
            problems
        _metrics: set of metrics that can be used to monitor training
            and evaluation

    """

    _model: nn.Module
    _optimizer: Optional[optim.Optimizer]
    _loss_func: nn.Module
    _step_eval: Optional[int]
    _metrics: Optional[list[IMetric]]

    def __init__(self, model: nn.Module) -> None:
        """Initialize Learner class

        Args:
            model: architecture of the model to be trained

        """
        self._model = model
        self._optimizer = None
        self._loss_func = None
        self._step_eval = None
        self._metrics = None

    def compile(
        self,
        optimizer: Optional[optim.Optimizer],
        loss_func: nn.Module,
        step_eval: int,
        metrics: Optional[list[IMetric]] = None,
    ) -> None:
        """Set optimizer, loss_func, and optionally metrics

        Args:
            optimizer: see _optimizer attribute
            loss_func: see _loss_func attribute
            metrics: see _metrics attribute, default to None

        """
        if optimizer is not None:
            self._optimizer = optimizer(self._model.parameters())
        self._loss_func = loss_func
        self._step_eval = step_eval
        self._metrics = metrics

    def fit(
        self,
        nb_epochs: int,
        device: torch.cuda.device,
        dataloader: Iterator[torch.Tensor],
        val_dataloader: Optional[Iterator[torch.Tensor]] = None,
    ) -> None:
        """Train the the a text classifier

        Args:
            nb_epochs: number of epochs
            device: device on which the network is trained
            dataloader: torchtext.data.DataLoader that contains two fields:
                text and label. This restrict this learner to text classification models. We should
                remove this restriction in the future (idea use a Facade).
            val_dataloader: torchtext.data.DataLoader to be used
                for model validation

        """
        if self._optimizer is None or self._loss_func is None:
            raise ValueError("learner has not been compiled")

        self._model.to(device)
        self._loss_func.to(device)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode="min", patience=5, verbose=True, factor=0.5
        )
        for epoch in range(1, nb_epochs + 1):
            self._model.train()

            if self._metrics is not None:
                for metric in self._metrics:
                    metric.reset_states()

            epoch_loss = 0
            loop = tqdm(enumerate(dataloader), total=len(dataloader))
            loop.set_description(f"Training Epoch [{epoch}/{nb_epochs}]")
            for batch_idx, batch in loop:
                self._optimizer.zero_grad()
                in_text, category_ohe, target = batch.text.to(device), batch.category.to(device), batch.label.to(device)

                predictions = self._model((in_text, category_ohe))
                # pylint: disable=not-callable
                loss = self._loss_func(predictions, target.float())

                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()
                if self._metrics is not None:
                    for metric in self._metrics:
                        metric.update_states(predictions, target)
                loop.set_postfix(loss=epoch_loss / (batch_idx + 1))
                # Evaluation during epoch
                if (batch_idx + 1) % self._step_eval == 0 and val_dataloader:
                    self.evaluate(val_dataloader, device, scheduler)

            print(f"Training at epoch {epoch}:")
            if self._metrics is not None:
                for metric in self._metrics:
                    print(f"\t{metric.name()} = {metric.result():.3f}")
            print(f"\tloss = {epoch_loss / len(dataloader):.3f}")

            if val_dataloader:
                self.evaluate(val_dataloader, device, scheduler)

    def evaluate(self, dataloader: Iterator[torch.Tensor], device: torch.cuda.device, scheduler) -> None:
        """Evaluate the classifier on a dataset that is dijoitn from training dataset, i.e. validation dataset

        Args:
            dataloader: see dataloader in fit
            device: device on which the network is trained

        Raises
            ValueError: if the learner was not compiled

        """
        if self._loss_func is None:
            raise ValueError("learner has not been compiled")

        self._model.to(device)
        self._loss_func.to(device)

        self._model.eval()

        total_loss = 0
        if self._metrics is not None:
            for metric in self._metrics:
                metric.reset_states()

        den = len(dataloader)
        with torch.no_grad():
            loop = tqdm(enumerate(dataloader), total=len(dataloader))
            loop.set_description("Validation")
            for batch_idx, batch in loop:
                in_text, category_ohe, target = batch.text.to(device), batch.category.to(device), batch.label.to(device)

                try:
                    predictions = self._model((in_text, category_ohe))
                except RuntimeError:
                    den -= 1
                    continue
                # pylint: disable=not-callable
                loss = self._loss_func(predictions, target.float())
                total_loss += loss.item()
                if self._metrics is not None:
                    for metric in self._metrics:
                        metric.update_states(predictions, target)
                loop.set_postfix(loss=total_loss / (batch_idx + 1))
        print("Validation:")
        if self._metrics is not None:
            for metric in self._metrics:
                print(f"\t{metric.name()} = {metric.result():.3f}")
        print(f"\tloss = {total_loss / den:.3f}")
        print("".join(["*" * 40]))
        scheduler.step(total_loss / den)

    def predict(
        self, dataloader: Iterator[torch.Tensor], device: torch.cuda.device
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
        """Predict classes of a set of texts

        Args:
            dataloader: iterator that parse test data
            device: device on which prediction is done

        Returns
            tuple[np.ndarray, np.ndarray]: predicted labels and probabilites associated with each
                class

        """
        self._model.to(device)
        self._model.eval()

        probs = np.array([])
        preds = np.array([])
        with torch.no_grad():
            for batch in dataloader:
                try:
                    in_text, category_ohe = batch.text.to(device), batch.category.to(device)

                    predictions = self._model((in_text, category_ohe))

                    if probs.size == 0:
                        probs = F.sigmoid(predictions).cpu().numpy()
                        preds = torch.where(predictions >= 0.5, 1, 0).cpu().numpy()
                    else:
                        # np.append has no return type
                        probs = np.append(probs, F.sigmoid(predictions).cpu().numpy(), axis=0)  # type: ignore
                        preds = np.append(preds, torch.where(predictions >= 0.5, 1, 0).cpu().numpy(), axis=0)  # type: ignore
                except RuntimeError as err:
                    raise err
        return preds, probs

    def summary(self) -> None:
        """Print a summary of the model used for training"""

        def print_layer_params(child: nn.Module, tabs: int = 0) -> None:
            """Print layer parameters

            Args:
                child: layer to print parameters
                tabs: Format tabulation. Defaults to 0.

            """
            nb_layer_params = sum(p.numel() for p in child.parameters())
            print(f"{''.join(['    ']*tabs)}{type(child).__name__}->{nb_layer_params}")
            for nephew in child.children():
                print_layer_params(nephew, tabs=tabs + 1)

        print("Layer -> Nb.Parameters")
        print("".join(["*"] * 40))
        for child in self._model.children():
            print_layer_params(child)
        print("".join(["*"] * 40))

        total_nb_params = sum(p.numel() for p in self._model.parameters())
        trainable_nb_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        print(f"Total number of parameters -> {total_nb_params}")
        print(f"Total number of trainable parameters -> {trainable_nb_params}")

    def freeze_to(self, first_trainable: int) -> None:
        """Set parameters to not trainable up to first_trainable group of layers excluded

        Args:
            first_trainable: the first group of trainable layers.

        """
        for child in list(self._model.children())[:first_trainable]:
            print(child)
            for param in child.parameters():
                param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all the layers, i.e. make all the parameters trainable"""
        for child in self._model.children():
            for param in child.parameters():
                param.requires_grad = True

    def save(self, fpath: PathOrStr) -> None:
        """Save model weights to a binary file

        Args:
            fpath: path to the destination file

        """
        torch.save(self._model.state_dict(), fpath)

    def load(self, fpath: PathOrStr) -> None:
        """Load pre-trained weights into _weights

        Args:
            fpath: path to file that contains pretrained model weights

        Raises
            ModelLoadError: if model architecture and loaded one do not match

        """
        try:
            self._model.load_state_dict(
                torch.load(
                    fpath,
                    map_location=None if torch.cuda.is_available() else "cpu",
                )
            )
        except RuntimeError as err:
            raise err
