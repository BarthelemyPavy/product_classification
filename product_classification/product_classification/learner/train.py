from functools import partial
from typing import Any, Optional
import torch
import numpy as np
from torch import nn, optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

from product_classification.learner.metrics import HammingLoss, FScore
from product_classification.data_processing import PRETRAINED_EMB, Dataset, TorchFields, TorchIterators
from product_classification.models import CnnHyperParameters
from product_classification.models.cnns import BaseCNN
from product_classification import logger
from product_classification.learner.learner import Learner


class CreateLearner:
    """
    Create CNN learner that is defined by a model architecture, a loss function
    and an optimisation algorithm
    """

    # pylint: disable=arguments-differ
    def execute(
        self,
        cnn_hparams: CnnHyperParameters,
        embedding_name: str,
        torch_fields: TorchFields,
        processed_data: Dataset,
        batch_size: int,
        label_number: int,
        one_hot_encoder: OneHotEncoder,
        pos_weight: Optional[torch.Tensor]=None
    ) -> Learner:
        """
        Execute node

        Args:
            cnn_hparams (CnnHyperParameters): hyper-parameters used to define the CNN architecture
            embedding_name (str): embedding that must match one of the options in PRETRAINED_EMB
            torch_fields (TorchFields): see output of InitFields.execute

        Raises:
            ValueError: if embedding_name not in PRETRAINED_EMB

        Returns:
            Learner: compiled learned obj, see Learner doc for details
        """

        if embedding_name not in PRETRAINED_EMB:
            raise ValueError(f"{embedding_name} is not allowed")

        f_text = torch_fields.text
        cnn_model = BaseCNN(
            voc_size=len(f_text.vocab),
            embedding_size=PRETRAINED_EMB[embedding_name],
            kernels=cnn_hparams.kernels,
            # -2 Because 1 column dropped for each categorical feature
            categorical_dim=len(np.concatenate(one_hot_encoder.categories_)) - 2,
            nb_filters=cnn_hparams.nb_filters,
            output_dim=label_number,
            drop_pct=cnn_hparams.dropout,
        )
        emb_weights = f_text.vocab.vectors
        cnn_model.embedding.weight.data.copy_(emb_weights)
        unk_idx = f_text.vocab.stoi[f_text.unk_token]
        cnn_model.embedding.weight.data[unk_idx] = torch.zeros(PRETRAINED_EMB[embedding_name])
        pad_idx = f_text.vocab.stoi[f_text.pad_token]
        cnn_model.embedding.weight.data[pad_idx] = torch.zeros(PRETRAINED_EMB[embedding_name])

        learner = Learner(cnn_model)

        optimizer = partial(optim.Adam, lr=cnn_hparams.lrates.init_phase)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # since it's a multi-label cls
        step_eval = int(20*int(len(processed_data.training) / batch_size) / 100)

        learner.compile(
            optimizer=optimizer, loss_func=criterion, step_eval=step_eval, metrics=[HammingLoss(), FScore()]
        )

        logger.info("CNN learner compiled for the text classification task")
        return learner


class TrainHighLevels:
    """Train classifier but freeze embedding layer"""

    # pylint: disable=arguments-differ
    def execute(
        self,
        cnn_learner: Learner,
        device: torch.device,
        torch_iterators: TorchIterators,
        cnn_hparams: CnnHyperParameters,
    ) -> Learner:
        """
        Execute node

        Args:
            cnn_learner (Learner): see output of Learner.execute
            device (torch.device): device on which the model is trained,
                GPU is highly recommended
            torch_iterators (TorchIterators): see output of BuildIterators.execute
            cnn_hparams (CnnHyperParameters): CNN hyper-parameters, we will use only the nb of
                epochs at init phase
        """

        logger.info("Starting training of CNN and classification layers")

        cnn_learner.freeze_to(1)
        cnn_learner.fit(
            nb_epochs=cnn_hparams.epochs.init_phase,
            device=device,
            dataloader=torch_iterators.training,
            val_dataloader=torch_iterators.validation,
        )
        return cnn_learner


class FinetuneAll:
    """Finetune all layers"""

    # pylint: disable=arguments-differ
    # pylint: disable=too-many-arguments
    def execute(
        self,
        cnn_learner: Learner,
        device: torch.device,
        torch_iterators: TorchIterators,
        cnn_hparams: CnnHyperParameters,
    ) -> Learner:
        """
        Execute node

        Args:
            cnn_learner (Learner): see output of Learner.execute
            device (torch.device): device on which the model is trained,
                GPU is highly recommended
            torch_iterators (TorchIterators): see output of BuildIterators.execute
            cnn_hparams (CnnHyperParameters): CNN hyper-parameters, we will use only the nb of
                epochs and learning rate at finetuning phase
        """

        logger.info("Unfreezing the whole CNN network")
        cnn_learner.unfreeze()

        logger.info("Starting finetuning of the whole CNN network")
        cnn_learner.fit(
            nb_epochs=cnn_hparams.epochs.init_phase,
            device=device,
            dataloader=torch_iterators.training,
            val_dataloader=torch_iterators.validation,
        )

        logger.info("Training done")
        return cnn_learner


class EvaluateClassifier:
    """Evaluate the trained classifier on the test set"""

    # pylint: disable=arguments-differ
    # pylint: disable=too-many-arguments
    def execute(
        self,
        cnn_learner: Learner,
        device: torch.device,
        torch_iterators: TorchIterators,
        multilabel_binarizer: MultiLabelBinarizer,
    ) -> dict[str, Any]:  # type: ignore
        """
        Execute the node

        Args:
            cnn_learner (Learner): CNN text classifier learner
            device (torch.device): GPU or CPU
            torch_iterators (TorchIterators): used to fetch the test iterator

        Returns:
            Mapping[str, Any]: performance report that contains: accuracy, precision, recall and
                confusion matrix
        """

        test_ds = torch_iterators.test.dataset

        y_gt = np.array([sample.label for sample in test_ds.examples])

        y_label, _ = cnn_learner.predict(torch_iterators.test, device)
        perfs = classification_report(y_gt, y_label, target_names=multilabel_binarizer.classes_, output_dict=True)
        logger.info("Test results")
        logger.info(perfs)

        return perfs
