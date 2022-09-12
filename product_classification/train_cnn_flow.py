"""File where data processing Flow is defined"""
from pathlib import Path
import torch
from metaflow import FlowSpec, step, Parameter
from product_classification import logger


class TrainCNNFlow(FlowSpec):
    """Flow used to make some data processing and cleaning\n
    In this flow we will:\n
        - Load inputs data.
        - Clean global dataset
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

    @step
    def start(self) -> None:
        """Get all needed data from flow DataProcessingFlow latest successful run"""
        from metaflow import Flow, get_metadata

        run_test_train_split = Flow("DataProcessingFlow").latest_successful_run

        self.datasets = run_test_train_split.data.datasets
        logger.info("Get training, test, validation datasets")
        for fname, df in self.datasets:
            logger.info(f"{fname} shape: {df.shape}")

        self.multilabel_binarizer = run_test_train_split.data.multilabel_binarizer
        self.pos_weight = torch.Tensor(run_test_train_split.data.pos_weight)

        self.next(self.load_config)

    @step
    def load_config(self) -> None:
        """Load training config from yaml file"""
        import yaml

        with open(self.config_path, "r") as stream:
            self.config = yaml.load(stream, Loader=None)
        logger.info(f"Config parsed: {self.config}")
        self.next(self.extract_cnn_hyperparam)

    @step
    def extract_cnn_hyperparam(self) -> None:
        """Get cnn hyperparam from config file"""
        from product_classification.models import CnnHyperParameters, Epochs, LearningRates

        cnn_hp = self.config.get("hyperparameters_cnn_classifier")
        learning_rates = cnn_hp.get("lrates")
        epochs = cnn_hp.get("epochs")
        self.cnn_hyperparameters = CnnHyperParameters(
            nb_filters=cnn_hp.get("nb_filters"),
            kernels=cnn_hp.get("kernels"),
            dropout=cnn_hp.get("dropout"),
            lrates=LearningRates(
                init_phase=learning_rates.get("init_phase"), finetuning_phase=learning_rates.get("finetuning_phase")
            ),
            epochs=Epochs(init_phase=epochs.get("init_phase"), finetuning_phase=epochs.get("finetuning_phase")),
        )

        self.next(self.init_fields)

    @step
    def init_fields(self) -> None:
        """Init fields for text, categorical data and labels"""
        from product_classification.data_processing.text_processing import FitCategoricalData, InitFields

        fit_categorical_data = FitCategoricalData()
        self.one_hot_encoder = fit_categorical_data.execute(processed_data=self.datasets)

        init_fields = InitFields()
        self.torch_fields = init_fields.execute(one_hot_encoder=self.one_hot_encoder)

        self.next(self.train_cnn)

    # This step can't be splited because of serialization error with torchtext objects 
    @step
    def train_cnn(self) -> None:
        """Create datasets with fields and torch examples"""
        from product_classification.data_processing.text_processing import CreateDatasets, BuildTextVocabulary, BuildIterators
        from product_classification.learner.train import CreateLearner, TrainHighLevels, FinetuneAll, EvaluateClassifier

        self.labels = list(self.datasets.training.iloc[:, 5:].columns)

        create_datasets = CreateDatasets()
        torch_dataset = create_datasets.execute(
            processed_data=self.datasets,
            torch_fields=self.torch_fields,
            cnn_hparams=self.cnn_hyperparameters,
            txt_col=self.config.get("textual_column"),
            cat_cols=self.config.get("categorical_columns"),
            lbl_cols=self.labels,
        )
        
        build_text_vocabulary = BuildTextVocabulary()
        self.torch_fields = build_text_vocabulary.execute(
            torch_datasets=torch_dataset,
            torch_fields=self.torch_fields,
            vocab_size=self.config.get("vocab_size"),
            embedding_name=self.config.get("embedding_name"),
            vectors_cache=self.config.get("vector_cache"),
        )
        
        build_iterators = BuildIterators()
        torch_iterators = build_iterators.execute(
            torch_datasets=torch_dataset,
            batch_size=self.config.get("batch_size"),
            device=self.config.get("device"),
        )
        self.pos_weight = self.pos_weight.to(self.config.get("device"))
        create_learner = CreateLearner()
        self.learner = create_learner.execute(
            cnn_hparams=self.cnn_hyperparameters,
            embedding_name=self.config.get("embedding_name"),
            torch_fields=self.torch_fields,
            processed_data=self.datasets,
            batch_size=self.config.get("batch_size"),
            label_number=len(self.labels),
            one_hot_encoder=self.one_hot_encoder,
            pos_weight=self.pos_weight
        )
        
        train_high_levels = TrainHighLevels()
        self.learner = train_high_levels.execute(
            cnn_learner=self.learner,
            device=self.config.get("device"),
            torch_iterators=torch_iterators,
            cnn_hparams=self.cnn_hyperparameters,
        )
        
        finetune_all = FinetuneAll()
        self.learner = finetune_all.execute(
            cnn_learner=self.learner,
            device=self.config.get("device"),
            torch_iterators=torch_iterators,
            cnn_hparams=self.cnn_hyperparameters,
        )
        
        evaluate_classifier = EvaluateClassifier()
        self.perfs = evaluate_classifier.execute(
            cnn_learner=self.learner,
            device=self.config.get("device"),
            torch_iterators=torch_iterators,
            multilabel_binarizer=self.multilabel_binarizer,
        )
        
        self.next(self.end)

    @step
    def end(self) -> None:
        """"""
        self.perfs
        self.learner
        self.multilabel_binarizer
        pass


if __name__ == "__main__":
    TrainCNNFlow()
