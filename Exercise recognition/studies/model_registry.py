"""One description per neural network, so every study script builds models the same way.

Before this existed, each `optimize_<model>.py` carried its own copy of "how do I
construct / train / evaluate this model". The copies drifted (different keyword names,
one of them seeded, the others not), which is exactly the kind of difference that shows
up later as an unexplained score gap between models. Everything model specific now lives
in `MODELS` below, and the runner in `study_runner.py` is model agnostic.

The six architectures share a common interface already:

    model = ModelClass(num_features=..., num_classes=..., <hidden kwarg>=..., <layer kwarg>=...)
    history = train_fn(model=..., train_loader=..., val_loader=..., class_weights=...,
                       device=..., learning_rate=..., num_epochs=..., clip_grad_norm=...,
                       patience=..., best_model_path=...)
    result  = eval_fn(model=..., data_loader=..., device=..., history=..., best_model_path=...)

so a spec only has to say which class and which two keyword names to use.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

# Shared training configuration. Identical across models on purpose: the studies compare
# architectures, so anything that is not the architecture has to be held constant.
CONFIG = {
    "data_path": "data2",
    "window_size": 40,
    "step_size": 20,
    "train_ratio": 0.6,   # subjects, not sessions; the remaining 0.2 becomes the test split
    "val_ratio": 0.2,
    "split_seed": 42,     # never varied - see study_runner.SPLIT_SEED_NOTE
    # Batch size used whenever the model is only scored, never trained. This was 1, which
    # made every scoring pass 18k separate forward passes of a 40x9 window - the GPU sat
    # at ~23% utilisation paying per-call overhead, and since early stopping validates
    # after every epoch it dominated the whole sweep (75-90% of all compute). Batching
    # changes nothing about the result: the model is in eval mode, so BatchNorm uses its
    # running statistics rather than batch statistics, no layer mixes samples within a
    # batch, and the metrics are computed over the fully concatenated predictions.
    # Measured on this dataset: identical predictions, 54x faster for CNN_BiLSTM and 150x
    # for the 6-layer multi-head model.
    "batch_size_eval": 256,
    "num_epochs": 50,
    "learning_rate": 2e-4,
    "clip_grad_norm": 1.0,
    "patience": 7,
    "hidden_dim": 64,
    "min_recordings_per_activity": 5,
}

# The hyperparameter space every model is searched over.
BATCH_SIZE_CHOICES = [64, 128, 256]
LAYER_RANGE = (2, 6)


@dataclass(frozen=True)
class ModelSpec:
    """Everything the runner needs to know about one architecture."""

    key: str                 # cli name, e.g. "cnn_bilstm"
    display_name: str        # name used in results.json and printed tables
    module: str              # dotted path of the module holding the model
    class_name: str          # model class inside that module
    train_fn_name: str
    eval_fn_name: str
    layer_param: str = ""    # "lstm_layers"/"gru_layers"; empty for models with nothing to tune
    hidden_param: str = "hidden_dim"   # "hidden_dim" or "hidden_size"
    # "nn" models are searched over batch size and depth. "classical" models (RandomForest,
    # XGBoost, Rocket) have no architecture knobs exposed here, so they skip the search
    # stage entirely and go straight to repeated evaluation.
    kind: str = "nn"
    # Batch size used to feed the loaders for classical models. They pull the whole split
    # into numpy before fitting, so this changes collection speed and nothing else.
    collect_batch_size: int = 256
    # Extra constructor arguments for this model, recorded here rather than edited into the
    # model file so the study's choice is explicit and the library default stays intact.
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Configurations worth trying before the sampler starts guessing.
    enqueue: Sequence[Dict[str, Any]] = field(default_factory=tuple)

    def load(self):
        """Import the model class and its train/evaluate functions.

        Imports are deferred so that listing models, or running one model, does not pull
        in the other five modules.
        """
        module = __import__(self.module, fromlist=["*"])
        return (
            getattr(module, self.class_name),
            getattr(module, self.train_fn_name),
            getattr(module, self.eval_fn_name),
        )

    @property
    def tunable(self) -> bool:
        """Whether this model has a hyperparameter space worth searching."""
        return self.kind == "nn"

    def build_model(self, params: Dict[str, Any], num_features: int, num_classes: int, device,
                    seed: Optional[int] = None):
        model_class, _, _ = self.load()

        if self.kind == "classical":
            # sklearn/xgboost/sktime estimators take their randomness from random_state, not
            # from the global torch/numpy seeding. Without threading the seed through here
            # all five "independent" repeats would fit byte-identical models.
            kwargs = {"num_features": num_features, "num_classes": num_classes}
            kwargs.update(self.extra_kwargs)
            if seed is not None:
                kwargs["random_state"] = seed
            return model_class(**kwargs).to(device)

        model = model_class(
            num_features=num_features,
            num_classes=num_classes,
            **{
                self.hidden_param: CONFIG["hidden_dim"],
                self.layer_param: params[self.layer_param],
            },
        )
        return model.to(device)

    def default_params(self) -> Dict[str, Any]:
        """The single configuration used when there is nothing to search."""
        return {"batch_size": self.collect_batch_size}

    def suggest_params(self, trial) -> Dict[str, Any]:
        return {
            "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES),
            self.layer_param: trial.suggest_int(self.layer_param, *LAYER_RANGE),
        }


MODELS: Dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec(
            key="cnn_bilstm",
            display_name="CNN_BiLSTM",
            module="Models.CNN_BiLSTM",
            class_name="CNNLSTM",
            train_fn_name="train_cnnlstm",
            eval_fn_name="evaluate_cnnlstm",
            layer_param="lstm_layers",
        ),
        ModelSpec(
            key="cnn_bigru",
            display_name="CNN_BiGRU",
            module="Models.CNN_BiGRU",
            class_name="CNNBiGRU",
            train_fn_name="train_cnn_bigru",
            eval_fn_name="evaluate_cnn_bigru",
            layer_param="gru_layers",
            hidden_param="hidden_size",
        ),
        ModelSpec(
            key="cnn_resbilstm",
            display_name="CNN_ResBiLSTM",
            module="Models.CNN_ResBiLSTM",
            class_name="CNNLSTM",
            train_fn_name="train_cnnlstm",
            eval_fn_name="evaluate_cnnlstm",
            layer_param="lstm_layers",
        ),
        ModelSpec(
            key="cnn_resbigru",
            display_name="CNN_ResBiGRU",
            module="Models.CNN_ResBiGru",
            class_name="CNNResBiGRU",
            train_fn_name="train_cnn_resbigru",
            eval_fn_name="evaluate_cnn_resbigru",
            layer_param="gru_layers",
            hidden_param="hidden_size",
        ),
        ModelSpec(
            key="multi_head_cnn_bilstm",
            display_name="MultiHead_CNN_BiLSTM",
            module="Models.Multi_head_CNN_BiLSTM",
            class_name="MULTI_HEAD_CNN_LSTM",
            train_fn_name="train_multi_head_cnn_lstm",
            eval_fn_name="evaluate_multi_head_cnn_lstm",
            layer_param="lstm_layers",
        ),
        ModelSpec(
            key="multi_head_cnn_resbilstm",
            display_name="MultiHead_CNN_ResBiLSTM",
            module="Models.Multi_head_CNN_ResBiLSTM",
            class_name="MULTI_HEAD_CNN_LSTM",
            train_fn_name="train_multi_head_cnn_lstm",
            eval_fn_name="evaluate_multi_head_cnn_lstm",
            layer_param="lstm_layers",
            # This combination was the previous hand-picked favourite; try it first so the
            # search never reports a best that is worse than what was already known.
            enqueue=({"batch_size": 256, "lstm_layers": 2},),
        ),
        # Classical baselines. No search stage - they are evaluated at their library
        # defaults, repeated across seeds and folds exactly like the networks so the
        # cross-validated numbers are directly comparable.
        ModelSpec(
            key="randomforest",
            display_name="RandomForest",
            module="Models.RandomForest",
            class_name="RandomForestModel",
            train_fn_name="train_random_forest",
            eval_fn_name="evaluate_random_forest",
            kind="classical",
        ),
        ModelSpec(
            key="xgboost",
            display_name="XGBoost",
            module="Models.XGBoost",
            class_name="XGBoostModel",
            train_fn_name="train_xgboost",
            eval_fn_name="evaluate_xgboost",
            kind="classical",
        ),
        ModelSpec(
            key="rocket",
            display_name="Rocket",
            module="Models.Rocket",
            class_name="RocketModel",
            train_fn_name="train_rocket",
            eval_fn_name="evaluate_rocket",
            kind="classical",
            # The library default of 10,000 kernels produces 20,000 features per window;
            # at ~45k training windows sktime's pipeline peaked near 40 GB and was killed
            # by the OOM killer on every fold. 1,000 kernels keeps the same method well
            # inside memory, and ROCKET's accuracy is known to saturate far below 10k.
            extra_kwargs={"num_kernels": 1000},
        ),
    ]
}

MODEL_KEYS: List[str] = list(MODELS)
