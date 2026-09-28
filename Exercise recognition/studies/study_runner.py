"""Search hyperparameters, then re-evaluate the best ones over several seeds.

Why the repeated evaluation exists
----------------------------------
Even with a fixed manual seed, two runs of the same hyperparameter combination did not
produce the same score. cuDNN picks non-deterministic kernels for the recurrent layers,
and the reductions inside them are not associative on GPU, so tiny floating point
differences accumulate over 50 epochs and change which epoch early stopping selects. A
single number per configuration therefore mixes the effect of the hyperparameters with
run-to-run noise, and an Optuna search that ranks configurations on one run each will
happily crown whichever configuration got lucky.

So the search is only used as a *filter*. For every model the top three configurations it
found are then retrained `N_REPEATS` times from independent seeds, and what gets reported
is the mean and standard deviation over those runs.

What is varied and what is not
------------------------------
Varied per repeat: the seed feeding torch/numpy/random, i.e. weight initialisation,
dropout masks and batch shuffling order.

Held fixed: the train/val/test split. `CONFIG["split_seed"]` is never touched, so all runs
of all models see exactly the same subjects in the same three groups. Re-splitting per
repeat would fold "which people ended up in the test set" into the standard deviation and
make the numbers incomparable across models.

Test set discipline
-------------------
The Optuna search only ever sees validation F1. The test split is scored during the
repeated runs, but selection between the three configurations is still made on validation
F1 - the test numbers are recorded alongside, never optimised against.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_pipeline as dp
from studies.model_registry import CONFIG, MODELS, ModelSpec

STUDIES_DIR = Path(__file__).resolve().parent
ROOT = STUDIES_DIR.parent
RESULTS_ROOT = STUDIES_DIR / "results"

# Defaults for the protocol described in the module docstring.
N_TRIALS = 15        # Optuna trials per model
TOP_K = 3            # configurations carried from the search into repeated evaluation
N_REPEATS = 5        # independent runs per carried configuration
REPEAT_SEEDS = [0, 1, 2, 3, 4]

METRIC_KEYS = ["accuracy", "precision", "recall", "f1_score"]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
class DataBundle:
    """Loads the dataset once and caches one set of loaders per batch size.

    Windowing the whole RecGym recording takes far longer than a training epoch, and the
    protocol rebuilds loaders 6 models x 3 configs x 5 seeds times. Since the split is
    deterministic (fixed `split_seed`) and the loaders are stateless between epochs, the
    same objects can be reused; only the RNG state that drives shuffling is reset per run.
    """

    def __init__(self, verbose: bool = True, fold: Optional[int] = None, n_folds: int = 5):
        # fold=None keeps the original single ratio-balanced split; fold=i selects
        # cross-validation fold i, so the same class serves both study types.
        self.fold = fold
        self.n_folds = n_folds
        # Resolve against the project root so the studies run the same from any cwd.
        data_path = Path(CONFIG["data_path"])
        if not data_path.is_absolute():
            data_path = ROOT / data_path

        self.data = dp.load_filtered_recordings(
            data_path=str(data_path),
            min_recordings_per_activity=CONFIG["min_recordings_per_activity"],
        )
        self.activity_to_id = dp.encode_activities(self.data)
        dp.clean_imu_columns(self.data, dp.IMU_FEATURES)
        self.num_classes = len(self.activity_to_id)
        self.num_features = len(dp.IMU_FEATURES)
        self.device = dp.get_device()
        self._cache: Dict[int, Any] = {}
        self._weights: Dict[int, torch.Tensor] = {}
        if verbose:
            where = "single split" if fold is None else f"CV fold {fold + 1}/{n_folds}"
            print(f"Loaded {len(self.data)} sessions | {self.num_classes} classes | "
                  f"device={self.device} | {where}")

    def splits_for(self, batch_size: int):
        if batch_size not in self._cache:
            self._cache[batch_size] = dp.make_train_val_test_loaders(
                data=self.data,
                imu_features=dp.IMU_FEATURES,
                window_size=CONFIG["window_size"],
                step_size=CONFIG["step_size"],
                train_ratio=CONFIG["train_ratio"],
                val_ratio=CONFIG["val_ratio"],
                batch_size_train=batch_size,
                batch_size_val=CONFIG["batch_size_eval"],
                batch_size_test=CONFIG["batch_size_eval"],
                seed=CONFIG["split_seed"],
                fold=self.fold,
                n_folds=self.n_folds,
            )
        return self._cache[batch_size]

    def class_weights_for(self, batch_size: int) -> torch.Tensor:
        # Depends only on the training windows, which are identical for every batch size,
        # so compute it once and share it.
        if not self._weights:
            splits = self.splits_for(batch_size)
            self._weights[0] = dp.compute_class_weights(
                splits.train_dataset, self.num_classes, self.device
            )
        return self._weights[0]


# --------------------------------------------------------------------------------------
# One training run
# --------------------------------------------------------------------------------------
def train_and_evaluate(
    spec: ModelSpec,
    params: Dict[str, Any],
    seed: int,
    bundle: DataBundle,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """Train one model from scratch at `seed` and score it on validation and test."""
    _, train_fn, eval_fn = spec.load()

    # Reseed before touching anything stochastic: weight init, dropout and the shuffle
    # order of the cached train loader all draw from these generators.
    dp.set_seed(seed)

    batch_size = params["batch_size"]
    splits = bundle.splits_for(batch_size)
    class_weights = bundle.class_weights_for(batch_size)
    model = spec.build_model(params, bundle.num_features, bundle.num_classes, bundle.device, seed=seed)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    history = train_fn(
        model=model,
        train_loader=splits.train_loader,
        val_loader=splits.val_loader,
        class_weights=class_weights,
        device=bundle.device,
        learning_rate=CONFIG["learning_rate"],
        num_epochs=CONFIG["num_epochs"],
        clip_grad_norm=CONFIG["clip_grad_norm"],
        patience=CONFIG["patience"],
        best_model_path=str(checkpoint_path),
    )
    train_seconds = time.time() - started

    # Both evaluations reload the early-stopping checkpoint, so they score the same
    # weights - the epoch with the lowest validation loss, not the last epoch.
    val_result = eval_fn(
        model=model,
        data_loader=splits.val_loader,
        device=bundle.device,
        history=history,
        best_model_path=str(checkpoint_path),
    )
    test_result = eval_fn(
        model=model,
        data_loader=splits.test_loader,
        device=bundle.device,
        history=history,
        best_model_path=str(checkpoint_path),
    )

    record = {
        "seed": seed,
        "params": dict(params),
        "val_metrics": val_result["metrics"],
        "test_metrics": test_result["metrics"],
        "val_confusion_matrix": val_result["confusion_matrix"],
        "test_confusion_matrix": test_result["confusion_matrix"],
        "y_true": test_result["y_true"],
        "y_pred": test_result["y_pred"],
        "param_count": test_result["param_count"],
        "epochs_trained": len(history.get("train_losses", [])),
        "train_seconds": train_seconds,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return record


# --------------------------------------------------------------------------------------
# Stage 1: the search
# --------------------------------------------------------------------------------------
def run_search(spec: ModelSpec, bundle: DataBundle, model_dir: Path, n_trials: int) -> optuna.Study:
    """Rank configurations by a single-run validation F1. Noisy on purpose - the repeated
    stage is what turns the shortlist into a defensible number."""
    tmp_dir = model_dir / "search_tmp"
    storage = f"sqlite:///{model_dir / 'study.db'}"
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{spec.key}_subject_independent",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=CONFIG["split_seed"]),
    )

    for enqueued in spec.enqueue:
        study.enqueue_trial(enqueued, skip_if_exists=True)

    def objective(trial):
        params = spec.suggest_params(trial)
        record = train_and_evaluate(
            spec=spec,
            params=params,
            seed=CONFIG["split_seed"],
            bundle=bundle,
            checkpoint_path=tmp_dir / f"trial_{trial.number}.pt",
        )
        trial.set_user_attr("val_metrics", record["val_metrics"])
        trial.set_user_attr("epochs_trained", record["epochs_trained"])
        trial.set_user_attr("train_seconds", record["train_seconds"])
        return record["val_metrics"]["f1_score"]

    remaining = max(0, n_trials - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
    if remaining:
        try:
            study.optimize(objective, n_trials=remaining)
        finally:
            # Search checkpoints are throwaway; the reported models come from stage 2.
            for stale in tmp_dir.glob("trial_*.pt"):
                stale.unlink()
            if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                tmp_dir.rmdir()
    else:
        print(f"[{spec.key}] study.db already has {n_trials}+ completed trials, skipping search")

    return study


def top_configurations(spec: ModelSpec, study: optuna.Study, top_k: int) -> List[Dict[str, Any]]:
    """The `top_k` distinct parameter sets with the highest search score.

    Deduplicated: the sampler often revisits a promising combination, and spending a whole
    5-run budget twice on the same configuration would waste two thirds of the shortlist.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)

    configs: List[Dict[str, Any]] = []
    seen = set()
    for trial in completed:
        params = {"batch_size": trial.params["batch_size"], spec.layer_param: trial.params[spec.layer_param]}
        signature = tuple(sorted(params.items()))
        if signature in seen:
            continue
        seen.add(signature)
        configs.append({
            "params": params,
            "search_trial_number": trial.number,
            "search_val_f1": trial.value,
        })
        if len(configs) == top_k:
            break

    return configs


# --------------------------------------------------------------------------------------
# Stage 2: repeated evaluation
# --------------------------------------------------------------------------------------
def aggregate(runs: List[Dict[str, Any]], split: str) -> Dict[str, Dict[str, float]]:
    """mean / std / min / max per metric across the repeats of one configuration."""
    summary = {}
    for metric in METRIC_KEYS:
        values = np.array([run[f"{split}_metrics"][metric] for run in runs], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            # Sample std (ddof=1): these runs are a sample of possible runs, not the
            # whole population of them.
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "values": [float(v) for v in values],
        }
    return summary


def evaluate_configuration(
    spec: ModelSpec,
    config: Dict[str, Any],
    config_index: int,
    bundle: DataBundle,
    model_dir: Path,
    seeds: List[int],
) -> Dict[str, Any]:
    params = config["params"]
    print(f"\n--- [{spec.key}] config {config_index + 1}: {params} over {len(seeds)} runs ---")

    runs = []
    for seed in seeds:
        print(f"\n[{spec.key}] config {config_index + 1}, run seed={seed}")
        checkpoint = model_dir / "checkpoints" / f"config{config_index + 1}_seed{seed}.pt"
        run = train_and_evaluate(spec, params, seed, bundle, checkpoint)
        print(
            f"[{spec.key}] config {config_index + 1} seed={seed}: "
            f"val F1={run['val_metrics']['f1_score']:.4f}  test F1={run['test_metrics']['f1_score']:.4f}"
        )
        runs.append(run)

    val_summary = aggregate(runs, "val")
    test_summary = aggregate(runs, "test")
    print(
        f"[{spec.key}] config {config_index + 1} {params}: "
        f"val F1 {val_summary['f1_score']['mean']:.4f} +/- {val_summary['f1_score']['std']:.4f} | "
        f"test F1 {test_summary['f1_score']['mean']:.4f} +/- {test_summary['f1_score']['std']:.4f}"
    )

    return {
        "config_index": config_index + 1,
        "params": params,
        "search_trial_number": config["search_trial_number"],
        "search_val_f1": config["search_val_f1"],
        "seeds": seeds,
        "param_count": runs[0]["param_count"],
        "val_summary": val_summary,
        "test_summary": test_summary,
        "runs": runs,
    }


# --------------------------------------------------------------------------------------
# Whole study for one model
# --------------------------------------------------------------------------------------
def run_model_study(
    key: str,
    bundle: Optional[DataBundle] = None,
    n_trials: int = N_TRIALS,
    top_k: int = TOP_K,
    seeds: Optional[List[int]] = None,
    results_root: Path = RESULTS_ROOT,
    fold: Optional[int] = None,
    n_folds: int = 5,
) -> Dict[str, Any]:
    spec = MODELS[key]
    seeds = seeds or list(REPEAT_SEEDS)
    bundle = bundle or DataBundle(fold=fold, n_folds=n_folds)

    model_dir = results_root / key
    model_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()

    if not spec.tunable:
        # Nothing to search: one fixed configuration, straight to repeated evaluation.
        print(f"\n{'=' * 78}\n[{spec.key}] no hyperparameter space - skipping search\n{'=' * 78}")
        study = None
        configs = [{"params": spec.default_params(), "search_trial_number": None, "search_val_f1": None}]
        _dump(model_dir / "search_trials.json", [])
        _dump(model_dir / "top_configs.json", configs)
        return _run_repeats(spec, configs, bundle, model_dir, seeds, n_trials, started, study, results_root)

    print(f"\n{'=' * 78}\n[{spec.key}] STAGE 1/2 - Optuna search ({n_trials} trials)\n{'=' * 78}")
    study = run_search(spec, bundle, model_dir, n_trials)

    trials_dump = [
        {
            "number": t.number,
            "state": t.state.name,
            "params": t.params,
            "val_f1": t.value,
            "epochs_trained": t.user_attrs.get("epochs_trained"),
            "train_seconds": t.user_attrs.get("train_seconds"),
            "val_metrics": t.user_attrs.get("val_metrics"),
        }
        for t in study.trials
    ]
    _dump(model_dir / "search_trials.json", trials_dump)

    configs = top_configurations(spec, study, top_k)
    if not configs:
        raise RuntimeError(f"[{key}] the search produced no completed trials, nothing to evaluate")
    _dump(model_dir / "top_configs.json", configs)

    return _run_repeats(spec, configs, bundle, model_dir, seeds, n_trials, started, study, results_root)


def _run_repeats(
    spec: ModelSpec,
    configs: List[Dict[str, Any]],
    bundle: DataBundle,
    model_dir: Path,
    seeds: List[int],
    n_trials: int,
    started: float,
    study: Optional[optuna.Study],
    results_root: Path,
) -> Dict[str, Any]:
    """Retrain every carried configuration across `seeds` and write the result files.

    Shared by both paths: models with a search space arrive here with the top-k
    configurations, models without one arrive with their single default configuration.
    """
    key = spec.key

    print(f"\n{'=' * 78}")
    stage = "STAGE 2/2" if spec.tunable else "REPEATED EVALUATION"
    print(f"[{spec.key}] {stage} - {len(configs)} configuration(s) x {len(seeds)} independent runs")
    for i, config in enumerate(configs):
        score = config.get("search_val_f1")
        suffix = f"  (search val F1 {score:.4f})" if score is not None else "  (library defaults, not searched)"
        print(f"    {i + 1}. {config['params']}{suffix}")
    print(f"{'=' * 78}")

    evaluated = [
        evaluate_configuration(spec, config, i, bundle, model_dir, seeds)
        for i, config in enumerate(configs)
    ]

    # Selection stays on validation. The test column is reported, never optimised.
    best = max(evaluated, key=lambda c: c["val_summary"]["f1_score"]["mean"])
    # Within the winning configuration, the checkpoint worth keeping is the run with the
    # best validation F1 - again chosen without looking at test.
    best_run = max(best["runs"], key=lambda r: r["val_metrics"]["f1_score"])

    subjects = bundle.splits_for(best["params"]["batch_size"]).subjects

    if spec.tunable:
        note = (
            "Scores vary between runs of identical hyperparameters despite a fixed manual "
            f"seed. The top {len(configs)} configurations from the search are therefore each "
            f"retrained {len(seeds)} times from independent seeds and reported as mean +/- std. "
            "The train/val/test split is identical in every run."
        )
        search_block = {
            "best_params": study.best_params,
            "best_val_f1": study.best_value,
            "n_completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        }
    else:
        note = (
            f"No hyperparameter search: this model is evaluated at its library defaults. It is "
            f"still fitted {len(seeds)} times from independent random_state values and reported "
            "as mean +/- std, so its numbers aggregate identically to the searched models."
        )
        search_block = None

    result = {
        "model_key": key,
        "display_name": spec.display_name,
        "model_kind": spec.kind,
        "protocol": {
            "n_trials": n_trials if spec.tunable else 0,
            "top_k": len(configs),
            "n_repeats": len(seeds),
            "seeds": seeds,
            "searched": spec.tunable,
            "selection_metric": "mean validation macro F1 over the repeated runs",
            "note": note,
        },
        "config": dict(CONFIG),
        "split": {
            "type": "subject_independent" if bundle.fold is None else "subject_independent_cv",
            "train_ratio": CONFIG["train_ratio"],
            "val_ratio": CONFIG["val_ratio"],
            "seed": CONFIG["split_seed"],
            "fold": bundle.fold,
            "n_folds": bundle.n_folds if bundle.fold is not None else None,
            "subjects": subjects,
        },
        "search": search_block,
        "configurations": evaluated,
        "selected": {
            "params": best["params"],
            "param_count": best["param_count"],
            "val_f1_mean": best["val_summary"]["f1_score"]["mean"],
            "val_f1_std": best["val_summary"]["f1_score"]["std"],
            "test_metrics_mean": {m: best["test_summary"][m]["mean"] for m in METRIC_KEYS},
            "test_metrics_std": {m: best["test_summary"][m]["std"] for m in METRIC_KEYS},
            "best_run_seed": best_run["seed"],
            "best_run_checkpoint": best_run["checkpoint_path"],
            "best_run_test_metrics": best_run["test_metrics"],
            "best_run_test_confusion_matrix": best_run["test_confusion_matrix"],
        },
        "total_seconds": time.time() - started,
    }

    _dump(model_dir / "results.json", result)
    # Compact companion file: same content minus the per-run histories, confusion matrices
    # and prediction dumps, so it can be opened and diffed by hand.
    _dump(model_dir / "summary.json", _strip_bulky(result))

    print(f"\n[{spec.key}] done in {result['total_seconds'] / 60:.1f} min -> {model_dir / 'results.json'}")
    return result


def _strip_bulky(result: Dict[str, Any]) -> Dict[str, Any]:
    slim = {k: v for k, v in result.items() if k != "configurations"}
    slim["configurations"] = [
        {
            "config_index": c["config_index"],
            "params": c["params"],
            "search_val_f1": c["search_val_f1"],
            "param_count": c["param_count"],
            "val_summary": {m: {k: v for k, v in c["val_summary"][m].items()} for m in METRIC_KEYS},
            "test_summary": {m: {k: v for k, v in c["test_summary"][m].items()} for m in METRIC_KEYS},
            "runs": [
                {
                    "seed": r["seed"],
                    "val_metrics": r["val_metrics"],
                    "test_metrics": r["test_metrics"],
                    "epochs_trained": r["epochs_trained"],
                    "train_seconds": r["train_seconds"],
                }
                for r in c["runs"]
            ],
        }
        for c in result["configurations"]
    ]
    slim["selected"] = {k: v for k, v in result["selected"].items() if k != "best_run_test_confusion_matrix"}
    return slim


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
