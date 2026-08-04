import optuna
import torch
import sys
import os
import shutil
import json
import inspect
from pathlib import Path

# Add parent directory to path to import Models and data_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_pipeline as dp
from Models.Multi_head_CNN_BiLSTM import MULTI_HEAD_CNN_LSTM as MultiHeadCNNBiLSTM, train_multi_head_cnn_lstm as train_multi_head_cnn_bilstm, evaluate_multi_head_cnn_lstm as evaluate_multi_head_cnn_bilstm

# Configuration (same as main.ipynb)
CONFIG = {
    "data_path": "data2",
    "window_size": 40,
    "step_size": 20,
    "train_ratio": 0.6,   # subjects, not sessions; the remaining 0.2 becomes the test split
    "val_ratio": 0.2,
    "batch_size_eval": 1,
    "num_epochs": 50,
    "learning_rate": 2e-4,
    "clip_grad_norm": 1.0,
    "patience": 7,
}

device = dp.get_device()
data = dp.load_filtered_recordings(data_path=CONFIG["data_path"], min_recordings_per_activity=5)
activity_to_id = dp.encode_activities(data)
dp.clean_imu_columns(data, dp.IMU_FEATURES)
num_classes = len(activity_to_id)
num_features = len(dp.IMU_FEATURES)

def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    lstm_layers = trial.suggest_int("lstm_layers", 2, 6)
    
    splits = dp.make_train_val_test_loaders(
        data=data,
        imu_features=dp.IMU_FEATURES,
        window_size=CONFIG["window_size"],
        step_size=CONFIG["step_size"],
        train_ratio=CONFIG["train_ratio"],
        val_ratio=CONFIG["val_ratio"],
        batch_size_train=batch_size,
        batch_size_val=CONFIG["batch_size_eval"],
        batch_size_test=CONFIG["batch_size_eval"],
    )

    class_weights = dp.compute_class_weights(splits.train_dataset, num_classes, device)

    model = MultiHeadCNNBiLSTM(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=64, 
        lstm_layers=lstm_layers,
    ).to(device)

    train_kwargs = {
        "model": model,
        "train_loader": splits.train_loader,
        "val_loader": splits.val_loader,
        "device": device,
        "learning_rate": CONFIG["learning_rate"],
        "num_epochs": CONFIG["num_epochs"],
        "clip_grad_norm": CONFIG["clip_grad_norm"],
        "patience": CONFIG["patience"],
        "best_model_path": f"studies/best_temp_MultiHeadCNNBiLSTM_trial_{trial.number}.pt"
    }

    train_sig = inspect.signature(train_multi_head_cnn_bilstm)
    if "class_weights" in train_sig.parameters:
        train_kwargs["class_weights"] = class_weights

    history = train_multi_head_cnn_bilstm(**train_kwargs)
    
    # Load the best model from this trial for evaluation
    tmp_path = Path(train_kwargs["best_model_path"])
    if tmp_path.exists():
        model.load_state_dict(torch.load(tmp_path, map_location=device))

    # Score the trial on VALIDATION. The test split stays untouched until the search is
    # over - optimizing against it would make the final number meaningless.
    eval_result = evaluate_multi_head_cnn_bilstm(
        model=model,
        data_loader=splits.val_loader,
        device=device,
        history=history,
        best_model_path=str(tmp_path)
    )

    val_f1 = eval_result["metrics"]["f1_score"]
    trial.set_user_attr("history", history)

    return val_f1

def save_best_model_callback(study, trial):
    if study.best_trial.number == trial.number:
        src = f"studies/best_temp_MultiHeadCNNBiLSTM_trial_{trial.number}.pt"
        if os.path.exists(src):
            optuna_dir = Path("studies/optuna_optimize_MultiHeadCNNBiLSTM")
            optuna_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, str(optuna_dir / "best_model.pt"))

if __name__ == "__main__":
    optuna_dir = Path("studies/optuna_optimize_MultiHeadCNNBiLSTM")
    optuna_dir.mkdir(parents=True, exist_ok=True)
    db_path = f"sqlite:///{optuna_dir}/study_subject_independent.db"
    
    study = optuna.create_study(direction="maximize", study_name="optimize_MultiHeadCNNBiLSTM_subject_independent", storage=db_path, load_if_exists=True)
    
    try:
        study.optimize(objective, n_trials=15, callbacks=[save_best_model_callback])
    finally:
        for f in Path("studies").glob("best_temp_MultiHeadCNNBiLSTM_trial_*.pt"):
            f.unlink()
    
    print("\n===============================")
    print("Best parameters for MultiHead_CNN_BiLSTM:")
    print(study.best_params)
    print("Best VALIDATION F1 reached during the search:")
    print(study.best_value)

    best_trial = study.best_trial
    best_batch_size = best_trial.params["batch_size"]
    best_lstm_layers = best_trial.params["lstm_layers"]
    history = best_trial.user_attrs.get("history")

    splits = dp.make_train_val_test_loaders(
        data=data,
        imu_features=dp.IMU_FEATURES,
        window_size=CONFIG["window_size"],
        step_size=CONFIG["step_size"],
        train_ratio=CONFIG["train_ratio"],
        val_ratio=CONFIG["val_ratio"],
        batch_size_train=best_batch_size,
        batch_size_val=CONFIG["batch_size_eval"],
        batch_size_test=CONFIG["batch_size_eval"],
    )

    model = MultiHeadCNNBiLSTM(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=64, 
        lstm_layers=best_lstm_layers,
    ).to(device)
    
    best_model_path = str(optuna_dir / "best_model.pt")
    if Path(best_model_path).exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    val_result = evaluate_multi_head_cnn_bilstm(
        model=model,
        data_loader=splits.val_loader,
        device=device,
        history=history,
        best_model_path=best_model_path,
    )

    # The single look at the test subjects. Hyperparameters are frozen by this point,
    # so this is the number to report.
    test_result = evaluate_multi_head_cnn_bilstm(
        model=model,
        data_loader=splits.test_loader,
        device=device,
        history=history,
        best_model_path=best_model_path,
    )

    print(f"Validation F1 of selected model: {val_result['metrics']['f1_score']:.4f}")
    print(f"TEST F1 (report this one):       {test_result['metrics']['f1_score']:.4f}")

    # Top-level keys hold the TEST scores so downstream analysis reads the honest number;
    # the validation scores that actually drove the search sit next to them.
    serializable = {
        "MultiHead_CNN_BiLSTM": {
            "history": history,
            "confusion_matrix": test_result["confusion_matrix"],
            "metrics": test_result["metrics"],
            "y_true": test_result["y_true"],
            "y_pred": test_result["y_pred"],
            "param_count": test_result["param_count"],
            "best_model_path": test_result["best_model_path"],
            "best_params": study.best_params,
            "val_metrics": val_result["metrics"],
            "val_confusion_matrix": val_result["confusion_matrix"],
            "best_val_f1_during_search": study.best_value,
            "split": {
                "type": "subject_independent",
                "train_ratio": CONFIG["train_ratio"],
                "val_ratio": CONFIG["val_ratio"],
                "subjects": splits.subjects,
            },
        }
    }

    results_path = optuna_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nFinal evaluation saved to {results_path}")
    print("===============================\n")
