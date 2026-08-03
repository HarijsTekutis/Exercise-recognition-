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
    "train_split": 0.8,
    "batch_size_test": 1,
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
    
    train_loader, test_loader, train_dataset, test_dataset = dp.make_train_test_loaders(
        data=data,
        imu_features=dp.IMU_FEATURES,
        window_size=CONFIG["window_size"],
        step_size=CONFIG["step_size"],
        train_split=CONFIG["train_split"],
        batch_size_train=batch_size,
        batch_size_test=CONFIG["batch_size_test"],
    )

    class_weights = dp.compute_class_weights(train_dataset, num_classes, device)

    model = MultiHeadCNNBiLSTM(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=64, 
        lstm_layers=lstm_layers,
    ).to(device)

    train_kwargs = {
        "model": model,
        "train_loader": train_loader,
        "val_loader": test_loader,
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

    # Evaluate to get the F1 score
    eval_result = evaluate_multi_head_cnn_bilstm(
        model=model,
        data_loader=test_loader,
        device=device,
        history=history,
        best_model_path=str(tmp_path)
    )
    
    best_f1 = eval_result["metrics"]["f1_score"]
    trial.set_user_attr("history", history)

    return best_f1

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
    db_path = f"sqlite:///{optuna_dir}/study.db"
    
    study = optuna.create_study(direction="maximize", study_name="optimize_MultiHeadCNNBiLSTM", storage=db_path, load_if_exists=True)
    
    try:
        study.optimize(objective, n_trials=15, callbacks=[save_best_model_callback])
    finally:
        for f in Path("studies").glob("best_temp_MultiHeadCNNBiLSTM_trial_*.pt"):
            f.unlink()
    
    print("\n===============================")
    print("Best parameters for MultiHead_CNN_BiLSTM:")
    print(study.best_params)
    print("Best Validation Accuracy:")
    print(study.best_value)

    best_trial = study.best_trial
    best_batch_size = best_trial.params["batch_size"]
    best_lstm_layers = best_trial.params["lstm_layers"]
    history = best_trial.user_attrs.get("history")

    train_loader, test_loader, train_dataset, test_dataset = dp.make_train_test_loaders(
        data=data,
        imu_features=dp.IMU_FEATURES,
        window_size=CONFIG["window_size"],
        step_size=CONFIG["step_size"],
        train_split=CONFIG["train_split"],
        batch_size_train=best_batch_size,
        batch_size_test=CONFIG["batch_size_test"],
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
    
    eval_result = evaluate_multi_head_cnn_bilstm(
        model=model,
        data_loader=test_loader,
        device=device,
        history=history,
        best_model_path=best_model_path,
    )
    
    results = {"MultiHead_CNN_BiLSTM": eval_result}
    serializable = {}
    for arch, result in results.items():
        serializable[arch] = {
            "history": result["history"],
            "confusion_matrix": result["confusion_matrix"],
            "metrics": result["metrics"],
            "y_true": result["y_true"],
            "y_pred": result["y_pred"],
            "param_count": result["param_count"],
            "best_model_path": result["best_model_path"],
            "best_params": study.best_params
        }
    
    results_path = optuna_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nFinal evaluation saved to {results_path}")
    print("===============================\n")
