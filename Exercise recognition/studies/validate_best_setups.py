import optuna
import torch
import sys
import os
import inspect
import numpy as np
from pathlib import Path

# Add parent directory to path to import Models and data_pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_pipeline as dp
from Models.CNN_ResBiLSTM import CNNLSTM as CNNResBiLSTM, train_cnnlstm as train_cnn_resbilstm, evaluate_cnnlstm as evaluate_cnn_resbilstm

# Configuration
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
    "runs_per_setup": 5  # Train each setup multiple times
}

def main():
    dp.set_seed(42)
    device = dp.get_device()
    print("Loading data...")
    data = dp.load_filtered_recordings(data_path=CONFIG["data_path"], min_recordings_per_activity=5)
    activity_to_id = dp.encode_activities(data)
    dp.clean_imu_columns(data, dp.IMU_FEATURES)
    num_classes = len(activity_to_id)
    num_features = len(dp.IMU_FEATURES)

    # Single model: CNN_ResBiLSTM
    model_config = {
        "model_name": "CNN_ResBiLSTM",
        "study_name": "optimize_CNN_ResBiLSTM",
        "optuna_dir": Path("studies/optuna_optimize_CNN_ResBiLSTM"),
        "model_class": CNNResBiLSTM,
        "train_fn": train_cnn_resbilstm,
        "eval_fn": evaluate_cnn_resbilstm,
    }

    model_name = model_config["model_name"]
    print(f"\n\n{'#'*50}")
    print(f"RUNNING VALIDATION FOR: {model_name}")
    print(f"{'#'*50}\n")

    optuna_dir = model_config["optuna_dir"]
    db_path = f"sqlite:///{optuna_dir}/study.db"

    print(f"Loading Optuna study from {db_path}...")
    study = optuna.load_study(study_name=model_config["study_name"], storage=db_path)
    df = study.trials_dataframe()
    completed_trials = df[df['state'] == 'COMPLETE'].sort_values('value', ascending=False)

    # Collect unique top-3 setups
    unique_setups = []
    for _, row in completed_trials.iterrows():
        params = {k.replace('params_', ''): int(v) for k, v in row.items() if k.startswith('params_')}
        if params not in unique_setups:
            unique_setups.append(params)
        if len(unique_setups) == 3:
            break

    print("\n==================================")
    print(f"Setups to validate for {model_name}:")
    for idx, s in enumerate(unique_setups):
        print(f"  {idx + 1}. {s}")
    print("==================================\n")

    results = {}

    for idx, setup in enumerate(unique_setups):
        print(f"\n--- Testing Setup {idx+1}/{len(unique_setups)}: {setup} ---")
        batch_size = setup.get("batch_size")
        lstm_layers = setup.get("lstm_layers")

        f1_scores = []

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

        for run in range(CONFIG["runs_per_setup"]):
            print(f"\n  [Run {run+1}/{CONFIG['runs_per_setup']} for {setup}]")

            model = model_config["model_class"](
                num_features=num_features,
                num_classes=num_classes,
                hidden_dim=64,
                lstm_layers=lstm_layers,
            ).to(device)

            tmp_model_path = f"studies/temp_val_model_{model_name}_setup{idx}_run{run}.pt"

            train_kwargs = {
                "model": model,
                "train_loader": train_loader,
                "val_loader": test_loader,
                "device": device,
                "learning_rate": CONFIG["learning_rate"],
                "num_epochs": CONFIG["num_epochs"],
                "clip_grad_norm": CONFIG["clip_grad_norm"],
                "patience": CONFIG["patience"],
                "best_model_path": tmp_model_path,
            }

            train_sig = inspect.signature(model_config["train_fn"])
            if "class_weights" in train_sig.parameters:
                train_kwargs["class_weights"] = class_weights

            history = model_config["train_fn"](**train_kwargs)

            if Path(tmp_model_path).exists():
                model.load_state_dict(torch.load(tmp_model_path, map_location=device))

            eval_result = model_config["eval_fn"](
                model=model,
                data_loader=test_loader,
                device=device,
                history=history,
                best_model_path=tmp_model_path,
            )

            f1 = eval_result["metrics"]["f1_score"]
            f1_scores.append(f1)
            print(f"  -> Finished Run {run+1}. F1 Score: {f1:.4f}")

            if Path(tmp_model_path).exists():
                Path(tmp_model_path).unlink()

        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        std_f1 = np.std(f1_scores) if f1_scores else 0.0

        results[str(setup)] = {
            "scores": f1_scores,
            "average": avg_f1,
            "std": std_f1,
        }
        print(f"  Summary for {setup} -> Avg: {avg_f1:.4f}, Variance (Std): {std_f1:.4f}")

    # Final summary
    summary_lines = []
    summary_lines.append("\n==================================")
    summary_lines.append(f"FINAL VALIDATION RESULTS FOR {model_name} (Highest Average Wins):")

    best_setup = None
    best_avg = -1

    for setup_str, stats in results.items():
        score_list = [f"{s:.4f}" for s in stats['scores']]
        score_str = ", ".join(score_list)
        summary_lines.append(f"- {setup_str}")
        summary_lines.append(f"      Avg F1: {stats['average']:.4f}  |  Std Dev: {stats['std']:.4f}")
        summary_lines.append(f"      Individual Runs: [{score_str}]")

        if stats['average'] > best_avg:
            best_avg = stats['average']
            best_setup = setup_str

    summary_lines.append(f"\n🏆 REAL WINNER: {best_setup} 🏆")
    summary_lines.append(f"With an average F1 score of {best_avg:.4f} across {CONFIG['runs_per_setup']} runs.")
    summary_lines.append("==================================\n")

    final_output = "\n".join(summary_lines)
    print(final_output)

    output_filename = f"studies/validation_results_summary_for_{model_name}.txt"
    with open(output_filename, "w") as f:
        f.write(final_output)
    print(f"Results successfully saved to {output_filename}\n")


if __name__ == "__main__":
    main()
