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
    "train_ratio": 0.6,   # subjects, not sessions; the remaining 0.2 becomes the test split
    "val_ratio": 0.2,
    "batch_size_eval": 1,
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
        "study_name": "optimize_CNN_ResBiLSTM_subject_independent",
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
    db_path = f"sqlite:///{optuna_dir}/study_subject_independent.db"

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

        val_f1_scores = []
        test_f1_scores = []

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
                "train_loader": splits.train_loader,
                "val_loader": splits.val_loader,
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

            # Validation drives the choice of setup.
            val_eval = model_config["eval_fn"](
                model=model,
                data_loader=splits.val_loader,
                device=device,
                history=history,
                best_model_path=tmp_model_path,
            )

            # Test is only measured, never compared against - the winner below is picked
            # on validation alone, so this stays an honest held-out estimate.
            test_eval = model_config["eval_fn"](
                model=model,
                data_loader=splits.test_loader,
                device=device,
                history=history,
                best_model_path=tmp_model_path,
            )

            val_f1 = val_eval["metrics"]["f1_score"]
            test_f1 = test_eval["metrics"]["f1_score"]
            val_f1_scores.append(val_f1)
            test_f1_scores.append(test_f1)
            print(f"  -> Finished Run {run+1}. Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

            if Path(tmp_model_path).exists():
                Path(tmp_model_path).unlink()

        avg_val_f1 = np.mean(val_f1_scores) if val_f1_scores else 0.0
        std_val_f1 = np.std(val_f1_scores) if val_f1_scores else 0.0
        avg_test_f1 = np.mean(test_f1_scores) if test_f1_scores else 0.0
        std_test_f1 = np.std(test_f1_scores) if test_f1_scores else 0.0

        results[str(setup)] = {
            "scores": val_f1_scores,
            "average": avg_val_f1,
            "std": std_val_f1,
            "test_scores": test_f1_scores,
            "test_average": avg_test_f1,
            "test_std": std_test_f1,
        }
        print(
            f"  Summary for {setup} -> Val Avg: {avg_val_f1:.4f} (Std {std_val_f1:.4f}) | "
            f"Test Avg: {avg_test_f1:.4f} (Std {std_test_f1:.4f})"
        )

    # Final summary
    summary_lines = []
    summary_lines.append("\n==================================")
    summary_lines.append(f"FINAL VALIDATION RESULTS FOR {model_name} (Highest Average VALIDATION F1 Wins):")

    best_setup = None
    best_avg = -1
    best_test_avg = 0.0
    best_test_std = 0.0

    for setup_str, stats in results.items():
        score_str = ", ".join(f"{s:.4f}" for s in stats['scores'])
        test_score_str = ", ".join(f"{s:.4f}" for s in stats['test_scores'])
        summary_lines.append(f"- {setup_str}")
        summary_lines.append(f"      Val  F1: {stats['average']:.4f}  |  Std Dev: {stats['std']:.4f}")
        summary_lines.append(f"      Val  Runs: [{score_str}]")
        summary_lines.append(f"      Test F1: {stats['test_average']:.4f}  |  Std Dev: {stats['test_std']:.4f}")
        summary_lines.append(f"      Test Runs: [{test_score_str}]")

        if stats['average'] > best_avg:
            best_avg = stats['average']
            best_test_avg = stats['test_average']
            best_test_std = stats['test_std']
            best_setup = setup_str

    summary_lines.append(f"\n🏆 REAL WINNER: {best_setup} 🏆")
    summary_lines.append(
        f"Chosen on validation F1 {best_avg:.4f} across {CONFIG['runs_per_setup']} runs."
    )
    summary_lines.append(
        f"Held-out TEST F1 for that setup: {best_test_avg:.4f} (Std {best_test_std:.4f}) <- report this."
    )
    summary_lines.append("==================================\n")

    final_output = "\n".join(summary_lines)
    print(final_output)

    output_filename = f"studies/validation_results_summary_for_{model_name}.txt"
    with open(output_filename, "w") as f:
        f.write(final_output)
    print(f"Results successfully saved to {output_filename}\n")


if __name__ == "__main__":
    main()
