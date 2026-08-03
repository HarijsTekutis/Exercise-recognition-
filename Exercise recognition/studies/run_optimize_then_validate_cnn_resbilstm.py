#!/usr/bin/env python3
"""Run optimize_cnn_resbilstm.py then validate_best_setups.py for CNN_ResBiLSTM.

This script executes the optimizer to create the Optuna `study.db`, then runs
the validation script which reads that study and evaluates the top-3 setups.
"""
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent


def run_script(script_path: Path) -> int:
    print(f"Running: {script_path}")
    try:
        res = subprocess.run([sys.executable, str(script_path)], cwd=ROOT)
        return res.returncode
    except Exception as e:
        print(f"Failed to run {script_path}: {e}")
        return 2


def main():
    opt_script = SCRIPTS_DIR / "optimize_cnn_resbilstm.py"
    val_script = SCRIPTS_DIR / "validate_best_setups.py"
    optuna_dir = ROOT / "studies" / "optuna_optimize_CNN_ResBiLSTM"
    optuna_db = optuna_dir / "study.db"

    if not opt_script.exists():
        print(f"Optimizer script not found: {opt_script}")
        sys.exit(1)
    if not val_script.exists():
        print(f"Validator script not found: {val_script}")
        sys.exit(1)

    if optuna_db.exists():
        print(f"Removing previous Optuna study database: {optuna_db}")
        for suffix in ("", "-wal", "-shm"):
            sidecar = Path(str(optuna_db) + suffix)
            if sidecar.exists():
                sidecar.unlink()

    print("Starting optimizer (this may take some time)...")
    rc = run_script(opt_script)
    if rc != 0:
        print(f"Optimizer exited with code {rc}. Aborting.")
        sys.exit(rc)

    # Check expected Optuna DB
    if not optuna_db.exists():
        print(f"Warning: expected Optuna DB not found at {optuna_db}")
    else:
        print(f"Found Optuna DB at {optuna_db}")

    print("Running validation of top setups...")
    rc = run_script(val_script)
    if rc != 0:
        print(f"Validator exited with code {rc}.")
        sys.exit(rc)

    print("Done. Validation summary saved to studies/validation_results_summary_for_CNN_ResBiLSTM.txt")


if __name__ == "__main__":
    main()
