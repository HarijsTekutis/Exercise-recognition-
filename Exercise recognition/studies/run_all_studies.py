#!/usr/bin/env python3
"""Run all Optuna hyperparameter-optimization studies back to back.

Previously each `optimize_*.py` script had to be launched by hand, one at a
time. This runs them all in sequence (each in its own subprocess, so a crash
or CUDA OOM in one model doesn't take down the rest), logs full output per
model to studies/logs/, and prints a summary table of best F1 scores at the
end.

Usage:
    python studies/run_all_studies.py                 # run all 6 studies
    python studies/run_all_studies.py --models cnn_bilstm cnn_bigru
    python studies/run_all_studies.py --skip-existing  # skip models that already have results.json
    python studies/run_all_studies.py --stop-on-fail
    python studies/run_all_studies.py --list
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
LOGS_DIR = SCRIPTS_DIR / "logs"

# model key -> (optimize script, optuna results dir)
STUDIES = {
    "cnn_bilstm": ("optimize_cnn_bilstm.py", "optuna_optimize_CNN_BiLSTM"),
    "cnn_bigru": ("optimize_cnn_bigru.py", "optuna_optimize_CNN_BiGRU"),
    "cnn_resbilstm": ("optimize_cnn_resbilstm.py", "optuna_optimize_CNN_ResBiLSTM"),
    "cnn_resbigru": ("optimize_cnn_resbigru.py", "optuna_optimize_CNN_ResBiGRU"),
    "multi_head_cnn_bilstm": ("optimize_multi_head_cnn_bilstm.py", "optuna_optimize_MultiHeadCNNBiLSTM"),
    "multi_head_cnn_resbilstm": ("optimize_multi_head_cnn_resbilstm.py", "optuna_optimize_MultiHeadCNNResBiLSTM"),
}


def run_one(key: str, script_name: str) -> int:
    script_path = SCRIPTS_DIR / script_name
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{key}.log"

    print(f"\n{'=' * 60}")
    print(f"Running study: {key}  ({script_name})")
    print(f"Logging to: {log_path}")
    print(f"{'=' * 60}\n")

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        proc.wait()
    elapsed = time.time() - start
    print(f"\n[{key}] finished in {elapsed / 60:.1f} min with exit code {proc.returncode}")
    return proc.returncode


def read_best_f1(optuna_dir_name: str):
    results_path = SCRIPTS_DIR / optuna_dir_name / "results.json"
    if not results_path.exists():
        return None
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for _, result in data.items():
            return result.get("metrics", {}).get("f1_score")
    except (json.JSONDecodeError, OSError):
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(STUDIES.keys()), default=None,
                         help="Subset of studies to run (default: all)")
    parser.add_argument("--skip-existing", action="store_true",
                         help="Skip a model if studies/optuna_optimize_<Model>/results.json already exists")
    parser.add_argument("--stop-on-fail", action="store_true",
                         help="Stop the whole run as soon as one study fails (default: continue with the rest)")
    parser.add_argument("--list", action="store_true", help="List available model keys and exit")
    args = parser.parse_args()

    if args.list:
        for key in STUDIES:
            print(key)
        return

    selected = args.models or list(STUDIES.keys())

    outcomes = {}
    for key in selected:
        script_name, optuna_dir_name = STUDIES[key]
        results_path = SCRIPTS_DIR / optuna_dir_name / "results.json"
        if args.skip_existing and results_path.exists():
            print(f"[{key}] results.json already exists, skipping (use without --skip-existing to rerun)")
            outcomes[key] = "skipped"
            continue

        rc = run_one(key, script_name)
        outcomes[key] = "ok" if rc == 0 else f"failed (exit {rc})"
        if rc != 0 and args.stop_on_fail:
            print(f"\nStopping early because {key} failed and --stop-on-fail was set.")
            break

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for key in selected:
        if key not in outcomes:
            print(f"  {key:30s} not run")
            continue
        _, optuna_dir_name = STUDIES[key]
        f1 = read_best_f1(optuna_dir_name)
        f1_str = f"F1={f1:.4f}" if f1 is not None else "F1=n/a"
        print(f"  {key:30s} {outcomes[key]:20s} {f1_str}")
    print()


if __name__ == "__main__":
    main()
