#!/usr/bin/env python3
"""Nested subject-wise cross-validation: the full study, repeated on every fold.

What this answers
-----------------
`run_all_studies.py` reports how each model does on one fixed pair of held-out subjects.
Its error bars come from re-seeding, so they describe "if I retrain, how much does the
score wiggle" - not "what if I had held out different people". The evidence that the
second question matters is in the single-split results themselves: test F1 came out
*higher* than validation F1 for all six models, because the two test subjects simply
happened to be easier than the two validation subjects.

This script rotates the held-out subjects. The 10 subjects are partitioned into `n_folds`
disjoint groups; fold i tests on group i and validates on group i+1, so every subject is
tested exactly once and validated exactly once. The spread across folds is the part of the
uncertainty a single split cannot show, and it is normally several times larger than the
seed-to-seed spread.

Why it is "nested"
------------------
The Optuna search is re-run inside every fold, scored only against that fold's validation
subjects. The fold's test subjects are therefore unseen by training, by early stopping and
by hyperparameter selection alike. Reusing the hyperparameters picked on the original split
would be cheaper by 5x but would carry a little knowledge of that split into every fold.

Cost: n_folds x models x (n_trials + top_k * n_repeats) trainings.
At the defaults that is 5 x 6 x 30 = 900 runs, roughly 18-20 hours on one RTX 4070.

Resuming
--------
Every (fold, model) job writes results.json when it completes, and a job whose
results.json already exists is skipped. So an interrupted sweep can simply be relaunched
with the same command and it picks up where it stopped - important for a run this long.
Optuna's per-job study.db resumes a partially finished search too.

Usage:
    python studies/cross_validate.py                          # full nested CV
    python studies/cross_validate.py --max-parallel 3          # recommended
    python studies/cross_validate.py --folds 0 1               # only some folds
    python studies/cross_validate.py --models cnn_resbigru
    python studies/cross_validate.py --aggregate-only          # rebuild the summary
    python studies/cross_validate.py --dry-run                 # list the jobs

Results:
    studies/cv_results/fold<i>/<model>/results.json   one full study per fold+model
    studies/cv_results/cv_summary.json                per-model mean +/- std across folds
    studies/cv_results/cv_summary.csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_pipeline as dp
from studies.model_registry import CONFIG, MODEL_KEYS, MODELS
from studies.study_runner import METRIC_KEYS, N_REPEATS, N_TRIALS, TOP_K

STUDIES_DIR = Path(__file__).resolve().parent
ROOT = STUDIES_DIR.parent
CV_RESULTS_ROOT = STUDIES_DIR / "cv_results"
CV_LOGS_DIR = STUDIES_DIR / "logs" / "cv"

LAUNCH_STAGGER_SECONDS = 10
POLL_SECONDS = 5


def job_dir(results_root: Path, fold: int, key: str) -> Path:
    return results_root / f"fold{fold}" / key


def is_done(results_root: Path, fold: int, key: str) -> bool:
    return (job_dir(results_root, fold, key) / "results.json").exists()


def build_command(fold: int, key: str, args: argparse.Namespace) -> List[str]:
    return [
        sys.executable, str(STUDIES_DIR / "run_model.py"), key,
        "--n-trials", str(args.n_trials),
        "--top-k", str(args.top_k),
        "--n-repeats", str(args.n_repeats),
        "--fold", str(fold),
        "--n-folds", str(args.n_folds),
        "--results-root", str(results_root_for_fold(args.results_root, fold)),
    ]


def results_root_for_fold(results_root: Path, fold: int) -> Path:
    # run_model.py writes <results-root>/<model>/, so point it at the fold directory.
    return results_root / f"fold{fold}"


# --------------------------------------------------------------------------------------
# Running the jobs
# --------------------------------------------------------------------------------------
def run_pool(jobs: List[Tuple[int, str]], args: argparse.Namespace) -> Dict[str, str]:
    """Run (fold, model) jobs, up to args.max_parallel at a time."""
    CV_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    pending = list(jobs)
    running: Dict[str, Any] = {}
    outcomes: Dict[str, str] = {}
    aborted: List[Tuple[int, str]] = []
    finished_count = 0
    total = len(jobs)
    sweep_started = time.time()

    while pending or running:
        while pending and len(running) < args.max_parallel:
            fold, key = pending.pop(0)
            name = f"fold{fold}/{key}"
            log_path = CV_LOGS_DIR / f"fold{fold}_{key}.log"
            handle = open(log_path, "w", encoding="utf-8")
            process = subprocess.Popen(
                build_command(fold, key, args), cwd=ROOT,
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
            running[name] = (process, handle, time.time())
            print(f"[start ] {name:<34} pid={process.pid}", flush=True)
            if pending and len(running) < args.max_parallel:
                time.sleep(LAUNCH_STAGGER_SECONDS)

        done_now = [n for n, (p, _, _) in running.items() if p.poll() is not None]
        if not done_now:
            time.sleep(POLL_SECONDS)
            continue

        for name in done_now:
            process, handle, started = running.pop(name)
            handle.close()
            code = process.returncode
            outcomes[name] = "ok" if code == 0 else f"failed (exit {code})"
            finished_count += 1
            elapsed = time.time() - sweep_started
            # Straight-line estimate; good enough to know whether to wait up for it.
            eta = (elapsed / finished_count) * (total - finished_count) if finished_count else 0
            print(
                f"[{'done  ' if code == 0 else 'FAILED'}] {name:<34} "
                f"{(time.time() - started) / 60:5.1f} min   "
                f"{finished_count}/{total} complete, {len(running)} running, "
                f"{len(pending)} queued, ETA {eta / 3600:.1f} h",
                flush=True,
            )
            if code != 0 and args.stop_on_fail:
                aborted.extend(pending)
                pending = []
                print(f"\n--stop-on-fail: {name} failed; draining {len(running)} running job(s).", flush=True)

    for fold, key in aborted:
        outcomes[f"fold{fold}/{key}"] = "not run (aborted)"
    return outcomes


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def collect(args: argparse.Namespace) -> Dict[str, Any]:
    """Gather every finished fold into per-model across-fold statistics."""
    per_model: Dict[str, Dict[str, Any]] = {}

    for key in args.models:
        folds = []
        for fold in args.folds:
            path = job_dir(args.results_root, fold, key) / "results.json"
            if not path.exists():
                continue
            r = json.load(open(path, encoding="utf-8"))
            sel = r["selected"]
            folds.append({
                "fold": fold,
                "subjects": r["split"]["subjects"],
                "params": sel["params"],
                "param_count": sel["param_count"],
                "val_f1_mean": sel["val_f1_mean"],
                "test_metrics_mean": sel["test_metrics_mean"],
                "test_metrics_std": sel["test_metrics_std"],
                "seed_f1_values": next(
                    c["test_summary"]["f1_score"]["values"]
                    for c in r["configurations"] if c["params"] == sel["params"]
                ),
            })
        if not folds:
            continue

        across = {}
        for metric in METRIC_KEYS:
            vals = np.array([f["test_metrics_mean"][metric] for f in folds], dtype=float)
            across[metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()), "max": float(vals.max()),
                "per_fold": [float(v) for v in vals],
            }

        # Two sources of spread, reported separately: re-seeding within a fold, versus
        # which subjects the fold held out. The second is the one a single split hides.
        within = float(np.mean([f["test_metrics_std"]["f1_score"] for f in folds]))
        between = across["f1_score"]["std"]
        chosen = [tuple(sorted(f["params"].items())) for f in folds]
        per_model[key] = {
            "display_name": MODELS[key].display_name,
            "n_folds_done": len(folds),
            "across_folds": across,
            "mean_within_fold_seed_std": within,
            "between_fold_std": between,
            "seed_vs_split_ratio": (between / within) if within > 0 else None,
            "hyperparameters_stable_across_folds": len(set(chosen)) == 1,
            "params_per_fold": {f["fold"]: f["params"] for f in folds},
            "folds": folds,
        }

    return per_model


def write_summary(per_model: Dict[str, Any], args: argparse.Namespace, outcomes: Dict[str, str]) -> List[Dict[str, Any]]:
    rows = []
    for key, m in per_model.items():
        row = {
            "model_key": key,
            "display_name": m["display_name"],
            "n_folds": m["n_folds_done"],
            "test_f1_mean": m["across_folds"]["f1_score"]["mean"],
            "test_f1_std_across_folds": m["between_fold_std"],
            "mean_seed_std_within_fold": m["mean_within_fold_seed_std"],
            "hyperparams_stable": m["hyperparameters_stable_across_folds"],
        }
        for metric in METRIC_KEYS:
            row[f"test_{metric}_mean"] = m["across_folds"][metric]["mean"]
            row[f"test_{metric}_std"] = m["across_folds"][metric]["std"]
        rows.append(row)
    rows.sort(key=lambda r: r["test_f1_mean"], reverse=True)

    args.results_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": (
            f"Nested subject-wise cross-validation over {args.n_folds} folds. Each fold "
            f"partitions the subjects so that its test group is unseen by training, early "
            f"stopping and hyperparameter search alike; the Optuna search "
            f"({args.n_trials} trials) is repeated inside every fold, its top {args.top_k} "
            f"configurations retrained {args.n_repeats} times each, and the configuration "
            f"with the best mean validation F1 selected. Reported std is ACROSS FOLDS, so "
            f"it includes the effect of which subjects were held out."
        ),
        "config": dict(CONFIG),
        "outcomes": outcomes,
        "models": rows,
        "details": per_model,
    }
    with open(args.results_root / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if rows:
        cols = [c for c in rows[0]]
        with open(args.results_root / "cv_summary.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return rows


def print_report(rows: List[Dict[str, Any]], per_model: Dict[str, Any], args: argparse.Namespace) -> None:
    print(f"\n\n{'=' * 104}")
    print("CROSS-VALIDATED RESULTS - test macro F1, mean +/- std ACROSS FOLDS (includes split variance)")
    print(f"{'=' * 104}")
    print(f"{'model':<26} {'test F1 (across folds)':<26} {'seed std within fold':<22} {'folds':<7} hyperparams stable")
    print("-" * 104)
    for r in rows:
        print(f"{r['display_name']:<26} "
              f"{r['test_f1_mean']:.4f} +/- {r['test_f1_std_across_folds']:.4f}        "
              f"{r['mean_seed_std_within_fold']:.4f}                "
              f"{r['n_folds']:<7} {'yes' if r['hyperparams_stable'] else 'NO - varies by fold'}")

    print(f"\nPer-fold test F1:")
    for key, m in per_model.items():
        vals = m["across_folds"]["f1_score"]["per_fold"]
        print(f"  {m['display_name']:<26} " + "  ".join(f"{v:.4f}" for v in vals))

    print(f"\nHow much bigger is split variance than seed variance?")
    for key, m in per_model.items():
        ratio = m["seed_vs_split_ratio"]
        print(f"  {m['display_name']:<26} between-fold/within-fold std = "
              f"{ratio:.1f}x" if ratio else f"  {m['display_name']:<26} n/a")
    print(f"\nSaved to {args.results_root / 'cv_summary.json'} and cv_summary.csv")
    print(f"{'=' * 104}\n")


# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=None)
    p.add_argument("--folds", nargs="+", type=int, default=None, help="Fold indices (default: all)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--n-repeats", type=int, default=N_REPEATS)
    p.add_argument("--max-parallel", type=int, default=1, metavar="N")
    p.add_argument("--results-root", type=Path, default=CV_RESULTS_ROOT)
    p.add_argument("--rerun-existing", action="store_true",
                   help="Redo jobs that already have results.json (default: skip them, so an "
                        "interrupted sweep resumes)")
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument("--aggregate-only", action="store_true", help="Only rebuild the summary")
    p.add_argument("--dry-run", action="store_true", help="List the jobs and exit")
    return p


def main():
    args = build_parser().parse_args()
    args.models = args.models or list(MODEL_KEYS)
    args.folds = args.folds if args.folds is not None else list(range(args.n_folds))

    for fold in args.folds:
        if not 0 <= fold < args.n_folds:
            build_parser().error(f"--folds value {fold} outside [0, {args.n_folds})")

    jobs = [(f, k) for f in args.folds for k in args.models]
    done = [(f, k) for f, k in jobs if is_done(args.results_root, f, k)]
    todo = jobs if args.rerun_existing else [j for j in jobs if j not in done]

    runs_per_job = args.n_trials + args.top_k * args.n_repeats
    print(f"Nested subject-wise CV: {len(args.folds)} folds x {len(args.models)} models")
    print(f"Each job: {args.n_trials} search trials + {args.top_k} configs x {args.n_repeats} seeds "
          f"= {runs_per_job} trainings")
    print(f"Jobs: {len(jobs)} total, {len(done)} already complete, {len(todo)} to run "
          f"-> {len(todo) * runs_per_job} trainings")

    # Show which subjects each fold holds out, so the design is visible before it starts.
    try:
        bundle_data = dp.load_filtered_recordings(
            data_path=str(ROOT / CONFIG["data_path"]),
            min_recordings_per_activity=CONFIG["min_recordings_per_activity"])
        subs = dp.get_session_subjects(bundle_data)
        groups = dp.subject_cv_folds(subs, n_folds=args.n_folds, seed=CONFIG["split_seed"])
        print("\nFold layout:")
        for i in range(args.n_folds):
            te, va = groups[i], groups[(i + 1) % args.n_folds]
            tr = [s for g in groups for s in g if s not in te and s not in va]
            mark = " " if i in args.folds else " (skipped)"
            print(f"  fold {i}: train={sorted(tr)}  val={sorted(va)}  test={sorted(te)}{mark}")
    except Exception as error:  # noqa: BLE001 - purely informational
        print(f"  (could not preview folds: {error})")

    if args.dry_run:
        print("\nJobs that would run:")
        for f, k in todo:
            print(f"  fold{f}/{k}")
        return

    outcomes = {f"fold{f}/{k}": "already complete" for f, k in done}
    if not args.aggregate_only and todo:
        started = time.time()
        outcomes.update(run_pool(todo, args))
        print(f"\nAll jobs finished in {(time.time() - started) / 3600:.2f} h")
    elif not todo:
        print("\nNothing to run - every job already has results.json")

    per_model = collect(args)
    rows = write_summary(per_model, args, outcomes)
    if rows:
        print_report(rows, per_model, args)
    else:
        print("\nNo finished folds yet, nothing to summarise.")


if __name__ == "__main__":
    main()
