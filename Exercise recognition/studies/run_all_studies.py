#!/usr/bin/env python3
"""Train and evaluate every neural network model end to end.

For each of the six architectures this runs a two-stage protocol:

  Stage 1 - Optuna searches the hyperparameter space, scoring each trial by validation
            macro F1 from a single training run.
  Stage 2 - the top three configurations the search found are each retrained five times
            from independent seeds, and reported as mean +/- standard deviation.

Stage 2 exists because the search score is not reproducible on its own: despite a fixed
manual seed, repeating a run with identical hyperparameters gave slightly different
scores (non-deterministic cuDNN kernels in the recurrent layers, whose accumulated
floating point differences change which epoch early stopping picks). Ranking
configurations on one noisy run each is not an objective evaluation, so the search is
treated as a shortlisting step and the five-run mean is what gets reported.

The train/val/test split is subject independent and identical in every run of every
model, so nothing in the reported spread comes from a different split.

Each model runs in its own subprocess, so a CUDA OOM or a crash in one architecture does
not take down the rest of the sweep. Full output per model goes to studies/logs/.

`--max-parallel N` trains N models at once. One model on its own does not saturate the
GPU - the per-epoch validation pass is dominated by kernel-launch latency rather than
arithmetic - so several workers overlap nearly for free.

When the sweep ends - finished or crashed - the final report is emailed. That needs SMTP
credentials in the environment or in studies/notify_credentials.env; see studies/notify.py
for the two variables involved. Without them the run just prints a warning at startup and
proceeds normally. `--no-email` turns it off.

Usage:
    python studies/run_all_studies.py                       # all six, one at a time
    python studies/run_all_studies.py --max-parallel 3       # three at a time
    python studies/run_all_studies.py --models cnn_bilstm cnn_bigru
    python studies/run_all_studies.py --n-trials 25 --n-repeats 10
    python studies/run_all_studies.py --skip-existing        # skip models already finished
    python studies/run_all_studies.py --in-process           # no subprocess isolation
    python studies/run_all_studies.py --stop-on-fail
    python studies/run_all_studies.py --no-email
    python studies/run_all_studies.py --email-to someone@example.com
    python studies/run_all_studies.py --list

Results (per model, under studies/results/<model_key>/):
    study.db            Optuna storage, resumable
    search_trials.json  every trial with its parameters and validation score
    top_configs.json    the three configurations carried into stage 2
    results.json        everything: per-run metrics, histories, confusion matrices
    summary.json        the same minus the bulky arrays, for reading by hand
    checkpoints/        config<i>_seed<s>.pt for every run

Aggregated across models, under studies/results/:
    all_models_summary.json
    all_models_summary.csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from studies.model_registry import MODEL_KEYS, MODELS
from studies.notify import DEFAULT_RECIPIENT, check_config, get_config, machine_line, send_email
from studies.study_runner import METRIC_KEYS, N_REPEATS, N_TRIALS, RESULTS_ROOT, TOP_K

STUDIES_DIR = Path(__file__).resolve().parent
ROOT = STUDIES_DIR.parent
LOGS_DIR = STUDIES_DIR / "logs"


# How long to wait between launching concurrent workers. Each one reads the 227 MB csv
# and creates a CUDA context on the way up; starting them simultaneously just makes them
# contend for disk and GPU init with nothing to show for it.
LAUNCH_STAGGER_SECONDS = 10
# How often the pool checks whether a worker has exited. Runs last hours, so a coarse
# poll costs nothing and keeps the driver off the CPU.
POLL_SECONDS = 5


# --------------------------------------------------------------------------------------
# Running one model
# --------------------------------------------------------------------------------------
def build_command(key: str, args: argparse.Namespace) -> List[str]:
    return [
        sys.executable, str(STUDIES_DIR / "run_model.py"), key,
        "--n-trials", str(args.n_trials),
        "--top-k", str(args.top_k),
        "--n-repeats", str(args.n_repeats),
        "--results-root", str(args.results_root),
    ]


def run_in_subprocess(key: str, args: argparse.Namespace) -> int:
    """Launch studies/run_model.py for one model, teeing its output to a log file.

    Used when models run one at a time, so the training output is worth watching live.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{key}.log"

    command = build_command(key, args)

    print(f"\n{'=' * 78}")
    print(f"MODEL: {key}  ({MODELS[key].display_name})")
    print(f"  {args.n_trials} search trials -> top {args.top_k} configs x {args.n_repeats} runs")
    print(f"  log: {log_path}")
    print(f"{'=' * 78}\n")

    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()

    print(f"\n[{key}] exit code {process.returncode} after {(time.time() - started) / 60:.1f} min")
    return process.returncode


def run_pool(keys: List[str], args: argparse.Namespace) -> Dict[str, str]:
    """Keep up to `args.max_parallel` model subprocesses running at once.

    Worth doing because a single model does not come close to saturating the GPU: the
    per-epoch validation pass runs at batch_size_eval, and at small batch sizes the
    bottleneck is kernel-launch latency and the Python loop, not arithmetic. Several
    workers therefore interleave almost for free rather than competing.

    Memory is not the constraint - the windowed dataset is ~113 MB and one worker holds
    well under 1 GB of VRAM - so the useful ceiling is set by how much GPU work is
    actually outstanding, not by capacity.

    Unlike the sequential path, output is not teed to the console: six interleaved
    training logs are unreadable. Each worker writes straight to studies/logs/<key>.log.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    pending = list(keys)
    running: Dict[str, Any] = {}     # key -> (process, log handle, start time)
    outcomes: Dict[str, str] = {}
    aborted: List[str] = []

    print(f"\n{'=' * 78}")
    print(f"Running {len(pending)} model(s), up to {args.max_parallel} at a time")
    print(f"Logs: {LOGS_DIR}/<model>.log   (follow one with: tail -f {LOGS_DIR}/<model>.log)")
    print(f"{'=' * 78}\n")

    while pending or running:
        # Fill any free slot.
        while pending and len(running) < args.max_parallel:
            key = pending.pop(0)
            log_path = LOGS_DIR / f"{key}.log"
            handle = open(log_path, "w", encoding="utf-8")
            process = subprocess.Popen(
                build_command(key, args), cwd=ROOT,
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
            running[key] = (process, handle, time.time())
            print(f"[start ] {key:<26} pid={process.pid}  -> {log_path}")
            if pending and len(running) < args.max_parallel:
                time.sleep(LAUNCH_STAGGER_SECONDS)

        finished = [key for key, (process, _, _) in running.items() if process.poll() is not None]
        if not finished:
            time.sleep(POLL_SECONDS)
            continue

        for key in finished:
            process, handle, started = running.pop(key)
            handle.close()
            code = process.returncode
            outcomes[key] = "ok" if code == 0 else f"failed (exit {code})"
            label = "done  " if code == 0 else "FAILED"
            print(
                f"[{label}] {key:<26} {(time.time() - started) / 60:.1f} min"
                f"   ({len(running)} running, {len(pending)} queued)"
            )

            if code != 0 and args.stop_on_fail:
                # Let whatever is already training finish - killing it would throw away
                # hours of work - but stop feeding the pool.
                aborted.extend(pending)
                pending = []
                print(
                    f"\n--stop-on-fail: {key} failed. Not starting the {len(aborted)} queued "
                    f"model(s); waiting for {len(running)} still running."
                )

    for key in aborted:
        outcomes[key] = "not run (aborted after a failure)"

    return outcomes


def run_all_in_process(keys: List[str], args: argparse.Namespace) -> Dict[str, str]:
    """Run every model in this process, sharing one loaded copy of the dataset.

    Faster (the dataset is windowed once instead of once per model) but a crash in one
    model ends the whole sweep, so it is not the default.
    """
    from studies.study_runner import DataBundle, run_model_study

    bundle = DataBundle()
    outcomes: Dict[str, str] = {}
    for key in keys:
        try:
            run_model_study(
                key=key,
                bundle=bundle,
                n_trials=args.n_trials,
                top_k=args.top_k,
                seeds=list(range(args.n_repeats)),
                results_root=args.results_root,
            )
            outcomes[key] = "ok"
        except Exception as error:  # noqa: BLE001 - one model failing must not hide the others
            outcomes[key] = f"failed ({type(error).__name__}: {error})"
            print(f"\n[{key}] FAILED: {type(error).__name__}: {error}")
            if args.stop_on_fail:
                break
    return outcomes


# --------------------------------------------------------------------------------------
# Aggregating what the models wrote
# --------------------------------------------------------------------------------------
def load_summary(key: str, results_root: Path) -> Optional[Dict[str, Any]]:
    path = results_root / key / "summary.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_aggregate(keys: List[str], outcomes: Dict[str, str], results_root: Path) -> List[Dict[str, Any]]:
    """Collect every model's selected configuration into one json + csv."""
    rows: List[Dict[str, Any]] = []

    for key in keys:
        summary = load_summary(key, results_root)
        if summary is None:
            continue
        selected = summary["selected"]
        row: Dict[str, Any] = {
            "model_key": key,
            "display_name": summary["display_name"],
            "params": selected["params"],
            "param_count": selected["param_count"],
            "n_repeats": summary["protocol"]["n_repeats"],
            "val_f1_mean": selected["val_f1_mean"],
            "val_f1_std": selected["val_f1_std"],
        }
        for metric in METRIC_KEYS:
            row[f"test_{metric}_mean"] = selected["test_metrics_mean"][metric]
            row[f"test_{metric}_std"] = selected["test_metrics_std"][metric]
        rows.append(row)

    rows.sort(key=lambda r: r["test_f1_score_mean"], reverse=True)

    payload = {
        "protocol": (
            "Per model: Optuna search on validation macro F1, then the top three "
            "configurations each retrained over independent runs; the configuration with "
            "the best mean validation F1 is selected and its test scores reported as "
            "mean +/- std. Subject-independent split, identical for every run."
        ),
        "outcomes": outcomes,
        "models": rows,
        "per_model_details": {
            key: str((results_root / key / "results.json").resolve())
            for key in keys
            if (results_root / key / "results.json").exists()
        },
    }

    results_root.mkdir(parents=True, exist_ok=True)
    with open(results_root / "all_models_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if rows:
        csv_columns = [c for c in rows[0] if c != "params"] + ["params"]
        with open(results_root / "all_models_summary.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "params": json.dumps(row["params"])})

    return rows


def format_report(keys: List[str], outcomes: Dict[str, str], rows: List[Dict[str, Any]], results_root: Path) -> str:
    """Build the final table as text, so it can be both printed and emailed."""
    lines = [
        "=" * 100,
        "FINAL RESULTS - selected configuration per model, test scores as mean +/- std over repeated runs",
        "=" * 100,
        f"{'model':<26} {'params':<34} {'test F1':<18} {'test acc':<18} {'runs':<5}",
        "-" * 100,
    ]
    for row in rows:
        params = ", ".join(f"{k}={v}" for k, v in row["params"].items())
        f1 = f"{row['test_f1_score_mean']:.4f} +/- {row['test_f1_score_std']:.4f}"
        accuracy = f"{row['test_accuracy_mean']:.4f} +/- {row['test_accuracy_std']:.4f}"
        lines.append(
            f"{row['display_name']:<26} {params:<34} {f1:<18} {accuracy:<18} {row['n_repeats']:<5}"
        )

    missing = [key for key in keys if key not in {r["model_key"] for r in rows}]
    if missing:
        lines.append("")
        lines.append("No results for: " + ", ".join(f"{key} [{outcomes.get(key, 'not run')}]" for key in missing))

    lines.append("")
    lines.append(f"Saved to {results_root / 'all_models_summary.json'} and all_models_summary.csv")
    lines.append("=" * 100)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# The "it's done" email
# --------------------------------------------------------------------------------------
def compose_email(
    keys: List[str],
    outcomes: Dict[str, str],
    report: str,
    args: argparse.Namespace,
    started_at: datetime,
    elapsed_minutes: float,
    failure: Optional[BaseException] = None,
) -> Tuple[str, str]:
    """Subject and plain text body for the notification mail.

    Sent on failure too - a sweep that died after twenty minutes is exactly the thing
    worth hearing about before the evening is spent assuming it is still training.
    """
    finished_at = datetime.now()
    ok_count = sum(1 for key in keys if outcomes.get(key) == "ok")

    if failure is not None:
        subject = f"[Exercise recognition] sweep CRASHED after {elapsed_minutes:.0f} min - {type(failure).__name__}"
    else:
        subject = f"[Exercise recognition] sweep finished - {ok_count}/{len(keys)} models ok, {elapsed_minutes:.0f} min"

    body = [
        f"Started  {started_at:%Y-%m-%d %H:%M:%S}",
        f"Finished {finished_at:%Y-%m-%d %H:%M:%S}  ({elapsed_minutes:.1f} min)",
        f"Where    {machine_line()}",
        "",
        f"Models   {', '.join(keys)}",
        f"Protocol {args.n_trials} search trials -> top {args.top_k} configs x {args.n_repeats} independent runs",
        "",
        "Outcomes:",
    ]
    body += [f"  {key:<28} {outcomes.get(key, 'not run')}" for key in keys]

    if failure is not None:
        body += [
            "",
            "The sweep driver itself raised:",
            "",
            "".join(traceback.format_exception(type(failure), failure, failure.__traceback__)).rstrip(),
            "",
            "Any model that had already finished still wrote its results.",
        ]

    if report:
        body += ["", report]

    body += [
        "",
        f"Results  {args.results_root}",
        f"Logs     {LOGS_DIR}",
    ]
    return subject, "\n".join(body)


# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=None,
                        help="Subset of models to run (default: all)")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS,
                        help=f"Optuna trials per model (default: {N_TRIALS})")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Configurations carried into the repeated stage (default: {TOP_K})")
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS,
                        help=f"Independent runs per configuration (default: {N_REPEATS})")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT,
                        help=f"Where results are written (default: {RESULTS_ROOT})")
    parser.add_argument("--max-parallel", type=int, default=1, metavar="N",
                        help="Train up to N models concurrently, each in its own subprocess "
                             "(default: 1, i.e. sequential with live output). A single model "
                             "leaves the GPU far from saturated, so 3-4 is usually a large "
                             "wall-clock win. With N>1 output goes to studies/logs/ only.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a model that already has results.json under --results-root")
    parser.add_argument("--in-process", action="store_true",
                        help="Run models in this process instead of one subprocess each "
                             "(shares the loaded dataset, but loses crash isolation)")
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="Abort the sweep on the first failing model (default: continue)")
    parser.add_argument("--email", dest="email", action="store_true", default=True,
                        help="Email the final report when the sweep ends (default: on, if "
                             "NOTIFY_SMTP_USER/NOTIFY_SMTP_PASSWORD are configured)")
    parser.add_argument("--no-email", dest="email", action="store_false",
                        help="Do not send the notification email")
    parser.add_argument("--email-to", default=None, metavar="ADDRESS",
                        help=f"Recipient of the notification (default: {DEFAULT_RECIPIENT})")
    parser.add_argument("--list", action="store_true", help="List model keys and exit")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        for key in MODEL_KEYS:
            print(f"{key:<28} {MODELS[key].display_name}")
        return

    keys = args.models or list(MODEL_KEYS)

    if args.skip_existing:
        kept = []
        for key in keys:
            if (args.results_root / key / "results.json").exists():
                print(f"[{key}] results.json exists, skipping")
            else:
                kept.append(key)
        skipped = {key: "skipped" for key in keys if key not in kept}
    else:
        kept, skipped = keys, {}

    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")
    if args.in_process and args.max_parallel > 1:
        print("Note: --in-process runs models sequentially in one process; --max-parallel is ignored.")

    # Check the mail settings *now*, not in six hours. If the password was never set, the
    # run should say so while there is still someone watching the terminal.
    mail_config = None
    if args.email:
        mail_config = get_config()
        if args.email_to:
            mail_config["to"] = args.email_to
        _, reason = check_config(mail_config)
        print(f"\n{reason}")

    print(f"\nRunning {len(kept)} model(s): {', '.join(kept) or 'none'}")
    print(f"Protocol: {args.n_trials} search trials -> top {args.top_k} configs x {args.n_repeats} independent runs")

    started_at = datetime.now()
    sweep_started = time.time()
    outcomes: Dict[str, str] = {}
    failure: Optional[BaseException] = None

    try:
        if args.in_process:
            outcomes = run_all_in_process(kept, args)
        elif args.max_parallel > 1:
            outcomes = run_pool(kept, args)
        else:
            for key in kept:
                code = run_in_subprocess(key, args)
                outcomes[key] = "ok" if code == 0 else f"failed (exit {code})"
                if code != 0 and args.stop_on_fail:
                    print(f"\nStopping: {key} failed and --stop-on-fail was set.")
                    break
    except Exception as error:  # noqa: BLE001 - re-raised below, but the mail goes out first
        failure = error
        traceback.print_exc()

    outcomes.update(skipped)
    elapsed_minutes = (time.time() - sweep_started) / 60

    report = ""
    if failure is None:
        print(f"\nSweep finished in {elapsed_minutes:.1f} min")
        rows = write_aggregate(keys, outcomes, args.results_root)
        report = format_report(keys, outcomes, rows, args.results_root)
        print(f"\n\n{report}\n")

    if args.email:
        subject, body = compose_email(keys, outcomes, report, args, started_at, elapsed_minutes, failure)
        send_email(subject, body, config=mail_config)

    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
