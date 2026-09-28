#!/usr/bin/env python3
"""Run the full study (search + repeated evaluation) for a single model.

This is the worker `run_all_studies.py` launches per model, and it is also the way to
rerun one model on its own:

    python studies/run_model.py cnn_bilstm
    python studies/run_model.py cnn_bilstm --n-trials 25 --n-repeats 10
    python studies/run_model.py cnn_bilstm --email      # mail the result when it ends

Unlike `run_all_studies.py`, the email here is opt-in: the sweep runs six of these as
subprocesses and mails once at the end, which is one message instead of seven.
"""
import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from studies.model_registry import MODEL_KEYS, MODELS
from studies.notify import DEFAULT_RECIPIENT, check_config, get_config, machine_line, send_email
from studies.study_runner import (
    METRIC_KEYS,
    N_REPEATS,
    N_TRIALS,
    RESULTS_ROOT,
    TOP_K,
    run_model_study,
)


def compose_email(
    key: str,
    args: argparse.Namespace,
    started_at: datetime,
    result: Optional[Dict[str, Any]] = None,
    failure: Optional[BaseException] = None,
) -> tuple:
    """Subject and body for a single model's notification."""
    display_name = MODELS[key].display_name
    finished_at = datetime.now()

    if failure is not None:
        subject = f"[Exercise recognition] {key} FAILED - {type(failure).__name__}"
    else:
        selected = result["selected"]
        f1 = selected["test_metrics_mean"]["f1_score"]
        subject = f"[Exercise recognition] {key} finished - test F1 {f1:.4f}"

    body = [
        f"Model    {display_name} ({key})",
        f"Started  {started_at:%Y-%m-%d %H:%M:%S}",
        f"Finished {finished_at:%Y-%m-%d %H:%M:%S}",
        f"Where    {machine_line()}",
        f"Protocol {args.n_trials} search trials -> top {args.top_k} configs x {args.n_repeats} independent runs",
        "",
    ]

    if failure is not None:
        body += [
            "The run raised:",
            "",
            "".join(traceback.format_exception(type(failure), failure, failure.__traceback__)).rstrip(),
        ]
    else:
        selected = result["selected"]
        body += [
            f"Ran in {result['total_seconds'] / 60:.1f} min",
            "",
            "Selected configuration: "
            + ", ".join(f"{k}={v}" for k, v in selected["params"].items())
            + f"  ({selected['param_count']:,} parameters)",
            f"Validation F1  {selected['val_f1_mean']:.4f} +/- {selected['val_f1_std']:.4f}",
            "",
            f"Test scores, mean +/- std over {args.n_repeats} runs:",
        ]
        body += [
            f"  {metric:<12} {selected['test_metrics_mean'][metric]:.4f} "
            f"+/- {selected['test_metrics_std'][metric]:.4f}"
            for metric in METRIC_KEYS
        ]
        body += ["", f"Results  {args.results_root / key / 'results.json'}"]

    return subject, "\n".join(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=MODEL_KEYS, help="Which model to run")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS,
                        help=f"Optuna trials in the search stage (default: {N_TRIALS})")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Configurations carried into the repeated stage (default: {TOP_K})")
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS,
                        help=f"Independent runs per configuration (default: {N_REPEATS})")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT,
                        help=f"Where to write results (default: {RESULTS_ROOT})")
    parser.add_argument("--fold", type=int, default=None, metavar="I",
                        help="Cross-validation fold index to run (default: none, i.e. the "
                             "single ratio-balanced split). With --fold the search is redone "
                             "against that fold's validation subjects, so its test subjects "
                             "stay unseen by hyperparameter selection.")
    parser.add_argument("--n-folds", type=int, default=5, metavar="N",
                        help="Total number of cross-validation folds (default: 5)")
    parser.add_argument("--email", action="store_true",
                        help="Email the result when this model finishes (default: off; "
                             "run_all_studies.py sends one mail for the whole sweep instead)")
    parser.add_argument("--email-to", default=None, metavar="ADDRESS",
                        help=f"Recipient, implies --email (default: {DEFAULT_RECIPIENT})")
    return parser


def main():
    args = build_parser().parse_args()

    if args.email_to:
        args.email = True

    # Verify the mail settings before training rather than hours later.
    mail_config = None
    if args.email:
        mail_config = get_config()
        if args.email_to:
            mail_config["to"] = args.email_to
        _, reason = check_config(mail_config)
        print(f"\n{reason}")

    started_at = datetime.now()
    result, failure = None, None
    try:
        result = run_model_study(
            key=args.model,
            n_trials=args.n_trials,
            top_k=args.top_k,
            seeds=list(range(args.n_repeats)),
            results_root=args.results_root,
            fold=args.fold,
            n_folds=args.n_folds,
        )
    except Exception as error:  # noqa: BLE001 - re-raised below, after the mail goes out
        failure = error
        traceback.print_exc()

    if args.email:
        subject, body = compose_email(args.model, args, started_at, result, failure)
        send_email(subject, body, config=mail_config)

    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
