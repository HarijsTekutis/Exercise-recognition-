#!/usr/bin/env python3
"""Wait for the cross-validation driver to exit, then email the final report.

`cross_validate.py` writes cv_summary.json itself; this only watches for it to finish and
mails the result, so the sweep can be left unattended. It is a separate process on purpose
- the driver was already running by the time notification was wanted, and restarting an
18-hour job to add a mail step would have cost more than it saved.

    python studies/cv_watcher.py <driver_pid>
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from studies.notify import machine_line, send_email

POLL_SECONDS = 60
SUMMARY = Path(__file__).resolve().parent / "cv_results" / "cv_summary.json"


def still_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def compose(started: datetime) -> tuple:
    finished = datetime.now()
    head = [
        f"Nested subject-wise cross-validation finished",
        f"Started  {started:%Y-%m-%d %H:%M:%S}",
        f"Finished {finished:%Y-%m-%d %H:%M:%S}",
        f"Elapsed  {(finished - started).total_seconds() / 3600:.1f} h",
        f"Where    {machine_line()}",
        "",
    ]

    if not SUMMARY.exists():
        return ("[Exercise recognition] CV finished - NO SUMMARY FOUND",
                "\n".join(head + [f"Expected {SUMMARY} but it does not exist.",
                                  "The driver likely crashed; check studies/logs/cv_driver.log."]))

    data = json.load(open(SUMMARY, encoding="utf-8"))
    rows = data.get("models", [])
    outcomes = data.get("outcomes", {})
    failed = [k for k, v in outcomes.items() if "fail" in str(v).lower() or "abort" in str(v).lower()]

    body = head + [
        f"{'model':<26} {'test F1 across folds':<24} {'seed std in fold':<18} folds  hyperparams",
        "-" * 92,
    ]
    for r in rows:
        body.append(
            f"{r['display_name']:<26} "
            f"{r['test_f1_mean']:.4f} +/- {r['test_f1_std_across_folds']:.4f}         "
            f"{r['mean_seed_std_within_fold']:.4f}            "
            f"{r['n_folds']:<6} {'stable' if r['hyperparams_stable'] else 'VARY BY FOLD'}"
        )

    body += ["", "Per-fold test F1:"]
    for key, det in data.get("details", {}).items():
        vals = det["across_folds"]["f1_score"]["per_fold"]
        body.append(f"  {det['display_name']:<26} " + "  ".join(f"{v:.4f}" for v in vals))

    if failed:
        body += ["", f"FAILED/INCOMPLETE JOBS ({len(failed)}):"] + [f"  {k}" for k in failed]
        body += ["", "Rerun the same command to resume - completed jobs are skipped."]
    else:
        body += ["", f"All {len(outcomes)} jobs completed."]

    body += ["", f"Results  {SUMMARY.parent}"]

    top = rows[0]["display_name"] if rows else "n/a"
    f1 = f"{rows[0]['test_f1_mean']:.4f}" if rows else "n/a"
    subject = f"[Exercise recognition] Cross-validation done - best {top} F1 {f1}"
    if failed:
        subject += f" ({len(failed)} jobs failed)"
    return subject, "\n".join(body)


def main():
    pid = int(sys.argv[1])
    started = datetime.now()
    print(f"[cv_watcher] watching pid {pid}, will email when it exits", flush=True)
    while still_running(pid):
        time.sleep(POLL_SECONDS)
    # The driver writes cv_summary.json as its last act; give the filesystem a moment.
    time.sleep(15)
    subject, body = compose(started)
    print(f"[cv_watcher] driver exited, sending: {subject}", flush=True)
    sent = send_email(subject, body)
    print(f"[cv_watcher] email sent: {sent}", flush=True)


if __name__ == "__main__":
    main()
