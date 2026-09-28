#!/usr/bin/env python3
"""Wait for every running cross_validate.py to finish, rebuild the combined summary, email it.

Two cross-validation runs are in flight at once (the classical baselines, and Rocket on a
reduced seed budget). Each writes cv_summary.json covering only the models it was given,
so whichever finishes last would otherwise leave a partial summary on disk. This waits for
both, re-aggregates across all nine models, and mails the result.

    python studies/finalize_cv.py
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

STUDIES_DIR = Path(__file__).resolve().parent
ROOT = STUDIES_DIR.parent
sys.path.append(str(ROOT))

from studies.notify import machine_line, send_email

POLL_SECONDS = 60
SUMMARY = STUDIES_DIR / "cv_results" / "cv_summary.json"


def runs_in_flight() -> int:
    """Count live cross_validate.py interpreter processes.

    Matching on the raw command line alone also catches the shell wrappers that launched
    them (their text contains both "python" and the script name), and a lingering wrapper
    would make this wait forever. So require the *executable* itself to be a python and
    the script to appear as its own argument.
    """
    out = subprocess.run(["ps", "-eo", "comm:32,args"], capture_output=True, text=True).stdout
    count = 0
    for line in out.splitlines()[1:]:
        comm, _, args = line.partition(" ")
        if not comm.startswith("python"):
            continue
        argv = args.split()
        if any(a.endswith("cross_validate.py") for a in argv):
            count += 1
    return count


def main():
    started = datetime.now()
    print(f"[finalize] waiting for cross-validation runs to finish ({runs_in_flight()} in flight)", flush=True)
    while runs_in_flight() > 0:
        time.sleep(POLL_SECONDS)

    print("[finalize] all runs done; rebuilding the combined summary across all models", flush=True)
    proc = subprocess.run(
        [sys.executable, str(STUDIES_DIR / "cross_validate.py"), "--aggregate-only"],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-2000:], flush=True)

    finished = datetime.now()
    body = [
        "Cross-validation complete - all models, including the classical baselines.",
        f"Started  {started:%Y-%m-%d %H:%M:%S}",
        f"Finished {finished:%Y-%m-%d %H:%M:%S}",
        f"Where    {machine_line()}",
        "",
    ]

    subject = "[Exercise recognition] Cross-validation complete (all models)"
    if SUMMARY.exists():
        data = json.load(open(SUMMARY, encoding="utf-8"))
        rows = data.get("models", [])
        body += [
            f"{'model':<26} {'test F1 across folds':<24} {'seed sd in fold':<17} folds",
            "-" * 80,
        ]
        for r in rows:
            body.append(
                f"{r['display_name']:<26} "
                f"{r['test_f1_mean']:.4f} +/- {r['test_f1_std_across_folds']:.4f}         "
                f"{r['mean_seed_std_within_fold']:.4f}           {r['n_folds']}"
            )
        body += ["", "Per-fold test F1:"]
        for key, det in data.get("details", {}).items():
            vals = det["across_folds"]["f1_score"]["per_fold"]
            body.append(f"  {det['display_name']:<26} " + "  ".join(f"{v:.4f}" for v in vals))
        if rows:
            subject = (f"[Exercise recognition] CV complete - best {rows[0]['display_name']} "
                       f"F1 {rows[0]['test_f1_mean']:.4f}")
        failed = [k for k, v in data.get("outcomes", {}).items()
                  if "fail" in str(v).lower() or "abort" in str(v).lower()]
        if failed:
            body += ["", f"FAILED JOBS ({len(failed)}):"] + [f"  {k}" for k in failed]
            body += ["", "Rerun cross_validate.py with the same arguments to resume."]
    else:
        body.append(f"No summary found at {SUMMARY} - check studies/logs/.")

    body += ["", f"Results  {SUMMARY.parent}"]
    print(f"[finalize] emailing: {subject}", flush=True)
    sent = send_email(subject, "\n".join(body))
    print(f"[finalize] sent: {sent}", flush=True)


if __name__ == "__main__":
    main()
