#!/usr/bin/env python3
"""Email the report for a sweep that is *already running*.

`run_all_studies.py` resolves its mail settings at startup, so a sweep launched before
studies/notify_credentials.env existed cannot send anything - the running process is
holding an empty configuration and there is no way to hand it new credentials.

This waits for that process to exit and then mails the report from what it wrote to
disk, which is the same content the sweep would have sent itself.

    python studies/notify_running_sweep.py <pid of run_all_studies.py>

One-off recovery. Sweeps started after the credentials exist need none of this.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from studies.notify import get_config, machine_line, send_email
from studies.run_all_studies import format_report
from studies.study_runner import RESULTS_ROOT

POLL_SECONDS = 30


def still_running(pid: int) -> bool:
    """Signal 0 tests for existence without touching the process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def process_started_at(pid: int) -> datetime:
    """Read the start time from ps, so the mail can report the real duration."""
    import subprocess
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return datetime.strptime(out, "%a %b %d %H:%M:%S %Y")
    except (ValueError, OSError, subprocess.SubprocessError):
        return datetime.now()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pid", type=int, help="pid of the running run_all_studies.py")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()

    started_at = process_started_at(args.pid) if still_running(args.pid) else datetime.now()

    print(f"Watching pid {args.pid} (started {started_at:%Y-%m-%d %H:%M:%S}); "
          f"will email when it exits.", flush=True)
    while still_running(args.pid):
        time.sleep(POLL_SECONDS)

    # The driver writes the aggregate as its last act; give the filesystem a moment.
    time.sleep(5)
    finished_at = datetime.now()
    elapsed_minutes = (finished_at - started_at).total_seconds() / 60

    summary_path = args.results_root / "all_models_summary.json"
    if not summary_path.exists():
        subject = "[Exercise recognition] sweep ended without writing a summary"
        body = (f"pid {args.pid} exited at {finished_at:%Y-%m-%d %H:%M:%S} after "
                f"{elapsed_minutes:.1f} min, but {summary_path} was never written.\n"
                f"Check studies/logs/ for what happened.\n\nWhere {machine_line()}")
        send_email(subject, body, config=get_config())
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    outcomes = payload.get("outcomes", {})
    rows = payload.get("models", [])
    keys = list(outcomes) or [r["model_key"] for r in rows]
    ok_count = sum(1 for key in keys if outcomes.get(key) == "ok")

    report = format_report(keys, outcomes, rows, args.results_root)
    print(report, flush=True)

    subject = (f"[Exercise recognition] sweep finished - {ok_count}/{len(keys)} models ok, "
               f"{elapsed_minutes:.0f} min")
    body = "\n".join([
        f"Started  {started_at:%Y-%m-%d %H:%M:%S}",
        f"Finished {finished_at:%Y-%m-%d %H:%M:%S}  ({elapsed_minutes:.1f} min)",
        f"Where    {machine_line()}",
        "",
        "Outcomes:",
        *[f"  {key:<28} {outcomes.get(key, 'not run')}" for key in keys],
        "",
        report,
        "",
        f"Results  {args.results_root}",
        f"Logs     {Path(__file__).resolve().parent / 'logs'}",
    ])
    send_email(subject, body, config=get_config())


if __name__ == "__main__":
    main()
