"""Session-level bootstrap analysis for saved test predictions.

Performs session-resampling bootstrapping (resample whole recordings with replacement)
to compute 95% percentile CIs for accuracy, macro precision, macro recall and macro F1.

Why session-level resampling?
- Test windows overlap by 50%; resampling individual windows would break the
  dependence structure between windows and produce over-optimistic CIs. Resampling
  whole sessions (all windows from a recording) preserves within-session dependence.

Exports CSV and LaTeX tables with per-run CIs, aggregated means/stds, and
pairwise paired-session bootstrap comparisons against `CNN_ResBiGRU`.

Usage: run from repository root::

    python studies/analysis/session_bootstrap.py

"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple

import data_pipeline as dp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Analysis parameters
N_BOOTSTRAP = 5000
RNG_SEED = 12345


def windows_per_session(session_df_len: int, window_size: int, step_size: int) -> int:
    if session_df_len < window_size:
        return 0
    return (session_df_len - window_size) // step_size + 1


def compute_metrics(y_true: List[int], y_pred: List[int], labels: List[int]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def session_bootstrap(y_true: List[int], y_pred: List[int], session_boundaries: List[Tuple[int,int]], labels: List[int], rng) -> Dict[str, Tuple[float,float]]:
    """Resample sessions with replacement and compute percentile CIs.

    session_boundaries: list of (start_idx, end_idx) inclusive start, exclusive end
    """
    n_sessions = len(session_boundaries)
    boot_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    # Pre-extract per-session windows
    session_windows = []
    for (s,e) in session_boundaries:
        session_windows.append((y_true[s:e], y_pred[s:e]))

    for i in range(N_BOOTSTRAP):
        sel = rng.integers(0, n_sessions, size=n_sessions)
        boot_y_true = []
        boot_y_pred = []
        for idx in sel:
            yt, yp = session_windows[idx]
            boot_y_true.extend(yt)
            boot_y_pred.extend(yp)

        if len(boot_y_true) == 0:
            # no windows in bootstrap sample (rare if many tiny sessions); record NaNs
            for k in boot_metrics:
                boot_metrics[k].append(np.nan)
            continue

        m = compute_metrics(boot_y_true, boot_y_pred, labels)
        for k,v in m.items():
            boot_metrics[k].append(v)

    ci = {}
    for k, vals in boot_metrics.items():
        arr = np.array(vals)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            ci[k] = (np.nan, np.nan)
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            ci[k] = (float(lo), float(hi))
    return ci


def paired_session_bootstrap_diff(y_true_a, y_pred_a, y_true_b, y_pred_b, session_boundaries, labels, rng) -> Dict[str, object]:
    """Paired session-level bootstrap for macro F1 difference (A - B).

    Use same resampled sessions for both models in each iteration.
    Returns observed diff, 95% CI, approximate two-sided p-value and bootstrap distribution.
    """
    n_sessions = len(session_boundaries)
    session_windows_a = [(y_true_a[s:e], y_pred_a[s:e]) for (s,e) in session_boundaries]
    session_windows_b = [(y_true_b[s:e], y_pred_b[s:e]) for (s,e) in session_boundaries]

    # observed difference
    m_a = compute_metrics(y_true_a, y_pred_a, labels)
    m_b = compute_metrics(y_true_b, y_pred_b, labels)
    observed = m_a["f1"] - m_b["f1"]

    diffs = []
    for i in range(N_BOOTSTRAP):
        sel = rng.integers(0, n_sessions, size=n_sessions)
        boot_y_true_a = []
        boot_y_pred_a = []
        boot_y_true_b = []
        boot_y_pred_b = []
        for idx in sel:
            ya, pa = session_windows_a[idx]
            yb, pb = session_windows_b[idx]
            boot_y_true_a.extend(ya); boot_y_pred_a.extend(pa)
            boot_y_true_b.extend(yb); boot_y_pred_b.extend(pb)

        if len(boot_y_true_a) == 0 or len(boot_y_true_b) == 0:
            diffs.append(np.nan)
            continue

        fa = compute_metrics(boot_y_true_a, boot_y_pred_a, labels)["f1"]
        fb = compute_metrics(boot_y_true_b, boot_y_pred_b, labels)["f1"]
        diffs.append(fa - fb)

    arr = np.array(diffs)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        ci = (np.nan, np.nan)
        pval = np.nan
    else:
        ci = tuple(np.percentile(arr, [2.5, 97.5]).astype(float))
        # two-sided p-value: proportion of bootstrap diffs as or more extreme than 0
        p_less = np.mean(arr <= 0)
        p_greater = np.mean(arr >= 0)
        pval = float(2.0 * min(p_less, p_greater))

    return {
        "observed_diff": float(observed),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p_value": pval,
        "bootstrap_diffs": arr,
    }


def analyze_all(results_root: Path, out_root: Path):
    rng = np.random.default_rng(RNG_SEED)
    out_root.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])

    per_run_rows = []
    per_model_summary = {}

    # load dataset once
    data_path = Path("data2")

    for model_dir in model_dirs:
        res_json = model_dir / "results.json"
        if not res_json.exists():
            continue
        r = json.loads(res_json.read_text())
        model_key = r.get("display_name", model_dir.name)
        config = r.get("config", {})
        window_size = config.get("window_size", 40)
        step_size = config.get("step_size", 20)
        split_seed = config.get("split_seed", 42)

        # rebuild test sessions to find session boundaries
        recordings = dp.load_filtered_recordings(data_path=str(data_path))
        dp.encode_activities(recordings)
        splits = dp.make_train_val_test_loaders(
            data=recordings,
            imu_features=dp.IMU_FEATURES,
            window_size=window_size,
            step_size=step_size,
            batch_size_train=32,
            batch_size_val=1,
            batch_size_test=1,
            seed=split_seed,
        )

        test_sessions = splits.test_dataset

        # build session boundaries over the test_dataset ordering
        session_subjects = dp.get_session_subjects(recordings)
        _, _, test_indices = dp.subject_independent_split(session_subjects, train_ratio=0.6, val_ratio=0.2, seed=split_seed)
        test_recordings = [recordings[i] for i in test_indices]

        session_counts = []
        for sess in test_recordings:
            cnt = windows_per_session(len(sess), window_size, step_size)
            session_counts.append(cnt)

        # Build session boundaries
        boundaries = []
        start = 0
        for cnt in session_counts:
            end = start + cnt
            boundaries.append((start, end))
            start = end

        # Now extract per-run y_true/y_pred from results.json
        configurations = r.get("configurations", [])
        if not configurations:
            continue
        chosen = configurations[0]
        runs = chosen.get("runs", [])
        print(f"Processing model {model_key}, found {len(runs)} runs")

        f1s = []
        run_cis = []
        for run in runs:
            y_true = run.get("y_true") or r.get("y_true")
            y_pred = run.get("y_pred") or r.get("y_pred")
            if y_true is None or y_pred is None:
                continue
            # labels: fixed complete list of class ids
            labels = list(range(len(dp.encode_activities(recordings))))
            # compute point estimates
            metrics = compute_metrics(y_true, y_pred, labels)
            f1s.append(metrics["f1"])

            # bootstrap CIs
            ci = session_bootstrap(y_true, y_pred, boundaries, labels, rng)
            run_cis.append(ci)

            per_run_rows.append({
                "model": model_key,
                "seed": run.get("seed"),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f1_ci_low": ci["f1"][0],
                "f1_ci_high": ci["f1"][1],
            })
            print(f"  run seed={run.get('seed')}: f1={metrics['f1']:.4f}, ci=[{ci['f1'][0]:.4f},{ci['f1'][1]:.4f}]")

        # aggregate across runs
        if f1s:
            per_model_summary[model_key] = {
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s, ddof=0)),
                "run_cis": run_cis,
                "run_f1s": f1s,
            }

    # save per-run CSV
    per_run_df = pd.DataFrame(per_run_rows)
    per_run_df.to_csv(out_root / "per_run_session_bootstrap.csv", index=False)
    try:
        per_run_df.to_latex(out_root / "per_run_session_bootstrap.tex", index=False, float_format="%.4f")
    except Exception:
        pass

    # save per-model summary
    rows = []
    for model, s in per_model_summary.items():
        rows.append({
            "model": model,
            "mean_f1": s["mean_f1"],
            "std_f1": s["std_f1"],
            "run_f1s": ";".join(f"{v:.4f}" for v in s["run_f1s"]),
            "run_cis": ";".join(f"[{c['f1'][0]:.4f},{c['f1'][1]:.4f}]" for c in s["run_cis"]),
        })
    pd.DataFrame(rows).to_csv(out_root / "model_summary.csv", index=False)
    try:
        pd.DataFrame(rows).to_latex(out_root / "model_summary.tex", index=False, float_format="%.4f")
    except Exception:
        pass


def paired_comparisons(results_root: Path, out_root: Path):
    rng = np.random.default_rng(RNG_SEED)
    out_root.mkdir(parents=True, exist_ok=True)

    # load all model results into dict: model -> dict(seed -> (y_true,y_pred,boundaries,labels))
    model_data = {}
    data_path = Path("data2")
    model_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    for model_dir in model_dirs:
        res_json = model_dir / "results.json"
        if not res_json.exists():
            continue
        r = json.loads(res_json.read_text())
        model_key = r.get("display_name", model_dir.name)
        config = r.get("config", {})
        window_size = config.get("window_size", 40)
        step_size = config.get("step_size", 20)
        split_seed = config.get("split_seed", 42)

        recordings = dp.load_filtered_recordings(data_path=str(data_path))
        dp.encode_activities(recordings)
        _, _, test_indices = dp.subject_independent_split(dp.get_session_subjects(recordings), train_ratio=0.6, val_ratio=0.2, seed=split_seed)
        test_recordings = [recordings[i] for i in test_indices]
        session_counts = [windows_per_session(len(sess), window_size, step_size) for sess in test_recordings]
        boundaries = []
        start = 0
        for cnt in session_counts:
            end = start + cnt
            boundaries.append((start,end))
            start = end

        configurations = r.get("configurations", [])
        if not configurations:
            continue
        chosen = configurations[0]
        runs = chosen.get("runs", [])
        seed_map = {}
        for run in runs:
            y_true = run.get("y_true") or r.get("y_true")
            y_pred = run.get("y_pred") or r.get("y_pred")
            if y_true is None:
                continue
            seed_map[run.get("seed")] = (y_true, y_pred, boundaries)

        model_data[model_key] = seed_map

    # comparisons against CNN_ResBiGRU
    ref = "CNN_ResBiGRU"
    if ref not in model_data:
        print(f"Reference model {ref} not found; skipping paired comparisons")
        return

    comparisons = []
    for model, seed_map in model_data.items():
        if model == ref:
            continue
        # determine common seeds
        ref_seeds = set(model_data[ref].keys())
        other_seeds = set(seed_map.keys())
        common = sorted(list(ref_seeds & other_seeds))
        pairs = []
        if common:
            for sd in common:
                a = model_data[ref][sd]
                b = seed_map[sd]
                pairs.append((sd, a, b))
        else:
            # fallback: compare best run per model (highest test f1) with ref best run
            def best_run(mmap):
                best = None
                best_f1 = -1
                for sd,(yt,yp,bd) in mmap.items():
                    m = compute_metrics(yt,yp,list(range(len(dp.encode_activities(dp.load_filtered_recordings('data2'))))))
                    if m['f1'] > best_f1:
                        best_f1 = m['f1']; best = (sd,(yt,yp,bd))
                return best

            ref_best = best_run(model_data[ref])
            other_best = best_run(seed_map)
            if ref_best and other_best:
                pairs.append((f"best_{ref_best[0]}", ref_best[1], other_best[1]))

        # run paired session bootstrap per pair and average results per model
        diffs = []
        cis = []
        pvals = []
        for pair in pairs:
            sd, a, b = pair if isinstance(pair[0], int) else (pair[0], pair[1], pair[2])
            yt_a, yp_a, bd = a
            yt_b, yp_b, _ = b
            labels = list(range(len(dp.encode_activities(dp.load_filtered_recordings('data2')))))
            res = paired_session_bootstrap_diff(yt_a, yp_a, yt_b, yp_b, bd, labels, rng)
            diffs.append(res['observed_diff'])
            cis.append((res['ci_low'], res['ci_high']))
            pvals.append(res['p_value'])

        if diffs:
            # summarise by taking mean observed diff and concatenating info
            avg_diff = float(np.mean(diffs))
            # aggregate CI by min lo and max hi across matched comparisons
            lo = float(np.min([c[0] for c in cis]))
            hi = float(np.max([c[1] for c in cis]))
            pval = float(np.mean(pvals))
            significant = not (lo <= 0 <= hi)
            interp = "significant" if significant else "not significant"
            comparisons.append({
                "ref": ref,
                "model": model,
                "delta_f1": avg_diff,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": pval,
                "interpretation": interp,
            })

    cmp_df = pd.DataFrame(comparisons)
    cmp_df.to_csv(out_root / "pairwise_comparisons.csv", index=False)
    try:
        cmp_df.to_latex(out_root / "pairwise_comparisons.tex", index=False, float_format="%.4f")
    except Exception:
        pass


if __name__ == "__main__":
    # Use repository root (two levels up from this file) so `studies/results` is found.
    ROOT = Path(__file__).resolve().parents[2]
    RESULTS_ROOT = ROOT / "studies" / "results"
    OUT_ROOT = RESULTS_ROOT / "bootstrap_results"
    analyze_all(RESULTS_ROOT, OUT_ROOT)
    paired_comparisons(RESULTS_ROOT, OUT_ROOT)
