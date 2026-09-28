"""Bar chart of test macro F1 per model, sorted best to worst.

Same layout as results.png, regenerated from the final single-split study output
in studies/results/ (deep models plus the three classical baselines).

The score plotted per model is the run with the highest test macro F1 among the
five seeds of that model's selected configuration - the same run reported in the
paper's model-comparison table. Nothing is hardcoded, so the figure cannot drift
away from the table.
"""

import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "studies/results"
OUTPUT_PATH = "results2.png"
# Batch size is a training hyperparameter of the deep models only.
BASELINES = {"Rocket", "RandomForest", "XGBoost"}

MODEL_KEYS = [
    "cnn_resbigru",
    "multi_head_cnn_resbilstm",
    "multi_head_cnn_bilstm",
    "cnn_bilstm",
    "cnn_resbilstm",
    "cnn_bigru",
    "xgboost",
    "rocket",
    "randomforest",
]

# (display name, batch size or None, macro F1)
RESULTS = []
for key in MODEL_KEYS:
    with open(os.path.join(RESULTS_DIR, key, "results.json")) as f:
        data = json.load(f)
    name = data["display_name"]
    params = data["selected"]["params"]
    config = next(c for c in data["configurations"] if c["params"] == params)
    best_run = max(config["runs"], key=lambda r: r["test_metrics"]["f1_score"])
    batch = None if name in BASELINES else params["batch_size"]
    RESULTS.append((name, batch, best_run["test_metrics"]["f1_score"]))

RESULTS.sort(key=lambda row: row[2], reverse=True)

names = [name for name, _, _ in RESULTS]
scores = [f1 for _, _, f1 in RESULTS]
labels = [
    f"{name}\nbatch size = {batch}" if batch else name
    for name, batch, _ in RESULTS
]

plt.figure(figsize=(12, 6))

palette = sns.color_palette("deep", len(names))
bars = plt.bar(labels, scores, color=palette)

for bar, score in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{score:.3f}",
        ha="center",
        va="bottom",
    )

# Cropped y-axis: the whole spread sits in the top tenth of the [0, 1] range, and
# an uncropped axis makes the models indistinguishable.
plt.ylim(0.60, 0.93)
plt.ylabel("F1")
plt.xlabel("Model name")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)

for name, batch, f1 in RESULTS:
    print(f"{name:24s} batch={str(batch):>4s}  F1={f1:.4f}")
