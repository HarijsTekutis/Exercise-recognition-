"""Bar chart of the best macro F1 per model across the batch-size sweep.

Regenerates results.png from results_finding_batch_sizes/. Batch size is a
training hyperparameter of the deep models only, so the classical baselines are
labelled with their name alone.
"""

import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZES = [8, 16, 32, 64, 128, 256]
SWEEP_DIR = "results_finding_batch_sizes"
BASELINES = {"Rocket", "RandomForest", "XGBoost"}

# best (f1, batch size) seen per model
best = {}
for bs in BATCH_SIZES:
    results_path = os.path.join(
        SWEEP_DIR, f"model_comparison_results_batch_size_{bs}", "results.json"
    )
    if not os.path.exists(results_path):
        continue
    with open(results_path) as f:
        data = json.load(f)
    for model_name, info in data.items():
        f1 = info.get("metrics", {}).get("f1_score")
        if f1 is None:
            continue
        if model_name not in best or f1 > best[model_name][0]:
            best[model_name] = (f1, bs)

rows = sorted(best.items(), key=lambda item: item[1][0], reverse=True)

labels = [
    name if name in BASELINES else f"{name}\nbatch size = {bs}"
    for name, (_, bs) in rows
]
values = [f1 for _, (f1, _) in rows]

plt.figure(figsize=(12, 6))

palette = sns.color_palette("deep", len(labels))
bars = plt.bar(labels, values, color=palette)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        min(value + 0.01, 0.99),
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.ylim(0, 1.0)
plt.grid(axis="y", alpha=0.3)
plt.xlabel("Model name", fontsize=12)
plt.ylabel("F1", fontsize=12)
plt.xticks(rotation=45, ha="center")
plt.tight_layout()
plt.savefig("results.png", dpi=180, bbox_inches="tight")
