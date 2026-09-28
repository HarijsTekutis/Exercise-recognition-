"""Row-normalised confusion matrix of the best single CNN_ResBiGRU run.

Same layout as the confusion matrix in result_analysis.ipynb, but taken from the
current study output: batch size 128, 3 GRU layers. The seed is not hardcoded -
the run with the highest test macro F1 of the selected configuration is picked
at load time, so this figure always matches the row reported in the paper table.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "studies/results/cnn_resbigru/results.json"
PARAMS = {"batch_size": 128, "gru_layers": 3}
OUTPUT_PATH = "CNN_ResBiGRU_confusion_matrix.png"

plt.rcParams.update({"font.size": 9})

with open(RESULTS_PATH, "r") as f:
    data = json.load(f)

config = next(c for c in data["configurations"] if c["params"] == PARAMS)
run = max(config["runs"], key=lambda r: r["test_metrics"]["f1_score"])
cm_counts = np.array(run["test_confusion_matrix"], dtype=np.float64)
metrics = run["test_metrics"]

# Class IDs follow the alphabetical order pandas assigns in encode_activities().
id_to_activity = {
    0: "Adductor", 1: "ArmCurl", 2: "BenchPress", 3: "LegCurl",
    4: "LegPress", 5: "NULL", 6: "Riding", 7: "RopeSkipping",
    8: "Running", 9: "Squat", 10: "StairClimber", 11: "Walking",
}
desired_order = [
    "Squat", "LegPress", "LegCurl", "Adductor", "BenchPress", "ArmCurl",
    "NULL", "RopeSkipping", "Riding", "StairClimber", "Walking", "Running",
]
activity_to_id = {activity: idx for idx, activity in id_to_activity.items()}
order_indices = [activity_to_id[name] for name in desired_order]
cm_counts = cm_counts[np.ix_(order_indices, order_indices)]
class_names = desired_order
num_classes = len(class_names)

# Normalise per true class, so each row sums to 100%.
row_sums = cm_counts.sum(axis=1, keepdims=True)
cm = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts), where=row_sums != 0)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)

ax.set_xlabel("Predicted", fontsize=9)
ax.set_ylabel("True", fontsize=9)

ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(class_names, fontsize=9)

# Add a cell grid
ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.35)
ax.tick_params(which="minor", bottom=False, left=False)

# Show non-zero densities as percentages
for i in range(num_classes):
    for j in range(num_classes):
        value = cm[i, j]
        if value > 0:
            text_color = "white" if value >= 0.6 else "black"
            ax.text(j, i, f"{value:.1%}", ha="center", va="center",
                    color=text_color, fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

print(
    f"seed {run['seed']}  accuracy {metrics['accuracy']:.4f}  "
    f"F1 {metrics['f1_score']:.4f} -> {OUTPUT_PATH}"
)
