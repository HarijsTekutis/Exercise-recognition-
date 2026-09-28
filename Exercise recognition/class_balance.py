import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import data_pipeline as dp

data = dp._load_recgym_recordings("data2/RecGym_new.csv", min_recordings_per_activity=5)
activity_to_id = dp.encode_activities(data)
dp.clean_imu_columns(data, dp.IMU_FEATURES)

dataset = dp.IMUDataset(
    data,
    dp.IMU_FEATURES,
    window_size=40,
    step_size=20,
)

labels = np.array([label for _, label in dataset.samples])
id_to_activity = {idx: name for name, idx in activity_to_id.items()}

counts = np.bincount(labels, minlength=len(activity_to_id))
order = np.argsort(counts)  # ascending, so the largest class ends up on top
class_names = [id_to_activity[i] for i in order]
class_counts = counts[order]

# Single-column figure: horizontal bars fix the height at one row per class, and
# the log axis keeps the 908-window class readable next to the 43k Null class.
plt.figure(figsize=(3.4, 2.8))

# Colour carries no information here, so one colour for the activities and a
# second one to set the dominant Null class apart.
deep = sns.color_palette("deep")
colors = [deep[3] if name == "Null" else deep[0] for name in class_names]
bars = plt.barh(class_names, class_counts, color=colors)

plt.xscale("log")
plt.xlim(100, max(class_counts) * 4)
plt.grid(axis="x", alpha=0.3, which="both")
plt.xlabel("Window count (log scale)", fontsize=8)
plt.yticks(fontsize=7)
plt.xticks(fontsize=7)

for bar, value in zip(bars, class_counts):
    plt.text(
        value * 1.12,
        bar.get_y() + bar.get_height() / 2,
        f"{value}",
        ha="left",
        va="center",
        fontsize=6.5,
    )

plt.tight_layout()
plt.savefig("class_balance.png", dpi=300, bbox_inches="tight")
