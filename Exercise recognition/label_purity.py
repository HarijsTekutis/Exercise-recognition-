"""Quantify transition-window label noise.

The reviewer asked how much ambiguity majority labeling of 50%-overlapping windows
introduces near exercise/Null boundaries. Purity of a window = fraction of its samples
that carry the majority label. Purity 1.0 means the window sits entirely inside one
activity; anything below means it straddles a transition.

Run from the project root:  python label_purity.py
"""

import numpy as np
import pandas as pd

import data_pipeline as dp

WINDOW_SIZE = 40
STEP_SIZE = 20
PURITY_THRESHOLD = 1.0  # windows below this are "mixed"

data = dp._load_recgym_recordings("data2/RecGym_new.csv", min_recordings_per_activity=5)
activity_to_id = dp.encode_activities(data)
dp.clean_imu_columns(data, dp.IMU_FEATURES)
id_to_activity = {idx: name for name, idx in activity_to_id.items()}

# Re-window without preprocessing: only the labels matter here.
records = []
for session_df in data:
    session_labels = session_df["activityEncoded"].to_numpy(dtype=np.int64)
    for start in range(0, len(session_labels) - WINDOW_SIZE + 1, STEP_SIZE):
        window_labels = session_labels[start : start + WINDOW_SIZE]
        counts = np.bincount(window_labels)
        majority = int(counts.argmax())
        records.append(
            {
                "assigned": majority,
                "purity": counts[majority] / WINDOW_SIZE,
                "n_classes": int((counts > 0).sum()),
                "involves_null": "Null" in {id_to_activity[c] for c in np.unique(window_labels)},
            }
        )

df = pd.DataFrame(records)
df["activity"] = df["assigned"].map(id_to_activity)

total = len(df)
mixed = df[df["purity"] < PURITY_THRESHOLD]

print(f"Windows: {total}  (size={WINDOW_SIZE}, step={STEP_SIZE})")
print(f"Pure windows (purity = 1.0):     {total - len(mixed):>7} ({(total - len(mixed)) / total:.2%})")
print(f"Mixed windows (purity < 1.0):    {len(mixed):>7} ({len(mixed) / total:.2%})")
print(f"  of which involve Null:         {int(mixed['involves_null'].sum()):>7}")
print(f"  spanning >2 classes:           {int((mixed['n_classes'] > 2).sum()):>7}")
print()
print("Purity distribution over mixed windows:")
if len(mixed):
    print(mixed["purity"].describe().to_string())
    print()
    for threshold in (0.9, 0.75, 0.6, 0.5):
        n = int((df["purity"] < threshold).sum())
        print(f"  purity < {threshold:<5}: {n:>6} ({n / total:.2%})")
print()
print("Per-class mixed-window rate (share of each class's windows that are impure):")
per_class = (
    df.assign(is_mixed=df["purity"] < PURITY_THRESHOLD)
    .groupby("activity")
    .agg(windows=("purity", "size"), mixed=("is_mixed", "sum"), mean_purity=("purity", "mean"))
)
per_class["mixed_rate"] = per_class["mixed"] / per_class["windows"]
print(per_class.sort_values("mixed_rate", ascending=False).to_string(float_format=lambda v: f"{v:.4f}"))
