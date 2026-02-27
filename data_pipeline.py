import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


IMU_FEATURES = [
    "wristMotion_rotationRateX",
    "wristMotion_rotationRateY",
    "wristMotion_rotationRateZ",
    "wristMotion_accelerationX",
    "wristMotion_accelerationY",
    "wristMotion_accelerationZ",
]


def _resolve_label_column(columns: Sequence[str], aliases: Sequence[str]) -> str:
    normalized = {c.lower().replace(" ", "").replace("_", ""): c for c in columns}
    for alias in aliases:
        key = alias.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    raise ValueError(f"None of aliases {aliases} found in columns: {list(columns)}")


def load_npy_segmented_recordings(
    dataset_root: str,
    sample_rate_hz: float,
    sensor_npy_name: Optional[str] = None,
    labels_csv_name: Optional[str] = None,
    feature_index_map: Optional[Dict[str, int]] = None,
    labels_time_unit: str = "seconds",
    min_recordings_per_activity: int = 0,
) -> List[pd.DataFrame]:
    """
    Convert session-level NPY files + labels CSV intervals into per-segment DataFrames.

    Expected labels columns (case-insensitive aliases supported):
      - start time: start / start_time / startTime / start_sec / start_s
      - end time: end / end_time / endTime / end_sec / end_s
      - exercise label: exercise / activity / label / class
      - reps (optional): repetitions / reps / rep

    Args:
        dataset_root: Root folder containing session subfolders/files.
        sample_rate_hz: Sensor sampling frequency used to derive secondsElapsed.
        sensor_npy_name: Optional exact sensor NPY filename per session folder.
        labels_csv_name: Optional exact labels CSV filename per session folder.
        feature_index_map: Mapping from canonical IMU feature names to NPY column indices.
            If omitted, first 6 channels are assumed in IMU_FEATURES order.
        labels_time_unit: "seconds" or "milliseconds" for label start/end values.
        min_recordings_per_activity: Keep only activities with strictly more recordings than this value.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")

    if labels_time_unit not in {"seconds", "milliseconds"}:
        raise ValueError("labels_time_unit must be 'seconds' or 'milliseconds'")

    segments: List[pd.DataFrame] = []
    activities: List[str] = []

    for session_dir, _, files in os.walk(dataset_root):
        files_set = set(files)
        npy_candidates = [f for f in files if f.lower().endswith(".npy")]
        csv_candidates = [f for f in files if f.lower().endswith(".csv")]

        if sensor_npy_name:
            if sensor_npy_name not in files_set:
                continue
            sensor_file = sensor_npy_name
        else:
            if not npy_candidates:
                continue
            sensor_file = sorted(npy_candidates)[0]

        if labels_csv_name:
            if labels_csv_name not in files_set:
                continue
            labels_file = labels_csv_name
        else:
            if not csv_candidates:
                continue
            labels_named = [f for f in csv_candidates if "label" in f.lower()]
            labels_file = sorted(labels_named if labels_named else csv_candidates)[0]

        sensor_path = Path(session_dir) / sensor_file
        labels_path = Path(session_dir) / labels_file

        sensor = np.load(sensor_path)
        if sensor.ndim != 2:
            raise ValueError(
                f"Expected 2D sensor array [time, channels], got shape {sensor.shape} at {sensor_path}"
            )

        num_channels = sensor.shape[1]
        if feature_index_map is None:
            if num_channels < len(IMU_FEATURES):
                raise ValueError(
                    f"Need at least {len(IMU_FEATURES)} channels, got {num_channels} at {sensor_path}"
                )
            feature_index_map_use = {feature: idx for idx, feature in enumerate(IMU_FEATURES)}
        else:
            feature_index_map_use = feature_index_map

        missing_features = [f for f in IMU_FEATURES if f not in feature_index_map_use]
        if missing_features:
            raise ValueError(
                f"feature_index_map missing required features: {missing_features}"
            )

        for feature, idx in feature_index_map_use.items():
            if idx < 0 or idx >= num_channels:
                raise ValueError(
                    f"Feature '{feature}' index {idx} out of range for {num_channels} channels at {sensor_path}"
                )

        seconds_elapsed = np.arange(sensor.shape[0], dtype=np.float32) / float(sample_rate_hz)
        sensor_df = pd.DataFrame({
            "secondsElapsed": seconds_elapsed,
            **{feature: sensor[:, feature_index_map_use[feature]] for feature in IMU_FEATURES},
        })

        labels_df = pd.read_csv(labels_path)
        start_col = _resolve_label_column(
            labels_df.columns,
            ["start", "start_time", "startTime", "start_sec", "start_s"],
        )
        end_col = _resolve_label_column(
            labels_df.columns,
            ["end", "end_time", "endTime", "end_sec", "end_s"],
        )
        exercise_col = _resolve_label_column(
            labels_df.columns,
            ["exercise", "activity", "label", "class"],
        )

        reps_col = None
        try:
            reps_col = _resolve_label_column(
                labels_df.columns,
                ["repetitions", "reps", "rep"],
            )
        except ValueError:
            reps_col = None

        for _, row in labels_df.iterrows():
            start_time = float(row[start_col])
            end_time = float(row[end_col])
            if labels_time_unit == "milliseconds":
                start_time /= 1000.0
                end_time /= 1000.0

            if end_time <= start_time:
                continue

            seg = sensor_df[
                (sensor_df["secondsElapsed"] >= start_time)
                & (sensor_df["secondsElapsed"] <= end_time)
            ].copy()

            if seg.empty:
                continue

            seg["activity"] = str(row[exercise_col])
            if reps_col is not None:
                seg["reps"] = row[reps_col]

            segments.append(seg)
            activities.append(seg["activity"].iloc[0])

    if not segments:
        raise ValueError(
            "No valid segments produced. Check folder layout, filenames, and labels columns."
        )

    if min_recordings_per_activity > 0:
        activity_counts = pd.Series(activities).value_counts()
        valid_activities = activity_counts[
            activity_counts > min_recordings_per_activity
        ].index
        segments = [
            df for df in segments if df["activity"].iloc[0] in valid_activities
        ]

    return segments


def _extract_xyz_from_modality_array(arr: np.ndarray, file_path: Path) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {file_path}, got shape {arr.shape}")

    if arr.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 columns (x,y,z) in {file_path}, got shape {arr.shape}"
        )

    if arr.shape[1] == 3:
        return arr.astype(np.float32)

    first_col = arr[:, 0]
    if np.all(np.diff(first_col) >= 0):
        return arr[:, -3:].astype(np.float32)

    return arr[:, :3].astype(np.float32)


def load_mmfit_wrist_recordings(
    dataset_root: str,
    sample_rate_hz: float = 100.0,
    labels_time_unit: str = "milliseconds",
    min_recordings_per_activity: int = 0,
) -> List[pd.DataFrame]:
    """
    Load MM-Fit sessions (w00..w20) using only:
      - wxx_sw_r_acc.npy
      - wxx_sw_r_gyr.npy
      - wxx_sw_l_acc.npy
      - wxx_sw_l_gyr.npy
      - wxx_labels.csv

    Returns per-interval DataFrames with these canonical columns:
      - secondsElapsed
      - sw_r_acc_x/y/z, sw_r_gyr_x/y/z, sw_l_acc_x/y/z, sw_l_gyr_x/y/z
      - activity
      - reps (if present in labels)
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")

    if labels_time_unit not in {"seconds", "milliseconds"}:
        raise ValueError("labels_time_unit must be 'seconds' or 'milliseconds'")

    sessions = sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("w")])
    segments: List[pd.DataFrame] = []
    activities: List[str] = []

    for session_dir in sessions:
        session_id = session_dir.name
        required_files = {
            "sw_r_acc": session_dir / f"{session_id}_sw_r_acc.npy",
            "sw_r_gyr": session_dir / f"{session_id}_sw_r_gyr.npy",
            "sw_l_acc": session_dir / f"{session_id}_sw_l_acc.npy",
            "sw_l_gyr": session_dir / f"{session_id}_sw_l_gyr.npy",
            "labels": session_dir / f"{session_id}_labels.csv",
        }

        if not all(path.exists() for path in required_files.values()):
            continue

        sw_r_acc = _extract_xyz_from_modality_array(
            np.load(required_files["sw_r_acc"]), required_files["sw_r_acc"]
        )
        sw_r_gyr = _extract_xyz_from_modality_array(
            np.load(required_files["sw_r_gyr"]), required_files["sw_r_gyr"]
        )
        sw_l_acc = _extract_xyz_from_modality_array(
            np.load(required_files["sw_l_acc"]), required_files["sw_l_acc"]
        )
        sw_l_gyr = _extract_xyz_from_modality_array(
            np.load(required_files["sw_l_gyr"]), required_files["sw_l_gyr"]
        )

        min_len = min(len(sw_r_acc), len(sw_r_gyr), len(sw_l_acc), len(sw_l_gyr))
        if min_len == 0:
            continue

        sw_r_acc = sw_r_acc[:min_len]
        sw_r_gyr = sw_r_gyr[:min_len]
        sw_l_acc = sw_l_acc[:min_len]
        sw_l_gyr = sw_l_gyr[:min_len]

        seconds_elapsed = np.arange(min_len, dtype=np.float32) / float(sample_rate_hz)
        sensor_df = pd.DataFrame(
            {
                "secondsElapsed": seconds_elapsed,
                "sw_r_acc_x": sw_r_acc[:, 0],
                "sw_r_acc_y": sw_r_acc[:, 1],
                "sw_r_acc_z": sw_r_acc[:, 2],
                "sw_r_gyr_x": sw_r_gyr[:, 0],
                "sw_r_gyr_y": sw_r_gyr[:, 1],
                "sw_r_gyr_z": sw_r_gyr[:, 2],
                "sw_l_acc_x": sw_l_acc[:, 0],
                "sw_l_acc_y": sw_l_acc[:, 1],
                "sw_l_acc_z": sw_l_acc[:, 2],
                "sw_l_gyr_x": sw_l_gyr[:, 0],
                "sw_l_gyr_y": sw_l_gyr[:, 1],
                "sw_l_gyr_z": sw_l_gyr[:, 2],
            }
        )

        labels_df = pd.read_csv(required_files["labels"])
        start_col = _resolve_label_column(
            labels_df.columns,
            ["start", "start_time", "startTime", "start_sec", "start_s"],
        )
        end_col = _resolve_label_column(
            labels_df.columns,
            ["end", "end_time", "endTime", "end_sec", "end_s"],
        )
        exercise_col = _resolve_label_column(
            labels_df.columns,
            ["exercise", "activity", "label", "class"],
        )

        reps_col = None
        try:
            reps_col = _resolve_label_column(
                labels_df.columns,
                ["repetitions", "reps", "rep"],
            )
        except ValueError:
            reps_col = None

        for _, row in labels_df.iterrows():
            start_time = float(row[start_col])
            end_time = float(row[end_col])
            if labels_time_unit == "milliseconds":
                start_time /= 1000.0
                end_time /= 1000.0

            if end_time <= start_time:
                continue

            seg = sensor_df[
                (sensor_df["secondsElapsed"] >= start_time)
                & (sensor_df["secondsElapsed"] <= end_time)
            ].copy()
            if seg.empty:
                continue

            seg["activity"] = str(row[exercise_col])
            if reps_col is not None:
                seg["reps"] = row[reps_col]

            segments.append(seg)
            activities.append(seg["activity"].iloc[0])

    if not segments:
        raise ValueError(
            "No valid MM-Fit segments found. Check folder names and required filenames."
        )

    if min_recordings_per_activity > 0:
        activity_counts = pd.Series(activities).value_counts()
        valid_activities = activity_counts[
            activity_counts > min_recordings_per_activity
        ].index
        segments = [
            df for df in segments if df["activity"].iloc[0] in valid_activities
        ]

    return segments


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    df_filtered = df.copy()
    min_time = df_filtered["secondsElapsed"].min() + 1.5
    max_time = df_filtered["secondsElapsed"].max() - 1.5
    df_filtered = df_filtered[
        (df_filtered["secondsElapsed"] >= min_time)
        & (df_filtered["secondsElapsed"] <= max_time)
    ].reset_index(drop=True)

    df_filtered = df_filtered.drop(
        columns=[
            "wristMotion_gravityX",
            "wristMotion_gravityY",
            "wristMotion_gravityZ",
            "wristMotion_quaternionW",
            "wristMotion_quaternionX",
            "wristMotion_quaternionY",
            "wristMotion_quaternionZ",
        ],
        errors="ignore",
    )
    return df_filtered


def smooth_columns(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df_smoothed = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df_smoothed[col] = df[col].rolling(window=window, center=True).mean()
    return df_smoothed


def load_filtered_recordings(
    data_path: str, min_recordings_per_activity: int = 8
) -> List[pd.DataFrame]:
    data_list: List[pd.DataFrame] = []
    activity_list: List[str] = []

    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(dirname, filename))
            df_filtered = filter_data(df)
            if not df_filtered.empty and "activity" in df_filtered.columns:
                data_list.append(df_filtered)
                activity_list.append(df_filtered["activity"].iloc[0])

    activity_counts = pd.Series(activity_list).value_counts()
    valid_activities = activity_counts[
        activity_counts > min_recordings_per_activity
    ].index

    filtered_data_list = [
        df for df in data_list if df["activity"].iloc[0] in valid_activities
    ]
    return filtered_data_list


def encode_activities(dataframes: List[pd.DataFrame]) -> Dict[str, int]:
    all_activities = pd.concat([df["activity"] for df in dataframes], ignore_index=True)
    categories = all_activities.astype("category").cat.categories
    activity_to_id = {cat: idx for idx, cat in enumerate(categories)}

    for i, df in enumerate(dataframes):
        df = df.copy()
        df["activityEncoded"] = df["activity"].map(activity_to_id)
        dataframes[i] = df

    return activity_to_id


def clean_imu_columns(dataframes: List[pd.DataFrame], imu_features: Sequence[str]) -> None:
    for i, df in enumerate(dataframes):
        df = df.copy()
        df[list(imu_features)] = df[list(imu_features)].fillna(0).replace([np.inf, -np.inf], 0)
        dataframes[i] = df


def preprocess_sample(
    window: np.ndarray,
    y: int,
    smooth_kernel: int = 5,
    downsample_factor: int = 2,
    downsample_mode: str = "avg",
) -> Tuple[np.ndarray, int]:
    X = window.astype(np.float32)

    if smooth_kernel and smooth_kernel > 1:
        k = int(smooth_kernel)
        if k % 2 == 0:
            k += 1

        kernel = np.ones(k, dtype=np.float32) / k
        X = np.vstack(
            [np.convolve(X[:, f], kernel, mode="same") for f in range(X.shape[1])]
        ).T.astype(np.float32)

    if downsample_factor and downsample_factor > 1:
        d = int(downsample_factor)
        if downsample_mode == "avg":
            T, F = X.shape
            pad_needed = (-T) % d
            if pad_needed:
                X = np.concatenate([X, np.repeat(X[-1:, :], pad_needed, axis=0)], axis=0)
            X = X.reshape(-1, d, F).mean(axis=1)
        else:
            X = X[::d]

    return X, y


class IMUDataset(Dataset):
    def __init__(
        self,
        dataframes: List[pd.DataFrame],
        features: Sequence[str],
        window_size: int,
        step_size: int,
        preprocess_fn=None,
        preprocess_kwargs=None,
    ):
        self.samples = []
        self.preprocess_fn = preprocess_fn
        self.preprocess_kwargs = preprocess_kwargs or {}

        for df in dataframes:
            X = df[list(features)].values
            y = int(df["activityEncoded"].iloc[0])

            for start in range(0, len(X) - window_size + 1, step_size):
                end = start + window_size
                window = X[start:end]

                if self.preprocess_fn is not None:
                    window, y_out = self.preprocess_fn(window, y, **self.preprocess_kwargs)
                    y_use = y_out
                else:
                    y_use = y

                self.samples.append((window, y_use))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return X, y


def stratified_session_split(
    data: List[pd.DataFrame], labels: np.ndarray, train_ratio: float = 0.8, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    idx_all = np.arange(len(labels))
    target_train = int(train_ratio * len(labels))

    cls_counts = dict(zip(*np.unique(labels, return_counts=True)))
    rare_classes = {c for c, k in cls_counts.items() if k < 2}

    forced_train_idx = np.array([i for i in idx_all if labels[i] in rare_classes], dtype=int)
    rest_idx = np.array([i for i in idx_all if labels[i] not in rare_classes], dtype=int)

    if len(rest_idx) == 0:
        warnings.warn("All classes are rare (count<2). Putting all sessions in TRAIN.")
        return forced_train_idx, np.array([], dtype=int)

    remaining_train_needed = max(0, target_train - len(forced_train_idx))
    remaining_total = len(rest_idx)

    if remaining_total == 0 or remaining_train_needed == 0:
        train_idx = forced_train_idx
        test_idx = np.setdiff1d(idx_all, train_idx, assume_unique=False)
        return np.sort(train_idx), np.sort(test_idx)

    adjusted_train_ratio = min(1.0, remaining_train_needed / remaining_total)
    adjusted_test_size = 1.0 - adjusted_train_ratio

    sss = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_test_size, random_state=seed)
    rest_labels = labels[rest_idx]
    (rest_train_sel, rest_test_sel), = sss.split(rest_idx, rest_labels)

    train_idx = np.concatenate([forced_train_idx, rest_idx[rest_train_sel]])
    test_idx = rest_idx[rest_test_sel]

    present_all = set(np.unique(labels))
    present_train = set(np.unique(labels[train_idx]))
    missing_in_train = present_all - present_train
    if missing_in_train:
        warnings.warn(
            f"Some classes had too few sessions; moving one session per missing class to TRAIN: {missing_in_train}"
        )
        for cls in list(missing_in_train):
            cand = np.where(labels[test_idx] == cls)[0]
            if len(cand) > 0:
                move = test_idx[cand[0]]
                train_idx = np.append(train_idx, move)
                test_idx = np.delete(test_idx, cand[0])

    return np.sort(train_idx), np.sort(test_idx)


def make_train_test_loaders(
    data: List[pd.DataFrame],
    imu_features: Sequence[str],
    window_size: int = 300,
    step_size: int = 100,
    train_split: float = 0.8,
    batch_size_train: int = 32,
    batch_size_test: int = 1,
):
    session_labels = np.array([int(df["activityEncoded"].iloc[0]) for df in data])
    train_idx, test_idx = stratified_session_split(
        data, session_labels, train_ratio=train_split, seed=42
    )

    train_dfs = [data[i] for i in train_idx]
    test_dfs = [data[i] for i in test_idx]

    scaler = StandardScaler()
    train_stack_df = pd.concat([df.loc[:, imu_features] for df in train_dfs], ignore_index=True)
    scaler.fit(train_stack_df)

    for df in train_dfs:
        df.loc[:, imu_features] = scaler.transform(df.loc[:, imu_features])

    for df in test_dfs:
        df.loc[:, imu_features] = scaler.transform(df.loc[:, imu_features])

    preprocess_kwargs = dict(smooth_kernel=5, downsample_factor=2, downsample_mode="avg")

    train_dataset = IMUDataset(
        dataframes=train_dfs,
        features=imu_features,
        window_size=window_size,
        step_size=step_size,
        preprocess_fn=preprocess_sample,
        preprocess_kwargs=preprocess_kwargs,
    )

    test_dataset = IMUDataset(
        dataframes=test_dfs,
        features=imu_features,
        window_size=window_size,
        step_size=step_size,
        preprocess_fn=preprocess_sample,
        preprocess_kwargs=preprocess_kwargs,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    assert set(map(id, train_dfs)).isdisjoint(
        set(map(id, test_dfs))
    ), "Leak: same session in both splits"

    return train_loader, test_loader, train_dataset, test_dataset
