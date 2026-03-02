import os
import warnings
from typing import Dict, List, Sequence, Tuple

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


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Trim noisy edges in time and drop sensor columns that are not used.

    This keeps only the central part of each recording and removes gravity/quaternion
    channels so the downstream model sees only the core IMU signals.
    """
    filtered_df = df.copy()

    # Remove the first/last ~1.5 seconds, which are often transition noise.
    min_time = filtered_df["secondsElapsed"].min() + 1.5
    max_time = filtered_df["secondsElapsed"].max() - 1.5
    filtered_df = filtered_df[
        (filtered_df["secondsElapsed"] >= min_time)
        & (filtered_df["secondsElapsed"] <= max_time)
    ].reset_index(drop=True)

    # Keep only the channels used by the training pipeline.
    filtered_df = filtered_df.drop(
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

    return filtered_df


def smooth_columns(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply a centered moving average to every numeric column."""
    smoothed_df = df.copy()
    for column_name in df.columns:
        if pd.api.types.is_numeric_dtype(df[column_name]):
            smoothed_df[column_name] = df[column_name].rolling(window=window, center=True).mean()
    return smoothed_df


def load_filtered_recordings(
    data_path: str,
    min_recordings_per_activity: int = 5,
) -> List[pd.DataFrame]:
    """Load CSV recordings, apply filtering, and keep only frequent classes.

    A class is kept when its number of recordings is >= min_recordings_per_activity.
    """
    recordings: List[pd.DataFrame] = []
    activity_labels: List[str] = []

    # Walk through all CSV files under the dataset directory.
    for directory_path, _, file_names in os.walk(data_path):
        for file_name in file_names:
            if not file_name.lower().endswith(".csv"):
                continue

            recording_df = pd.read_csv(os.path.join(directory_path, file_name))
            filtered_recording = filter_data(recording_df)
            if not filtered_recording.empty and "activity" in filtered_recording.columns:
                recordings.append(filtered_recording)
                activity_labels.append(str(filtered_recording["activity"].iloc[0]))

    if not recordings:
        raise ValueError(f"No CSV recordings found in: {data_path}")

    # Count recordings per activity and keep classes above threshold.
    activity_counts = pd.Series(activity_labels).value_counts()
    valid_activities = activity_counts[
        activity_counts >= min_recordings_per_activity
    ].index

    filtered_recordings = [
        recording for recording in recordings if str(recording["activity"].iloc[0]) in valid_activities
    ]

    if not filtered_recordings:
        raise ValueError(
            "No recordings left after activity frequency filtering. "
            "Lower min_recordings_per_activity or check labels."
        )

    return filtered_recordings


def encode_activities(dataframes: List[pd.DataFrame]) -> Dict[str, int]:
    """Create activity -> class_id mapping and write it into each dataframe."""
    all_activity_values = pd.concat([df["activity"] for df in dataframes], ignore_index=True)
    activity_categories = all_activity_values.astype("category").cat.categories
    activity_to_id = {activity_name: idx for idx, activity_name in enumerate(activity_categories)}

    for index, dataframe in enumerate(dataframes):
        updated_dataframe = dataframe.copy()
        updated_dataframe["activityEncoded"] = updated_dataframe["activity"].map(activity_to_id)
        dataframes[index] = updated_dataframe

    return activity_to_id


def clean_imu_columns(dataframes: List[pd.DataFrame], imu_features: Sequence[str]) -> None:
    """Replace NaN/inf values in feature columns so training stays stable."""
    for index, dataframe in enumerate(dataframes):
        cleaned_dataframe = dataframe.copy()
        cleaned_dataframe[list(imu_features)] = (
            cleaned_dataframe[list(imu_features)]
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
        )
        dataframes[index] = cleaned_dataframe


def preprocess_sample(
    window: np.ndarray,
    y: int,
    smooth_kernel: int = 5,
    downsample_factor: int = 2,
    downsample_mode: str = "avg",
) -> Tuple[np.ndarray, int]:
    """Preprocess one window: smoothing + optional downsampling.

    Returns the transformed window and the unchanged class label.
    """
    processed_window = window.astype(np.float32)

    # Smooth each feature with a moving-average kernel.
    if smooth_kernel and smooth_kernel > 1:
        kernel_size = int(smooth_kernel)
        if kernel_size % 2 == 0:
            kernel_size += 1

        smoothing_kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        processed_window = np.vstack(
            [
                np.convolve(processed_window[:, feature_idx], smoothing_kernel, mode="same")
                for feature_idx in range(processed_window.shape[1])
            ]
        ).T.astype(np.float32)

    # Downsample either by block averaging or simple slicing.
    if downsample_factor and downsample_factor > 1:
        downsample_step = int(downsample_factor)
        if downsample_mode == "avg":
            num_timesteps, num_features = processed_window.shape
            pad_needed = (-num_timesteps) % downsample_step
            if pad_needed:
                processed_window = np.concatenate(
                    [
                        processed_window,
                        np.repeat(processed_window[-1:, :], pad_needed, axis=0),
                    ],
                    axis=0,
                )
            processed_window = processed_window.reshape(-1, downsample_step, num_features).mean(axis=1)
        else:
            processed_window = processed_window[::downsample_step]

    return processed_window, y


class IMUDataset(Dataset):
    """Windowed IMU dataset built from session-level dataframes."""

    def __init__(
        self,
        dataframes: List[pd.DataFrame],
        features: Sequence[str],
        window_size: int,
        step_size: int,
        preprocess_fn=None,
        preprocess_kwargs=None,
    ):
        """Create fixed-size windows from each session dataframe.

        Each dataframe contributes multiple overlapping windows with one session label.
        """
        self.samples = []
        self.preprocess_fn = preprocess_fn
        self.preprocess_kwargs = preprocess_kwargs or {}

        for session_df in dataframes:
            feature_matrix = session_df[list(features)].values
            session_label = int(session_df["activityEncoded"].iloc[0])

            # Sliding window extraction.
            for start_index in range(0, len(feature_matrix) - window_size + 1, step_size):
                end_index = start_index + window_size
                window = feature_matrix[start_index:end_index]

                if self.preprocess_fn is not None:
                    processed_window, processed_label = self.preprocess_fn(
                        window,
                        session_label,
                        **self.preprocess_kwargs,
                    )
                else:
                    processed_window, processed_label = window, session_label

                self.samples.append((processed_window, processed_label))

    def __len__(self):
        """Number of window samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return one sample as torch tensors: (X, y)."""
        feature_window, class_label = self.samples[idx]
        feature_window = torch.tensor(feature_window, dtype=torch.float32)
        class_label = torch.tensor(class_label, dtype=torch.long)
        return feature_window, class_label


def stratified_session_split(
    data: List[pd.DataFrame],
    labels: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split sessions into train/test while handling rare classes safely.

    Classes with fewer than 2 sessions cannot be stratified, so they are forced into
    train to avoid split errors.
    """
    label_array = np.asarray(labels)
    all_session_indices = np.arange(len(label_array))
    target_train_count = int(train_ratio * len(label_array))

    class_counts = dict(zip(*np.unique(label_array, return_counts=True)))
    rare_classes = {class_id for class_id, count in class_counts.items() if count < 2}

    forced_train_indices = np.array(
        [index for index in all_session_indices if label_array[index] in rare_classes],
        dtype=int,
    )
    remaining_indices = np.array(
        [index for index in all_session_indices if label_array[index] not in rare_classes],
        dtype=int,
    )

    if len(remaining_indices) == 0:
        warnings.warn("All classes are rare (count<2). Putting all sessions in TRAIN.")
        return forced_train_indices, np.array([], dtype=int)

    remaining_train_needed = max(0, target_train_count - len(forced_train_indices))
    remaining_total = len(remaining_indices)

    if remaining_total == 0 or remaining_train_needed == 0:
        train_indices = forced_train_indices
        test_indices = np.setdiff1d(all_session_indices, train_indices, assume_unique=False)
        return np.sort(train_indices), np.sort(test_indices)

    adjusted_train_ratio = min(1.0, remaining_train_needed / remaining_total)
    adjusted_test_size = 1.0 - adjusted_train_ratio

    # Stratified split on classes that can be stratified.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_test_size, random_state=seed)
    remaining_labels = label_array[remaining_indices]
    (remaining_train_selection, remaining_test_selection), = splitter.split(
        remaining_indices,
        remaining_labels,
    )

    train_indices = np.concatenate(
        [forced_train_indices, remaining_indices[remaining_train_selection]]
    )
    test_indices = remaining_indices[remaining_test_selection]

    all_present_classes = set(np.unique(label_array))
    train_present_classes = set(np.unique(label_array[train_indices]))
    classes_missing_from_train = all_present_classes - train_present_classes
    if classes_missing_from_train:
        warnings.warn(
            "Some classes had too few sessions; moving one session per missing class "
            f"to TRAIN: {classes_missing_from_train}"
        )
        for class_id in list(classes_missing_from_train):
            candidate_positions = np.where(label_array[test_indices] == class_id)[0]
            if len(candidate_positions) > 0:
                move_index = test_indices[candidate_positions[0]]
                train_indices = np.append(train_indices, move_index)
                test_indices = np.delete(test_indices, candidate_positions[0])

    return np.sort(train_indices), np.sort(test_indices)


def make_train_test_loaders(
    data: List[pd.DataFrame],
    imu_features: Sequence[str],
    window_size: int = 300,
    step_size: int = 100,
    train_split: float = 0.8,
    batch_size_train: int = 32,
    batch_size_test: int = 1,
):
    """Build train/test DataLoaders from session DataFrames.

    Pipeline:
    1) stratified session split
    2) fit scaler on train only
    3) transform train/test
    4) create window datasets and loaders
    """
    session_labels = np.array([int(session_df["activityEncoded"].iloc[0]) for session_df in data])
    train_indices, test_indices = stratified_session_split(
        data,
        session_labels,
        train_ratio=train_split,
        seed=42,
    )

    train_sessions = [data[index] for index in train_indices]
    test_sessions = [data[index] for index in test_indices]

    # Fit normalization on train only (prevents test leakage).
    scaler = StandardScaler()
    train_feature_table = pd.concat(
        [session_df.loc[:, imu_features] for session_df in train_sessions],
        ignore_index=True,
    )
    scaler.fit(train_feature_table)

    for session_df in train_sessions:
        session_df.loc[:, imu_features] = scaler.transform(session_df.loc[:, imu_features])

    for session_df in test_sessions:
        session_df.loc[:, imu_features] = scaler.transform(session_df.loc[:, imu_features])

    preprocess_kwargs = dict(smooth_kernel=5, downsample_factor=2, downsample_mode="avg")

    train_dataset = IMUDataset(
        dataframes=train_sessions,
        features=imu_features,
        window_size=window_size,
        step_size=step_size,
        preprocess_fn=preprocess_sample,
        preprocess_kwargs=preprocess_kwargs,
    )

    test_dataset = IMUDataset(
        dataframes=test_sessions,
        features=imu_features,
        window_size=window_size,
        step_size=step_size,
        preprocess_fn=preprocess_sample,
        preprocess_kwargs=preprocess_kwargs,
    )

    if len(train_dataset) == 0:
        raise ValueError("Train dataset has 0 windows. Reduce window_size/step_size.")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset has 0 windows. Reduce window_size/step_size.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    # Extra safety check: no session appears in both splits.
    assert set(map(id, train_sessions)).isdisjoint(
        set(map(id, test_sessions))
    ), "Leak: same session in both splits"

    return train_loader, test_loader, train_dataset, test_dataset
