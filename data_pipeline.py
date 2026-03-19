import os
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


#used features for training
IMU_FEATURES = [
    "A_x",
    "A_y",
    "A_z",
    "G_x",
    "G_y",
    "G_z",
]

ACTIVITY_COLUMN = "Workout"
SESSION_COLUMN_CANDIDATES = [
    "Subject",
    "Position",
    "Session"
]


def _validate_recgym_schema(df: pd.DataFrame) -> None:
    """Validate that required columns exist before processing."""
    missing_feature_columns = [feature for feature in IMU_FEATURES if feature not in df.columns]
    if missing_feature_columns:
        raise ValueError(
            "columns are missing: "
            f"{missing_feature_columns}"
        )

    if ACTIVITY_COLUMN not in df.columns:
        raise ValueError(
            f"column '{ACTIVITY_COLUMN}' is missing. "
        )


# Get one representative label for a session by majority vote.
def _session_majority_label(session_df: pd.DataFrame) -> str:
    """Return the most frequent activity label in one session dataframe."""
    return str(session_df["activity"].mode(dropna=False).iloc[0])


def _apply_scaler_inplace(
    sessions: List[pd.DataFrame],
    scaler: StandardScaler,
    imu_features: Sequence[str],
) -> None:
    """Apply a fitted scaler to all sessions in place for selected feature columns."""
    for session_df in sessions:
        session_df.loc[:, imu_features] = scaler.transform(session_df.loc[:, imu_features])


# Keep only required IMU + label columns and standardize label name to `activity`.
def filter_data(df: pd.DataFrame) -> pd.DataFrame:

    _validate_recgym_schema(df)

    filtered_df = df.loc[:, list(IMU_FEATURES) + [ACTIVITY_COLUMN]].copy()
    filtered_df = filtered_df.rename(columns={ACTIVITY_COLUMN: "activity"})
    filtered_df = filtered_df.dropna(subset=list(IMU_FEATURES) + ["activity"])
    filtered_df["activity"] = filtered_df["activity"].astype(str).str.strip()

    return filtered_df.reset_index(drop=True)


# Split one big table into per session recordings.
def _split_into_sessions(df: pd.DataFrame) -> List[pd.DataFrame]:
    available_session_columns = [
        column_name for column_name in SESSION_COLUMN_CANDIDATES if column_name in df.columns
    ]

    grouped_sessions = []
    for _, group_df in df.groupby(available_session_columns, dropna=False, sort=False):
        session_df = filter_data(group_df)
        if not session_df.empty:
            grouped_sessions.append(session_df)

    return grouped_sessions


# Remove activities that have too few recordings.
def _keep_frequent_activities(
    recordings: List[pd.DataFrame],
    min_recordings_per_activity: int,
) -> List[pd.DataFrame]:
    activity_labels = [_session_majority_label(recording) for recording in recordings]
    activity_counts = pd.Series(activity_labels).value_counts()
    valid_activities = activity_counts[
        activity_counts >= min_recordings_per_activity
    ].index

    filtered_recordings = [
        recording
        for recording in recordings
        if _session_majority_label(recording) in valid_activities
    ]

    return filtered_recordings


# Read csv and return sessionized recordings.
def _load_recgym_recordings(
    recgym_csv_path: str,
    min_recordings_per_activity: int,
) -> List[pd.DataFrame]:
    full_df = pd.read_csv(recgym_csv_path)
    recordings = _split_into_sessions(full_df)
   
    return _keep_frequent_activities(recordings, min_recordings_per_activity)



# filtering
def load_filtered_recordings(
    data_path: str,
    min_recordings_per_activity: int = 5,
) -> List[pd.DataFrame]:
    recgym_csv_path = os.path.join(data_path, "RecGym.csv")
    return _load_recgym_recordings(
        recgym_csv_path=recgym_csv_path,
        min_recordings_per_activity=min_recordings_per_activity,
    )


# Convert string activity labels to integer class IDs and add `activityEncoded` column.
def encode_activities(dataframes: List[pd.DataFrame]) -> Dict[str, int]:
    all_activity_values = pd.concat([df["activity"] for df in dataframes], ignore_index=True)
    activity_categories = all_activity_values.astype("category").cat.categories
    activity_to_id = {activity_name: idx for idx, activity_name in enumerate(activity_categories)}

    for index, dataframe in enumerate(dataframes):
        updated_dataframe = dataframe.copy()
        updated_dataframe["activityEncoded"] = updated_dataframe["activity"].map(activity_to_id)
        dataframes[index] = updated_dataframe

    return activity_to_id


# Replace NaN/inf values in feature columns
def clean_imu_columns(dataframes: List[pd.DataFrame], imu_features: Sequence[str]) -> None:
    for index, dataframe in enumerate(dataframes):
        cleaned_dataframe = dataframe.copy()
        cleaned_dataframe[list(imu_features)] = (
            cleaned_dataframe[list(imu_features)]
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
        )
        dataframes[index] = cleaned_dataframe


# Apply per-window smoothing
def preprocess_sample(
    window: np.ndarray,
    y: int,
    smooth_kernel: int = 5,
) -> Tuple[np.ndarray, int]:
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

    return processed_window, y


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

        for session_df in dataframes:
            feature_matrix = session_df[list(features)].values
            session_labels = session_df["activityEncoded"].to_numpy(dtype=np.int64)

            # Sliding window extraction.
            for start_index in range(0, len(feature_matrix) - window_size + 1, step_size):
                end_index = start_index + window_size
                window = feature_matrix[start_index:end_index]
                window_labels = session_labels[start_index:end_index]
                window_label = int(np.bincount(window_labels).argmax()) #label for the window is the most common label in it

                if self.preprocess_fn is not None:
                    processed_window, processed_label = self.preprocess_fn(
                        window,
                        window_label,
                        **self.preprocess_kwargs,
                    )
                else:
                    processed_window, processed_label = window, window_label

                self.samples.append((processed_window, processed_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_window, class_label = self.samples[idx]
        feature_window = torch.tensor(feature_window, dtype=torch.float32)
        class_label = torch.tensor(class_label, dtype=torch.long)
        return feature_window, class_label


    # Split sessions into train/test while keeping label distribution as balanced as possible.
def stratified_session_split(
    data: List[pd.DataFrame],
    labels: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
   
    label_array = np.asarray(labels)
    all_session_indices = np.arange(len(label_array))
    target_train_count = int(train_ratio * len(label_array))

    class_counts = dict(zip(*np.unique(label_array, return_counts=True)))
    rare_classes = {class_id for class_id, count in class_counts.items() if count < 2} #if class has only one session it cant be stratified

    forced_train_indices = np.array(
        [index for index in all_session_indices if label_array[index] in rare_classes],
        dtype=int,
    )
    remaining_indices = np.array(
        [index for index in all_session_indices if label_array[index] not in rare_classes],
        dtype=int,
    )


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

    #check if a class is missing from train split than it is moved form test to train split
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


# Build fully prepared train/test loaders (split, scale, window, dataloaders).
def make_train_test_loaders(
    data: List[pd.DataFrame],
    imu_features: Sequence[str],
    window_size: int,
    step_size: int,
    train_split: float = 0.8,
    batch_size_train: int = 32,
    batch_size_test: int = 1,
):
   

    session_labels = np.array(
        [int(np.bincount(session_df["activityEncoded"].to_numpy(dtype=np.int64)).argmax()) for session_df in data]
    )
    train_indices, test_indices = stratified_session_split(
        data,
        session_labels,
        train_ratio=train_split,
        seed=42,
    )

    train_sessions = [data[index] for index in train_indices]
    test_sessions = [data[index] for index in test_indices]

    scaler = StandardScaler()
    train_feature_table = pd.concat(
        [session_df.loc[:, imu_features] for session_df in train_sessions],
        ignore_index=True,
    )
    scaler.fit(train_feature_table)

    _apply_scaler_inplace(train_sessions, scaler, imu_features)
    _apply_scaler_inplace(test_sessions, scaler, imu_features)

    preprocess_kwargs = dict(smooth_kernel=5)

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


    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    # Extra safety check: no session appears in both splits.
    assert set(map(id, train_sessions)).isdisjoint(
        set(map(id, test_sessions))
    ), "Leak: same session in both splits"

    return train_loader, test_loader, train_dataset, test_dataset
