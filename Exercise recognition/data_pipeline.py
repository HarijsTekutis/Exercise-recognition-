import os
import warnings
from typing import Dict, List, NamedTuple, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import random

def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#used features for training
IMU_FEATURES = [
    "A_x",
    "A_y",
    "A_z",
    "G_x",
    "G_y",
    "G_z",
    "body_a_x",
    "body_a_y",
    "body_a_z",
]

ACTIVITY_COLUMN = "Workout"
SUBJECT_COLUMN = "Subject"
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


def _scale_sessions(
    sessions: List[pd.DataFrame],
    scaler: StandardScaler,
    imu_features: Sequence[str],
) -> List[pd.DataFrame]:
    """Return scaled copies of the given sessions, keeping only the columns needed downstream.

    Copies rather than scaling in place: the caller's session list is shared across calls
    (Optuna re-splits the same list every trial), so in-place scaling would standardize
    already standardized frames again and again.
    """
    feature_columns = list(imu_features)
    scaled_sessions = []

    for session_df in sessions:
        kept_columns = feature_columns + ["activityEncoded"]
        if "subject" in session_df.columns:
            kept_columns.append("subject")

        scaled_df = session_df.loc[:, kept_columns].copy()
        scaled_df.loc[:, feature_columns] = scaler.transform(session_df.loc[:, feature_columns])
        scaled_sessions.append(scaled_df)

    return scaled_sessions


# Keep only required IMU + label columns and standardize label name to `activity`.
# The subject id is carried along so splits can be made subject independent later.
def filter_data(df: pd.DataFrame) -> pd.DataFrame:

    _validate_recgym_schema(df)

    columns_to_keep = list(IMU_FEATURES) + [ACTIVITY_COLUMN]
    if SUBJECT_COLUMN in df.columns:
        columns_to_keep.append(SUBJECT_COLUMN)

    filtered_df = df.loc[:, columns_to_keep].copy()
    filtered_df = filtered_df.rename(
        columns={ACTIVITY_COLUMN: "activity", SUBJECT_COLUMN: "subject"}
    )
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


# Read the subject id of every session. Each session comes from exactly one subject
# because sessions are grouped by (Subject, Position, Session).
def get_session_subjects(sessions: List[pd.DataFrame]) -> np.ndarray:
    sessions_without_subject = [
        index for index, session_df in enumerate(sessions) if "subject" not in session_df.columns
    ]
    if sessions_without_subject:
        raise ValueError(
            f"Sessions {sessions_without_subject} have no 'subject' column, so a "
            f"subject-independent split is not possible. Check that the source CSV has a "
            f"'{SUBJECT_COLUMN}' column."
        )

    return np.array([session_df["subject"].iloc[0] for session_df in sessions])


# Split sessions into train/val/test by whole subjects.
def subject_independent_split(
    subject_ids: Sequence,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign every subject to exactly one split and return the session indices per split.

    Splitting by subject (instead of by session) means the model is always validated and
    tested on people whose data it has never seen. Session-level splits leak person
    specific traits - body shape, sensor placement, movement style - across the splits and
    inflate the scores.

    Ratios are targets on the number of *sessions*, not subjects, since subjects can
    contribute different numbers of sessions. Each split is guaranteed at least one subject.
    """
    subject_array = np.asarray(subject_ids)
    unique_subjects = np.unique(subject_array)

    if len(unique_subjects) < 3:
        raise ValueError(
            f"A three-way subject-independent split needs at least 3 subjects, "
            f"found {len(unique_subjects)}."
        )
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            f"train_ratio and val_ratio must be positive and sum to less than 1 "
            f"(got {train_ratio} + {val_ratio} = {train_ratio + val_ratio})."
        )

    shuffled_subjects = np.random.default_rng(seed).permutation(unique_subjects)
    sessions_per_subject = {
        subject: int(np.sum(subject_array == subject)) for subject in unique_subjects
    }

    # Place the subjects contributing the most sessions first: assigning the big ones
    # while every split is still empty is what keeps the final sizes close to target.
    ordered_subjects = sorted(
        shuffled_subjects,
        key=lambda subject: sessions_per_subject[subject],
        reverse=True,
    )

    total_sessions = len(subject_array)
    split_names = ["train", "val", "test"]
    target_sessions = np.array(
        [
            train_ratio * total_sessions,
            val_ratio * total_sessions,
            (1.0 - train_ratio - val_ratio) * total_sessions,
        ]
    )
    assigned_sessions = np.zeros(3)
    subjects_per_split: Dict[str, List] = {name: [] for name in split_names}

    # Give each subject to whichever split is furthest below its session target. Filling
    # train to capacity first instead would starve whichever split is filled last.
    for subject in ordered_subjects:
        split_position = int(np.argmax(target_sessions - assigned_sessions))
        subjects_per_split[split_names[split_position]].append(subject)
        assigned_sessions[split_position] += sessions_per_subject[subject]

    # Guarantee every split gets at least one subject by taking from the largest split.
    for split_name in split_names:
        if subjects_per_split[split_name]:
            continue
        donor_name = max(split_names, key=lambda name: len(subjects_per_split[name]))
        if len(subjects_per_split[donor_name]) < 2:
            raise ValueError(
                f"Cannot give every split at least one subject with "
                f"{len(unique_subjects)} subjects."
            )
        subjects_per_split[split_name].append(subjects_per_split[donor_name].pop())

    def session_indices_for(split_subjects: List) -> np.ndarray:
        return np.sort(np.where(np.isin(subject_array, split_subjects))[0])

    train_subjects = subjects_per_split["train"]
    val_subjects = subjects_per_split["val"]
    test_subjects = subjects_per_split["test"]

    return (
        session_indices_for(train_subjects),
        session_indices_for(val_subjects),
        session_indices_for(test_subjects),
    )


# Warn when a split does not cover every activity class.
def _warn_on_missing_classes(
    split_name: str,
    sessions: List[pd.DataFrame],
    all_class_ids: set,
) -> None:
    present_class_ids = set()
    for session_df in sessions:
        present_class_ids.update(session_df["activityEncoded"].unique().tolist())

    missing_class_ids = all_class_ids - present_class_ids
    if missing_class_ids:
        warnings.warn(
            f"{split_name} split is missing activity classes {sorted(missing_class_ids)}. "
            f"Metrics for those classes will be meaningless on this split."
        )


class SplitLoaders(NamedTuple):
    """Everything a training script needs from a three-way subject-independent split."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: "IMUDataset"
    val_dataset: "IMUDataset"
    test_dataset: "IMUDataset"
    subjects: Dict[str, List]


# Build train/val/test loaders from a subject-independent split.
def make_train_val_test_loaders(
    data: List[pd.DataFrame],
    imu_features: Sequence[str],
    window_size: int,
    step_size: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    batch_size_train: int = 32,
    batch_size_val: int = 1,
    batch_size_test: int = 1,
    seed: int = 42,
) -> SplitLoaders:
    """Split by subject, fit the scaler on train only, and window each split.

    Use `val_loader` for early stopping and hyperparameter search. Touch `test_loader`
    only once, after every hyperparameter is frozen - otherwise the reported score is
    tuned on the data it claims to be held out from.
    """
    session_subjects = get_session_subjects(data)
    train_indices, val_indices, test_indices = subject_independent_split(
        session_subjects,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_sessions = [data[index] for index in train_indices]
    val_sessions = [data[index] for index in val_indices]
    test_sessions = [data[index] for index in test_indices]

    # Fit on train only: the scaler is part of the model, and fitting it on val/test
    # would leak their distribution into training.
    scaler = StandardScaler()
    train_feature_table = pd.concat(
        [session_df.loc[:, imu_features] for session_df in train_sessions],
        ignore_index=True,
    )
    scaler.fit(train_feature_table)

    scaled_train_sessions = _scale_sessions(train_sessions, scaler, imu_features)
    scaled_val_sessions = _scale_sessions(val_sessions, scaler, imu_features)
    scaled_test_sessions = _scale_sessions(test_sessions, scaler, imu_features)

    all_class_ids = set()
    for session_df in data:
        all_class_ids.update(session_df["activityEncoded"].unique().tolist())
    _warn_on_missing_classes("TRAIN", scaled_train_sessions, all_class_ids)
    _warn_on_missing_classes("VAL", scaled_val_sessions, all_class_ids)
    _warn_on_missing_classes("TEST", scaled_test_sessions, all_class_ids)

    preprocess_kwargs = dict(smooth_kernel=5)

    def build_dataset(sessions: List[pd.DataFrame]) -> IMUDataset:
        return IMUDataset(
            dataframes=sessions,
            features=imu_features,
            window_size=window_size,
            step_size=step_size,
            preprocess_fn=preprocess_sample,
            preprocess_kwargs=preprocess_kwargs,
        )

    train_dataset = build_dataset(scaled_train_sessions)
    val_dataset = build_dataset(scaled_val_sessions)
    test_dataset = build_dataset(scaled_test_sessions)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    subjects_per_split = {
        "train": sorted(set(session_subjects[train_indices].tolist())),
        "val": sorted(set(session_subjects[val_indices].tolist())),
        "test": sorted(set(session_subjects[test_indices].tolist())),
    }

    # No subject may appear in more than one split - this is the property the whole
    # split exists to guarantee, so check it rather than trust it.
    train_subject_set = set(subjects_per_split["train"])
    val_subject_set = set(subjects_per_split["val"])
    test_subject_set = set(subjects_per_split["test"])
    assert train_subject_set.isdisjoint(val_subject_set), "Leak: subject in TRAIN and VAL"
    assert train_subject_set.isdisjoint(test_subject_set), "Leak: subject in TRAIN and TEST"
    assert val_subject_set.isdisjoint(test_subject_set), "Leak: subject in VAL and TEST"

    print(
        f"Subject-independent split (seed={seed}): "
        f"train subjects {subjects_per_split['train']} "
        f"({len(train_sessions)} sessions, {len(train_dataset)} windows) | "
        f"val subjects {subjects_per_split['val']} "
        f"({len(val_sessions)} sessions, {len(val_dataset)} windows) | "
        f"test subjects {subjects_per_split['test']} "
        f"({len(test_sessions)} sessions, {len(test_dataset)} windows)"
    )

    return SplitLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        subjects=subjects_per_split,
    )


    # Split sessions into train/test while keeping label distribution as balanced as possible.
    # Superseded by subject_independent_split: this splits by session, so the same person
    # can land in both train and test. Kept only so older scripts keep importing.
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


# Deprecated two-way wrapper kept so existing scripts keep running.
def make_train_test_loaders(
    data: List[pd.DataFrame],
    imu_features: Sequence[str],
    window_size: int,
    step_size: int,
    train_split: float = 0.8,
    batch_size_train: int = 32,
    batch_size_test: int = 1,
):
    """Return the TRAIN and VALIDATION halves of a subject-independent three-way split.

    Old callers used the returned second loader both for early stopping and for their
    final reported score, which meant hyperparameters were tuned on the test set. This
    wrapper hands back the validation split instead, so the test subjects stay untouched
    until something explicitly calls make_train_val_test_loaders.

    `train_split` is ignored; the three-way ratios are used instead.
    """
    warnings.warn(
        "make_train_test_loaders is deprecated - use make_train_val_test_loaders. "
        "It now returns the train and VALIDATION loaders of a subject-independent "
        f"three-way split, and ignores train_split={train_split}.",
        DeprecationWarning,
        stacklevel=2,
    )

    splits = make_train_val_test_loaders(
        data=data,
        imu_features=imu_features,
        window_size=window_size,
        step_size=step_size,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_test,
    )

    return splits.train_loader, splits.val_loader, splits.train_dataset, splits.val_dataset


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(train_dataset, num_classes: int, device: torch.device) -> torch.Tensor:
    # Collect integer class labels from all training windows.
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels, minlength=num_classes)

    # Avoid division by zero for classes not present in the sampled windows.
    safe_counts = np.maximum(class_counts, 1)
    class_weights_np = class_counts.sum() / (num_classes * safe_counts)

    
    class_weights_np = class_weights_np.astype(np.float32)
    return torch.tensor(class_weights_np, dtype=torch.float32, device=device)


def build_training_objects(
    model: torch.nn.Module,
    class_weights: torch.Tensor,
    learning_rate: float,
    num_epochs: int,
):
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    return criterion, optimizer, scheduler