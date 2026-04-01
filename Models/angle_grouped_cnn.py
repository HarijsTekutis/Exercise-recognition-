import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset


TensorLike = Union[torch.Tensor, np.ndarray]


class GroupCNN1D(nn.Module):
    """Small 1D CNN used by each angle group."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        base_channels: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(num_features, base_channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(base_channels * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(base_channels * 2)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 2 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [batch, time, features], got {tuple(x.shape)}")

        x = x.permute(0, 2, 1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        mean_pool = x.mean(dim=2)
        max_pool, _ = x.max(dim=2)
        features = torch.cat([mean_pool, max_pool], dim=1)

        features = self.dropout(features)
        return self.fc(features)


@dataclass
class GroupTrainingConfig:
    learning_rate: float = 1e-3
    num_epochs: int = 50
    clip_grad_norm: float = 1.0
    patience: int = 7
    batch_size_train: int = 32
    batch_size_val: int = 128


class AngleGroupedCNN(nn.Module):
    """
    Two-stage time-series classifier:
    1) Assign each window to a group from its initial angle.
    2) Run a group-specific CNN to predict global class IDs.
    """

    REQUIRED_FEATURES = ("A_x", "A_y", "A_z", "body_a_x", "body_a_y", "body_a_z")

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        feature_names: Sequence[str],
        points_for_angle: int = 5,
        grouping_method: str = "bins",
        num_groups: int = 3,
        min_group_size: int = 16,
        eps: float = 1e-8,
        cnn_base_channels: int = 64,
        cnn_dropout: float = 0.3,
        random_state: int = 42,
    ):
        super().__init__()

        if num_features <= 0 or num_classes <= 1:
            raise ValueError("num_features must be > 0 and num_classes must be > 1")
        if len(feature_names) != num_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match num_features ({num_features})"
            )
        if points_for_angle <= 0:
            raise ValueError("points_for_angle must be > 0")
        if num_groups <= 0:
            raise ValueError("num_groups must be > 0")
        if grouping_method not in {"bins", "kmeans"}:
            raise ValueError("grouping_method must be one of: {'bins', 'kmeans'}")

        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.feature_names = list(feature_names)

        self.points_for_angle = int(points_for_angle)
        self.grouping_method = grouping_method
        self.num_groups = int(num_groups)
        self.min_group_size = int(max(1, min_group_size))
        self.eps = float(eps)
        self.random_state = int(random_state)

        self.cnn_base_channels = int(cnn_base_channels)
        self.cnn_dropout = float(cnn_dropout)

        self._feature_indices = self._resolve_feature_indices(self.feature_names)

        self.group_cnns = nn.ModuleDict()
        self.grouping_fitted = False

        self.grouping_state: Dict[str, object] = {}
        self.group_centers: Dict[int, float] = {}
        self.available_group_ids: List[int] = []
        self.default_group_id: Optional[int] = None
        self.group_sample_counts: Dict[int, int] = {}
        self.skipped_groups: List[int] = []

    def _resolve_feature_indices(self, feature_names: Sequence[str]) -> Dict[str, int]:
        missing = [name for name in self.REQUIRED_FEATURES if name not in feature_names]
        if missing:
            raise ValueError(
                f"Missing required features for angle computation: {missing}. "
                f"Available features: {list(feature_names)}"
            )
        return {name: int(feature_names.index(name)) for name in self.REQUIRED_FEATURES}

    def _validate_windows(self, windows: torch.Tensor) -> None:
        if windows.ndim != 3:
            raise ValueError(
                f"Expected windows shape [batch, time, features], got {tuple(windows.shape)}"
            )
        if windows.shape[2] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {windows.shape[2]}"
            )

    def compute_initial_angle(self, windows: TensorLike, points_for_angle: Optional[int] = None) -> torch.Tensor:
        """
        Compute per-window initial angle feature as mean of first N per-step angles.
        Angle formula: arccos(dot(A, body_a) / (||A|| * ||body_a||)).
        """
        x = torch.as_tensor(windows, dtype=torch.float32)
        self._validate_windows(x)

        n_points = int(points_for_angle or self.points_for_angle)
        n_points = min(n_points, x.shape[1])
        if n_points <= 0:
            raise ValueError("points_for_angle must be >= 1 and sequence length must be >= 1")

        a_idx = [
            self._feature_indices["A_x"],
            self._feature_indices["A_y"],
            self._feature_indices["A_z"],
        ]
        b_idx = [
            self._feature_indices["body_a_x"],
            self._feature_indices["body_a_y"],
            self._feature_indices["body_a_z"],
        ]

        a_vec = x[:, :n_points, a_idx]
        b_vec = x[:, :n_points, b_idx]

        dot_product = (a_vec * b_vec).sum(dim=-1)
        a_norm = torch.linalg.norm(a_vec, dim=-1)
        b_norm = torch.linalg.norm(b_vec, dim=-1)

        cosine = dot_product / torch.clamp(a_norm * b_norm, min=self.eps)
        cosine = torch.clamp(cosine, min=-1.0, max=1.0)

        angles = torch.arccos(cosine)
        return angles.mean(dim=1)

    def _build_bins(self, angle_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        min_angle = float(np.min(angle_values))
        max_angle = float(np.max(angle_values))

        if np.isclose(min_angle, max_angle):
            # Degenerate case: all windows have almost identical initial angle.
            spread = 1e-3
            min_angle -= spread
            max_angle += spread

        edges = np.linspace(min_angle, max_angle, self.num_groups + 1, dtype=np.float64)
        assignments = np.digitize(angle_values, bins=edges[1:-1], right=False).astype(np.int64)

        centers = np.zeros(self.num_groups, dtype=np.float64)
        for group_id in range(self.num_groups):
            members = angle_values[assignments == group_id]
            if members.size == 0:
                centers[group_id] = float((edges[group_id] + edges[group_id + 1]) / 2.0)
            else:
                centers[group_id] = float(np.mean(members))

        self.grouping_state["bin_edges"] = edges.tolist()
        return assignments, centers

    def _build_kmeans(self, angle_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = int(angle_values.shape[0])
        n_clusters = int(min(self.num_groups, max(1, n_samples)))

        model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        raw_labels = model.fit_predict(angle_values.reshape(-1, 1))
        raw_centers = model.cluster_centers_.reshape(-1)

        # Stabilize IDs by sorting groups from smallest to largest center angle.
        sorted_old_ids = np.argsort(raw_centers)
        old_to_new = {int(old_id): int(new_id) for new_id, old_id in enumerate(sorted_old_ids)}

        assignments = np.array([old_to_new[int(lbl)] for lbl in raw_labels], dtype=np.int64)
        centers = raw_centers[sorted_old_ids]

        return assignments, centers

    def _merge_small_groups(
        self,
        assignments: np.ndarray,
        centers: np.ndarray,
        angle_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        unique_groups, counts = np.unique(assignments, return_counts=True)
        count_by_group = {int(g): int(c) for g, c in zip(unique_groups, counts)}

        major_groups = [group_id for group_id in unique_groups if count_by_group[int(group_id)] >= self.min_group_size]
        if not major_groups:
            major_groups = [int(unique_groups[int(np.argmax(counts))])]

        merged = assignments.copy()
        for group_id in unique_groups:
            group_id = int(group_id)
            if group_id in major_groups:
                continue
            distances = [(int(candidate), abs(float(centers[group_id] - centers[int(candidate)]))) for candidate in major_groups]
            target_group = min(distances, key=lambda item: item[1])[0]
            merged[merged == group_id] = target_group

        final_groups = np.sort(np.unique(merged))
        remap = {int(old_id): int(new_id) for new_id, old_id in enumerate(final_groups)}
        remapped = np.array([remap[int(g)] for g in merged], dtype=np.int64)

        remapped_centers = np.zeros(len(final_groups), dtype=np.float64)
        for old_id, new_id in remap.items():
            member_angles = angle_values[merged == old_id]
            if member_angles.size == 0:
                remapped_centers[new_id] = float(centers[old_id])
            else:
                remapped_centers[new_id] = float(np.mean(member_angles))

        return remapped, remapped_centers

    def fit_grouping(self, windows: TensorLike) -> np.ndarray:
        x = torch.as_tensor(windows, dtype=torch.float32)
        self._validate_windows(x)

        if x.shape[0] == 0:
            raise ValueError("Cannot fit grouping on empty windows")

        angle_values = self.compute_initial_angle(x).detach().cpu().numpy()

        if self.grouping_method == "bins":
            assignments, centers = self._build_bins(angle_values)
        else:
            assignments, centers = self._build_kmeans(angle_values)

        assignments, centers = self._merge_small_groups(assignments, centers, angle_values)

        unique_groups, counts = np.unique(assignments, return_counts=True)
        self.available_group_ids = [int(group_id) for group_id in unique_groups.tolist()]
        self.group_sample_counts = {
            int(group_id): int(count)
            for group_id, count in zip(unique_groups.tolist(), counts.tolist())
        }
        self.group_centers = {
            int(group_id): float(centers[int(group_id)]) for group_id in self.available_group_ids
        }

        self.default_group_id = int(unique_groups[int(np.argmax(counts))])

        self.grouping_state.update(
            {
                "method": self.grouping_method,
                "points_for_angle": self.points_for_angle,
                "num_requested_groups": self.num_groups,
                "num_actual_groups": len(self.available_group_ids),
                "group_centers": {str(k): v for k, v in self.group_centers.items()},
                "group_sample_counts": {str(k): v for k, v in self.group_sample_counts.items()},
                "default_group_id": int(self.default_group_id),
                "min_group_size": int(self.min_group_size),
            }
        )

        # Rebuild per-group CNNs to match fitted groups.
        self.group_cnns = nn.ModuleDict(
            {
                str(group_id): GroupCNN1D(
                    num_features=self.num_features,
                    num_classes=self.num_classes,
                    base_channels=self.cnn_base_channels,
                    dropout=self.cnn_dropout,
                )
                for group_id in self.available_group_ids
            }
        )

        self.grouping_fitted = True
        return assignments

    def _assign_from_angles(self, angle_values: np.ndarray) -> np.ndarray:
        if not self.grouping_fitted:
            raise RuntimeError("Grouping is not fitted. Call fit_grouping before assigning groups.")

        if self.grouping_method == "bins":
            bin_edges = np.array(self.grouping_state.get("bin_edges", []), dtype=np.float64)
            if bin_edges.size < 2:
                raise RuntimeError("Invalid bin edges in grouping state")
            groups = np.digitize(angle_values, bins=bin_edges[1:-1], right=False).astype(np.int64)
        else:
            ordered_group_ids = np.array(sorted(self.group_centers.keys()), dtype=np.int64)
            ordered_centers = np.array([self.group_centers[int(g)] for g in ordered_group_ids], dtype=np.float64)
            distances = np.abs(angle_values[:, None] - ordered_centers[None, :])
            nearest_idx = np.argmin(distances, axis=1)
            groups = ordered_group_ids[nearest_idx].astype(np.int64)

        # Resolve unseen or removed group IDs by nearest trained group center.
        available = np.array(self.available_group_ids, dtype=np.int64)
        available_centers = np.array([self.group_centers[int(g)] for g in available], dtype=np.float64)

        resolved = groups.copy()
        for idx, group_id in enumerate(groups.tolist()):
            if int(group_id) in self.available_group_ids:
                continue
            distance = np.abs(angle_values[idx] - available_centers)
            resolved[idx] = int(available[int(np.argmin(distance))])

        return resolved

    def assign_groups(self, windows: TensorLike) -> np.ndarray:
        x = torch.as_tensor(windows, dtype=torch.float32)
        self._validate_windows(x)

        angle_values = self.compute_initial_angle(x).detach().cpu().numpy()
        return self._assign_from_angles(angle_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_windows(x)
        if not self.grouping_fitted:
            raise RuntimeError("Grouping is not fitted. Call fit_grouping before forward/predict.")

        device = x.device
        groups = self.assign_groups(x.detach().cpu())
        logits = torch.zeros((x.shape[0], self.num_classes), dtype=torch.float32, device=device)

        for group_id in np.unique(groups):
            group_id = int(group_id)
            if str(group_id) not in self.group_cnns:
                if self.default_group_id is None or str(self.default_group_id) not in self.group_cnns:
                    raise RuntimeError("No trained group CNN available for fallback")
                group_id = int(self.default_group_id)

            indices = np.where(groups == group_id)[0]
            if indices.size == 0:
                continue

            idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
            group_model = self.group_cnns[str(group_id)].to(device)
            group_logits = group_model(x.index_select(dim=0, index=idx_tensor))
            logits.index_copy_(0, idx_tensor, group_logits)

        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        windows: TensorLike,
        device: torch.device,
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return probabilities and assigned groups for each input window."""
        if not self.grouping_fitted:
            raise RuntimeError("Grouping is not fitted. Train or fit grouping before inference.")

        x = torch.as_tensor(windows, dtype=torch.float32)
        self._validate_windows(x)

        self.eval()
        x = x.to(device)
        groups = self.assign_groups(x.detach().cpu())

        probabilities = torch.zeros((x.shape[0], self.num_classes), dtype=torch.float32, device=device)

        unique_groups = np.unique(groups)
        for group_id in unique_groups:
            group_id = int(group_id)
            model_key = str(group_id)
            if model_key not in self.group_cnns:
                model_key = str(self.default_group_id)
            if model_key not in self.group_cnns:
                raise RuntimeError("No trained group CNN available for prediction")

            group_model = self.group_cnns[model_key].to(device)
            group_model.eval()

            indices = np.where(groups == group_id)[0]
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                idx_tensor = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
                logits = group_model(x.index_select(dim=0, index=idx_tensor))
                probs = torch.softmax(logits, dim=1)
                probabilities.index_copy_(0, idx_tensor, probs)

        return probabilities.detach().cpu().numpy(), groups

    @torch.no_grad()
    def predict(
        self,
        windows: TensorLike,
        device: torch.device,
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return predicted class, confidence, and group ID per window."""
        probs, groups = self.predict_proba(windows=windows, device=device, batch_size=batch_size)
        predictions = np.argmax(probs, axis=1).astype(np.int64)
        confidences = np.max(probs, axis=1).astype(np.float32)
        return predictions, confidences, groups.astype(np.int64)


def _collect_loader_tensors(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    windows: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []

    for batch_x, batch_y in loader:
        windows.append(batch_x.detach().cpu())
        labels.append(batch_y.detach().cpu().long())

    if not windows:
        raise ValueError("DataLoader is empty")

    x = torch.cat(windows, dim=0)
    y = torch.cat(labels, dim=0)

    if x.ndim != 3:
        raise ValueError(f"Expected windows tensor with 3 dims, got {tuple(x.shape)}")
    if y.ndim != 1:
        y = y.view(-1)

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Number of windows ({x.shape[0]}) does not match labels ({y.shape[0]})"
        )

    return x.float(), y.long()


def _compute_group_class_weights(y_group: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(y_group, minlength=num_classes).to(torch.float32)
    present_mask = counts > 0

    weights = torch.zeros(num_classes, dtype=torch.float32)
    if bool(present_mask.any()):
        n_present = present_mask.sum().item()
        total = counts[present_mask].sum()
        weights[present_mask] = total / (n_present * counts[present_mask])
        weights[present_mask] = torch.clamp(weights[present_mask], min=0.5, max=5.0)
    else:
        weights[:] = 1.0

    return weights.to(device)


def _make_subset_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    idx = torch.as_tensor(indices, dtype=torch.long)
    subset_x = x.index_select(dim=0, index=idx)
    subset_y = y.index_select(dim=0, index=idx)
    return DataLoader(TensorDataset(subset_x, subset_y), batch_size=batch_size, shuffle=shuffle)


def train_AngleGroupedCNN(
    model: AngleGroupedCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float = 1e-3,
    num_epochs: int = 50,
    clip_grad_norm: float = 1.0,
    patience: int = 7,
    best_model_dir: str = "model_comparison_results/angle_grouped_cnn_checkpoints",
    min_samples_to_train_group: int = 8,
) -> Dict[str, object]:
    """Train one 1D CNN per angle group with class weighting and early stopping."""
    if min_samples_to_train_group <= 0:
        raise ValueError("min_samples_to_train_group must be > 0")

    os.makedirs(best_model_dir, exist_ok=True)

    train_x, train_y = _collect_loader_tensors(train_loader)
    val_x, val_y = _collect_loader_tensors(val_loader)

    model._validate_windows(train_x)
    model._validate_windows(val_x)

    train_groups = model.fit_grouping(train_x)
    val_groups = model.assign_groups(val_x)

    config = GroupTrainingConfig(
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        clip_grad_norm=clip_grad_norm,
        patience=patience,
        batch_size_train=getattr(train_loader, "batch_size", 32) or 32,
        batch_size_val=max(64, int((getattr(val_loader, "batch_size", 1) or 1) * 8)),
    )

    by_group_history: Dict[int, Dict[str, object]] = {}
    trained_group_ids: List[int] = []
    skipped_group_ids: List[int] = []

    for group_id in model.available_group_ids:
        train_idx = np.where(train_groups == int(group_id))[0]
        val_idx = np.where(val_groups == int(group_id))[0]

        if train_idx.size < int(min_samples_to_train_group):
            skipped_group_ids.append(int(group_id))
            by_group_history[int(group_id)] = {
                "status": "skipped",
                "reason": f"too_few_train_samples ({train_idx.size})",
                "train_losses": [],
                "train_accuracies": [],
                "val_losses": [],
                "val_accuracies": [],
                "train_samples": int(train_idx.size),
                "val_samples": int(val_idx.size),
                "best_model_path": "",
            }
            continue

        group_model = model.group_cnns[str(group_id)].to(device)
        class_weights = _compute_group_class_weights(train_y[train_idx], model.num_classes, device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        optimizer = torch.optim.AdamW(group_model.parameters(), lr=config.learning_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs, eta_min=1e-6
        )

        group_train_loader = _make_subset_loader(
            x=train_x,
            y=train_y,
            indices=train_idx,
            batch_size=config.batch_size_train,
            shuffle=True,
        )
        group_val_loader = _make_subset_loader(
            x=val_x,
            y=val_y,
            indices=val_idx,
            batch_size=config.batch_size_val,
            shuffle=False,
        )

        group_history = {
            "status": "trained",
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
            "train_samples": int(train_idx.size),
            "val_samples": int(val_idx.size),
            "best_model_path": os.path.join(best_model_dir, f"angle_group_{group_id}.pt"),
        }

        best_loss = float("inf")
        no_improvement = 0

        for _epoch in range(config.num_epochs):
            group_model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for batch_inputs, batch_labels in group_train_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                logits = group_model(batch_inputs)
                loss = criterion(logits, batch_labels)

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(group_model.parameters(), max_norm=config.clip_grad_norm)
                optimizer.step()

                running_loss += float(loss.item())
                preds = logits.argmax(dim=1)
                running_correct += int((preds == batch_labels).sum().item())
                running_total += int(batch_labels.shape[0])

            train_loss = running_loss / max(1, len(group_train_loader))
            train_acc = 100.0 * running_correct / max(1, running_total)

            group_model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            eval_loader = group_val_loader if len(group_val_loader) > 0 else group_train_loader
            with torch.no_grad():
                for batch_inputs, batch_labels in eval_loader:
                    batch_inputs = batch_inputs.to(device)
                    batch_labels = batch_labels.to(device)

                    logits = group_model(batch_inputs)
                    loss = criterion(logits, batch_labels)

                    val_loss_sum += float(loss.item())
                    preds = logits.argmax(dim=1)
                    val_correct += int((preds == batch_labels).sum().item())
                    val_total += int(batch_labels.shape[0])

            val_loss = val_loss_sum / max(1, len(eval_loader))
            val_acc = 100.0 * val_correct / max(1, val_total)

            group_history["train_losses"].append(float(train_loss))
            group_history["train_accuracies"].append(float(train_acc))
            group_history["val_losses"].append(float(val_loss))
            group_history["val_accuracies"].append(float(val_acc))

            if val_loss < best_loss:
                best_loss = val_loss
                no_improvement = 0
                torch.save(group_model.state_dict(), group_history["best_model_path"])
            else:
                no_improvement += 1
                if no_improvement >= config.patience:
                    break

            scheduler.step()

        if os.path.exists(group_history["best_model_path"]):
            state = torch.load(group_history["best_model_path"], map_location=device)
            group_model.load_state_dict(state)

        trained_group_ids.append(int(group_id))
        by_group_history[int(group_id)] = group_history

    model.skipped_groups = skipped_group_ids

    if not trained_group_ids:
        raise RuntimeError(
            "No group model was trained. Consider lowering min_samples_to_train_group "
            "or reducing num_groups/min_group_size."
        )

    available_after_training = sorted(trained_group_ids)
    model.available_group_ids = available_after_training

    if model.default_group_id not in available_after_training:
        most_common_group = max(
            model.group_sample_counts.items(),
            key=lambda item: item[1],
        )[0]
        model.default_group_id = int(most_common_group)
        if model.default_group_id not in available_after_training:
            model.default_group_id = int(available_after_training[0])

    # Aggregate plotting-compatible history from trained groups.
    trained_histories: List[Dict[str, Any]] = []
    for group_id in available_after_training:
        group_hist = by_group_history[group_id]
        train_losses_group = group_hist.get("train_losses", [])
        if group_hist.get("status") == "trained" and isinstance(train_losses_group, list) and len(train_losses_group) > 0:
            trained_histories.append(group_hist)

    if trained_histories:
        min_len = min(len(h["train_losses"]) for h in trained_histories)
        train_losses = [
            float(np.mean([h["train_losses"][i] for h in trained_histories])) for i in range(min_len)
        ]
        val_losses = [
            float(np.mean([h["val_losses"][i] for h in trained_histories])) for i in range(min_len)
        ]
        train_accuracies = [
            float(np.mean([h["train_accuracies"][i] for h in trained_histories])) for i in range(min_len)
        ]
        val_accuracies = [
            float(np.mean([h["val_accuracies"][i] for h in trained_histories])) for i in range(min_len)
        ]
    else:
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    history: Dict[str, object] = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "by_group": by_group_history,
        "grouping": model.grouping_state,
        "trained_groups": available_after_training,
        "skipped_groups": skipped_group_ids,
        "best_model_dir": best_model_dir,
    }

    return history


@torch.no_grad()
def evaluate_AngleGroupedCNN(
    model: AngleGroupedCNN,
    data_loader: DataLoader,
    device: torch.device,
    history: Optional[Dict[str, object]] = None,
    best_model_path: Optional[str] = None,
) -> Dict[str, object]:
    """Evaluate on a loader and return a JSON/CSV-friendly result dictionary."""
    x, y = _collect_loader_tensors(data_loader)
    y_true = y.numpy().astype(np.int64)

    y_pred, confidences, groups = model.predict(x, device=device)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(model.num_classes)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    result = {
        "history": history if history is not None else {},
        "confusion_matrix": cm.tolist(),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        },
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "confidence": confidences.tolist(),
        "assigned_groups": groups.tolist(),
        "param_count": int(
            sum(
                p.numel()
                for group_id in model.available_group_ids
                for p in model.group_cnns[str(group_id)].parameters()
            )
        ),
        "best_model_path": best_model_path or "",
        "grouping": model.grouping_state,
    }
    return result
