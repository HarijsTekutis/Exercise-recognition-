from typing import Dict, List, Optional
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _flatten(x: np.ndarray) -> np.ndarray: # Each window becomes a single column
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(x.shape[0], -1)


def _collect_from_loader(data_loader): # Collects all data from a PyTorch DataLoader into numpy arrays in memory
    xs, ys = [], []
    for batch_x, batch_y in data_loader:
        xs.append(_to_numpy(batch_x))
        ys.append(_to_numpy(batch_y))
    if not xs:
        raise ValueError("data_loader is empty")
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    return x, y



class RandomForestModel:
    """Simple sklearn RandomForest wrapper."""

    def __init__(
        self,
        num_features: int = 9,
        num_classes: int = 6,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        class_weight: Optional[str] = None,
    ):
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight=class_weight,
        )

    def to(self, device):
        # Exists only for compatibility with shared training code.
        return self

    def load_state_dict(self, state_dict, strict: bool = True):
        # Keep compatibility if common loading logic is used.
        if isinstance(state_dict, RandomForestClassifier):
            self.classifier = state_dict
        elif isinstance(state_dict, dict) and "classifier" in state_dict:
            self.classifier = state_dict["classifier"]
        else:
            raise ValueError("Invalid state_dict for RandomForestModel")
        return self

    def state_dict(self):
        return {"classifier": self.classifier}

    def fit(self, x, y, sample_weight=None):
        self.classifier.fit(_flatten(x), np.asarray(y, dtype=np.int64), sample_weight=sample_weight)

    def predict(self, x):
        return self.classifier.predict(_flatten(x))

    def predict_proba(self, x):
        return self.classifier.predict_proba(_flatten(x))


RandomForest = RandomForestModel


def train_random_forest(
    model: RandomForestModel,
    train_loader,
    val_loader,
    class_weights=None,
    device=None,
    learning_rate: float = 0.001,
    num_epochs: int = 1,
    clip_grad_norm: float = 1.0,
    patience: int = 7,
    best_model_path: str = "random_forest.pt",
) -> Dict[str, List[float]]:
    del device, learning_rate, num_epochs, clip_grad_norm, patience # Not used by RandomForest

    x_train, y_train = _collect_from_loader(train_loader)
    x_val, y_val = _collect_from_loader(val_loader)

    sample_weight = None
    if class_weights is not None:
        weights = _to_numpy(class_weights).reshape(-1)
        sample_weight = weights[y_train]

    model.fit(x_train, y_train, sample_weight=sample_weight)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    train_acc = float(accuracy_score(y_train, train_pred) * 100.0)
    val_acc = float(accuracy_score(y_val, val_pred) * 100.0)

    history = {
        "train_losses": [1.0 - (train_acc / 100.0)], # Using 1 - accuracy  for loss since RandomForest doesn't have a built-in loss
        "train_accuracies": [train_acc],
        "val_losses": [1.0 - (val_acc / 100.0)],
        "val_accuracies": [val_acc],
    }

    checkpoint_dir = os.path.dirname(best_model_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), best_model_path)
    print(f"Train accuracy: {train_acc:.2f}%")
    print(f"Val accuracy: {val_acc:.2f}%")
    return history


def evaluate_random_forest(
    model: RandomForestModel,
    data_loader,
    device=None,
    history: Optional[Dict[str, object]] = None,
    best_model_path: str = "",
) -> Dict[str, object]:
    del device

    x, y_true = _collect_from_loader(data_loader)
    y_pred = model.predict(x).astype(np.int64)
    probs = model.predict_proba(x)

    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    return {
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
        "confidence": probs.max(axis=1).tolist(),
        "param_count": int(model.classifier.n_estimators),
        "best_model_path": best_model_path,
    }
