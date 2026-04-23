from typing import Dict, List, Optional
import os

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sktime.classification.kernel_based import RocketClassifier


def _to_numpy(x):
	if isinstance(x, torch.Tensor):
		return x.detach().cpu().numpy()
	return np.asarray(x)


def _to_sktime_3d(x: np.ndarray) -> np.ndarray:
	"""Convert windowed data to sktime panel shape: (n_samples, n_channels, n_timepoints)."""
	x = np.asarray(x)
	if x.ndim != 3:
		raise ValueError(f"Expected 3D input (n_samples, window, features), got shape {x.shape}")

	# Data loaders provide (n_samples, window_size, num_features); sktime expects channels-first.
	x = np.transpose(x, (0, 2, 1))


	return np.ascontiguousarray(x, dtype=np.float64)


def _collect_from_loader(data_loader):
	xs, ys = [], []
	for batch_x, batch_y in data_loader:
		xs.append(_to_numpy(batch_x))
		ys.append(_to_numpy(batch_y))
	if not xs:
		raise ValueError("data_loader is empty")
	x = np.concatenate(xs, axis=0)
	y = np.concatenate(ys, axis=0).astype(np.int64)
	return x, y


class RocketModel:
	

	def __init__(
		self,
		num_features: int = 9,
		num_classes: int = 6,
		num_kernels: int = 10000,
		n_jobs: int = -1,
		random_state: int = 42,
	):
		del n_jobs
		self.num_features = int(num_features)
		self.num_classes = int(num_classes)
		self.classifier = RocketClassifier(
			num_kernels=num_kernels,
			rocket_transform="rocket",
			random_state=random_state,
		)

	def to(self, device):
		# Exists only for compatibility with shared training code.
		return self

	def load_state_dict(self, state_dict, strict: bool = True):
		del strict
		# Keep compatibility if common loading logic is used.
		if isinstance(state_dict, RocketClassifier):
			self.classifier = state_dict
		elif isinstance(state_dict, dict) and "classifier" in state_dict:
			self.classifier = state_dict["classifier"]
		else:
			raise ValueError("Invalid state_dict for RocketModel")
		return self

	def state_dict(self):
		return {"classifier": self.classifier}

	def fit(self, x, y):
		self.classifier.fit(
			_to_sktime_3d(x),
			np.asarray(y, dtype=np.int64),
		)

	def predict(self, x):
		return self.classifier.predict(_to_sktime_3d(x))

	def predict_proba(self, x):
		return self.classifier.predict_proba(_to_sktime_3d(x))


Rocket = RocketModel


def train_rocket(
	model: RocketModel,
	train_loader,
	val_loader,
	class_weights=None,
	device=None,
	learning_rate: float = 0.001,
	num_epochs: int = 1,
	clip_grad_norm: float = 1.0,
	patience: int = 7,
	best_model_path: str = "rocket.pt",
) -> Dict[str, List[float]]:
	del class_weights, device, learning_rate, num_epochs, clip_grad_norm, patience

	x_train, y_train = _collect_from_loader(train_loader)
	x_val, y_val = _collect_from_loader(val_loader)

	model.fit(x_train, y_train)

	train_pred = model.predict(x_train).astype(np.int64)
	val_pred = model.predict(x_val).astype(np.int64)
	train_acc = float(accuracy_score(y_train, train_pred) * 100.0)
	val_acc = float(accuracy_score(y_val, val_pred) * 100.0)

	history = {
		"train_losses": [1.0 - (train_acc / 100.0)],
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


def evaluate_rocket(
	model: RocketModel,
	data_loader,
	device=None,
	history: Optional[Dict[str, object]] = None,
	best_model_path: str = "",
) -> Dict[str, object]:
	del device

	print("Evaluating ROCKET model...")
	x, y_true = _collect_from_loader(data_loader, desc="Test data")
	print(f"Loaded {len(x)} test samples")
	
	print("Making predictions...")
	y_pred = model.predict(x).astype(np.int64)
	probs = model.predict_proba(x)
	print("Predictions complete!")

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
		"param_count": int(getattr(model.classifier, "num_kernels", 0)),
		"best_model_path": best_model_path,
	}
