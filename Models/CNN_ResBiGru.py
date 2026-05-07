import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline import build_training_objects


class ResBiGRUBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Per-direction input size:
        # - first block: both directions receive the full CNN output (input_size=128)
        # - later blocks: each direction receives its own half (hidden_size=64)
        per_dir = input_size if input_size != hidden_size * 2 else hidden_size

        self.gru_f = nn.GRU(per_dir, hidden_size, batch_first=True)
        self.gru_b = nn.GRU(per_dir, hidden_size, batch_first=True)

        self.proj = (
            nn.Linear(per_dir, hidden_size)
            if per_dir != hidden_size
            else nn.Identity()
        )

        self.ln_f = nn.LayerNorm(hidden_size)   # one LN per direction
        self.ln_b = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.hidden_size * 2:
            # layers 2+: split into the two per-direction streams
            x_f = x[..., :self.hidden_size]
            x_b = x[..., self.hidden_size:]
        else:
            # first layer: both directions share the same input
            x_f = x_b = x

        # Forward direction (left → right)
        out_f, _ = self.gru_f(x_f)
        h_f = self.ln_f(self.proj(x_f) + out_f)

        # Backward direction (right → left, then flip back)
        out_b, _ = self.gru_b(torch.flip(x_b, dims=[1]))
        out_b = torch.flip(out_b, dims=[1])
        h_b = self.ln_b(self.proj(x_b) + out_b)

        return torch.cat([h_f, h_b], dim=-1)


class Model_1D_CNN_ResBiGRU(nn.Module):

    def __init__(self, num_classes: int = 6, num_features: int = 9, hidden_size: int = 64, gru_layers: int = 2):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.res_bigru_blocks = nn.ModuleList()
        self.res_bigru_blocks.append(ResBiGRUBlock(input_size=128, hidden_size=hidden_size))
        for _ in range(1, gru_layers):
            self.res_bigru_blocks.append(ResBiGRUBlock(input_size=hidden_size * 2, hidden_size=hidden_size))

        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input from dataset: (batch, time, features)
        x = x.permute(0, 2, 1)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        # (batch, time, channels)
        x = x.permute(0, 2, 1)
        for res_bigru in self.res_bigru_blocks:
            x = res_bigru(x)

        # AdaptiveAvgPool1d expects channel-first.
        x = x.permute(0, 2, 1)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


CNNResBiGRU = Model_1D_CNN_ResBiGRU


def train_cnn_resbigru(
    model: Model_1D_CNN_ResBiGRU,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    device: torch.device,
    learning_rate: float = 0.001,
    num_epochs: int = 50,
    clip_grad_norm: float = 1.0,
    patience: int = 7,
    best_model_path: str = "CNN_ResBiGRU.pt",
) -> Dict[str, List[float]]:
    
    criterion, optimizer, scheduler = build_training_objects(
        model, class_weights, learning_rate, num_epochs
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    history = {
        "train_losses": [],
        "train_accuracies": [],
        "val_losses": [],
        "val_accuracies": [],
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_examples = 0

        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()

            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)

            if torch.isnan(loss):
                print("NaN detected! Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

            running_loss += loss.item()
            predicted_classes = logits.argmax(dim=1)
            correct_predictions += (predicted_classes == batch_labels).sum().item()
            total_examples += batch_labels.size(0)

        train_accuracy = 100 * correct_predictions / total_examples if total_examples > 0 else 0.0
        average_train_loss = running_loss / len(train_loader)
        history["train_losses"].append(average_train_loss)
        history["train_accuracies"].append(train_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}: loss={average_train_loss:.4f}, accuracy={train_accuracy:.2f}%, lr={current_lr:.2e}"
        )

        model.eval()
        validation_loss_sum = 0.0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():
            for validation_inputs, validation_labels in val_loader:
                validation_inputs = validation_inputs.to(device)
                validation_labels = validation_labels.to(device)

                validation_logits = model(validation_inputs)
                validation_loss = criterion(validation_logits, validation_labels)

                validation_loss_sum += validation_loss.item()
                validation_predictions = validation_logits.argmax(dim=1)
                validation_correct += (validation_predictions == validation_labels).sum().item()
                validation_total += validation_labels.size(0)

        average_validation_loss = validation_loss_sum / len(val_loader)
        validation_accuracy = 100 * validation_correct / validation_total if validation_total > 0 else 0.0

        history["val_losses"].append(average_validation_loss)
        history["val_accuracies"].append(validation_accuracy)

        print(f"Validation: loss={average_validation_loss:.4f}, accuracy={validation_accuracy:.2f}%")

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            epochs_without_improvement = 0
            checkpoint_dir = os.path.dirname(best_model_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement. Trigger times: {epochs_without_improvement}")
            if epochs_without_improvement >= patience:
                print("Early stopping!")
                break

        scheduler.step()

    return history


@torch.no_grad()
def evaluate_cnn_resbigru(
    model: Model_1D_CNN_ResBiGRU,
    data_loader,
    device: torch.device,
    history: Dict[str, object] = None,
    best_model_path: str = "",
) -> Dict[str, object]:
    """Evaluate CNN-ResBiGRU on a loader and return standard metrics."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    model.eval()
    y_true = []
    y_pred = []
    confidences = []

    with torch.no_grad():
        for batch_inputs, batch_labels in data_loader:
            batch_inputs = batch_inputs.to(device)
            logits = model(batch_inputs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            confs = probs.max(dim=1).values.cpu().numpy()

            y_pred.extend(preds.tolist())
            y_true.extend(batch_labels.numpy().tolist())
            confidences.extend(confs.tolist())

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

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
        "confidence": confidences,
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "best_model_path": best_model_path,
    }