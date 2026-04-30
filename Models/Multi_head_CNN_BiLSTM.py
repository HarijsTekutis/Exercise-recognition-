import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline import build_training_objects



class BiLSTMBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm(x)
        return out


class MULTI_HEAD_CNN_LSTM(nn.Module):
    """
    
    Expected input feature order:
    [A_x, A_y, A_z, G_x, G_y, G_z, body_a_x, body_a_y, body_a_z]
    
    Input shape: (batch, time, 9 features)
    Output shape: (batch, num_classes)
    """
    
    def __init__(self, num_features: int = 9, num_classes: int = 6, hidden_dim: int = 64, lstm_layers: int = 1):
        super().__init__()

        # Three heads:
        # 1) Accelerometer: A_x, A_y, A_z
        # 2) Gyroscope: G_x, G_y, G_z
        # 3) Body acceleration: body_a_x, body_a_y, body_a_z
        head_out_channels = 128

        def create_cnn_head():
            return nn.Sequential(
                nn.Conv1d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(in_channels=128, out_channels=head_out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(head_out_channels),
                nn.ReLU(),
            )

        self.acc_head = create_cnn_head()
        self.gyr_head = create_cnn_head()
        self.body_head = create_cnn_head()

        lstm_input_dim = head_out_channels * 3
        num_bilstm_blocks = max(1, lstm_layers)
        self.bilstm_blocks = nn.ModuleList()
        for block_index in range(num_bilstm_blocks):
            block_input = lstm_input_dim if block_index == 0 else hidden_dim * 2
            self.bilstm_blocks.append(BiLSTMBlock(input_size=block_input, hidden_size=hidden_dim))

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, time, 9 features)
               Feature order: [A_x, A_y, A_z, G_x, G_y, G_z, body_a_x, body_a_y, body_a_z]
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Conv1d uses channel-first tensors.
        x = x.permute(0, 2, 1)

        acc_x = x[:, 0:3, :]
        gyr_x = x[:, 3:6, :]
        body_x = x[:, 6:9, :]


        acc_feat = self.acc_head(acc_x)
        gyr_feat = self.gyr_head(gyr_x)
        body_feat = self.body_head(body_x)

        fused = torch.cat([acc_feat, gyr_feat, body_feat], dim=1)

        # Recurrent blocks expect (batch, time, channels).
        fused = fused.permute(0, 2, 1)
        for block in self.bilstm_blocks:
            fused = block(fused)
        lstm_out = fused

        mean_pool = lstm_out.mean(dim=1)
        max_pool, _ = lstm_out.max(dim=1)
        out = torch.cat([mean_pool, max_pool], dim=1)

        out = self.dropout(out)
        out = self.fc(out)
        return out


def train_multi_head_cnn_lstm(
    model: MULTI_HEAD_CNN_LSTM,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    device: torch.device,
    learning_rate: float = 0.001,
    num_epochs: int = 50,
    clip_grad_norm: float = 1.0,
    patience: int = 7,
    best_model_path: str = "multi_head_CNN_LSTM.pt",
) -> Dict[str, List[float]]:
    """
    Train the MULTI_HEAD_CNN_LSTM model with validation, LR scheduling, and early stopping.
    
    Args:
        model: MULTI_HEAD_CNN_LSTM model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        class_weights: Class weights tensor for loss function
        device: Device to train on
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        clip_grad_norm: Gradient clipping norm value
        patience: Early stopping patience
        best_model_path: Path to save best model
    
    Returns:
        Dictionary containing training history with keys:
        - train_losses, train_accuracies
        - val_losses, val_accuracies
    """
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
        # Training phase
        model.train()
        correct_predictions = 0
        total_examples = 0
        running_loss = 0.0

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

        # Validation phase
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

        # Early stopping
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
def evaluate_multi_head_cnn_lstm(
    model: MULTI_HEAD_CNN_LSTM,
    data_loader,
    device: torch.device,
    history: Dict[str, object] = None,
    best_model_path: str = "",
) -> Dict[str, object]:
    """Evaluate MULTI_HEAD_CNN_LSTM on a test loader and return metrics."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    if best_model_path and os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
    elif best_model_path:
        print(f"Warning: best checkpoint not found at {best_model_path}; evaluating current model weights.")
    
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
        "confidence": confidences,
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "best_model_path": best_model_path,
    }
    return result

