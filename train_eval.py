from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(train_dataset, num_classes: int, device: torch.device) -> torch.Tensor:
    y_train = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(y_train, minlength=num_classes)
    safe_counts = np.maximum(class_counts, 1)
    class_weights_np = class_counts.sum() / (num_classes * safe_counts)
    class_weights_np = np.clip(class_weights_np, 0.5, 3.0).astype(np.float32)
    return torch.tensor(class_weights_np, dtype=torch.float32, device=device)


def build_training_objects(
    model: torch.nn.Module,
    class_weights: torch.Tensor,
    learning_rate: float,
    num_epochs: int,
):
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    return criterion, optimizer, scheduler


def train_model(
    model: torch.nn.Module,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int = 50,
    clip_grad_norm: float = 1.0,
    patience: int = 7,
    best_model_path: str = "best_model_CNN+LSTM.pt",
) -> Dict[str, List[float]]:
    best_val_loss = float("inf")
    trigger_times = 0

    history = {
        "train_losses": [],
        "train_accuracies": [],
        "val_losses": [],
        "val_accuracies": [],
    }

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            outputs = model(X)
            loss = criterion(outputs, y)

            if torch.isnan(loss):
                print("NaN detected! Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        epoch_acc = 100 * correct / total if total > 0 else 0.0
        avg_loss = running_loss / len(train_loader)
        history["train_losses"].append(avg_loss)
        history["train_accuracies"].append(epoch_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}: loss={avg_loss:.4f}, accuracy={epoch_acc:.2f}%, lr={current_lr:.2e}"
        )

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs_val = model(X_val)
                loss_val = criterion(outputs_val, y_val)

                val_loss += loss_val.item()
                _, predicted_val = torch.max(outputs_val, dim=1)
                val_correct += (predicted_val == y_val).sum().item()
                val_total += y_val.size(0)

        val_loss /= len(test_loader)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0

        history["val_losses"].append(val_loss)
        history["val_accuracies"].append(val_acc)

        print(f"Validation: loss={val_loss:.4f}, accuracy={val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            trigger_times += 1
            print(f"No improvement in validation loss. Trigger times: {trigger_times}")
            if trigger_times >= patience:
                print("Early stopping!")
                break

        scheduler.step()

    return history


def evaluate_model(model: torch.nn.Module, test_loader, device: torch.device) -> Tuple[List[int], List[int], np.ndarray]:
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1)

            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    cm = confusion_matrix(y_true, y_pred)
    return y_true, y_pred, cm


def build_classification_report(
    y_true: List[int], y_pred: List[int], id_to_activity: Dict[int, str]
) -> str:
    labels_present = sorted(set(y_true) | set(y_pred))
    target_names = [
        id_to_activity[i] if i in id_to_activity else f"class_{i}" for i in labels_present
    ]
    return classification_report(
        y_true,
        y_pred,
        labels=labels_present,
        target_names=target_names,
        zero_division=0,
    )
