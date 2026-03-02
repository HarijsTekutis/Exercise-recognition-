from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


def get_device() -> torch.device:
    """Pick CUDA when available, otherwise run on CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(train_dataset, num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute clipped inverse-frequency class weights from the training dataset."""
    # Collect integer class labels from all training windows.
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels, minlength=num_classes)

    # Avoid division by zero for classes not present in the sampled windows.
    safe_counts = np.maximum(class_counts, 1)
    class_weights_np = class_counts.sum() / (num_classes * safe_counts)

    # Keep weights in a stable range to avoid extreme loss scaling.
    class_weights_np = np.clip(class_weights_np, 0.5, 3.0).astype(np.float32)
    return torch.tensor(class_weights_np, dtype=torch.float32, device=device)


def build_training_objects(
    model: torch.nn.Module,
    class_weights: torch.Tensor,
    learning_rate: float,
    num_epochs: int,
):
    """Create criterion, optimizer, and scheduler used during training."""
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
    best_model_path: str = "best_model.pt",
) -> Dict[str, List[float]]:
    """Train the model with validation, LR scheduling, and early stopping."""
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    history = {
        "train_losses": [],
        "train_accuracies": [],
        "val_losses": [],
        "val_accuracies": [],
    }

    for epoch in range(num_epochs):
        # ---- Training phase ----
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

            # Skip unstable batches instead of breaking the full run.
            if torch.isnan(loss):
                print("NaN detected! Skipping batch.")
                continue

            loss.backward()
            # Clip gradients to reduce exploding-gradient issues.
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

        # ---- Validation phase ----
        model.eval()
        validation_loss_sum = 0.0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():
            for validation_inputs, validation_labels in test_loader:
                validation_inputs = validation_inputs.to(device)
                validation_labels = validation_labels.to(device)

                validation_logits = model(validation_inputs)
                validation_loss = criterion(validation_logits, validation_labels)

                validation_loss_sum += validation_loss.item()
                validation_predictions = validation_logits.argmax(dim=1)
                validation_correct += (validation_predictions == validation_labels).sum().item()
                validation_total += validation_labels.size(0)

        average_validation_loss = validation_loss_sum / len(test_loader)
        validation_accuracy = 100 * validation_correct / validation_total if validation_total > 0 else 0.0

        history["val_losses"].append(average_validation_loss)
        history["val_accuracies"].append(validation_accuracy)

        print(f"Validation: loss={average_validation_loss:.4f}, accuracy={validation_accuracy:.2f}%")

        # Save best checkpoint using validation loss.
        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss. Trigger times: {epochs_without_improvement}")
            if epochs_without_improvement >= patience:
                print("Early stopping!")
                break

        scheduler.step()

    return history


def evaluate_model(model: torch.nn.Module, test_loader, device: torch.device) -> Tuple[List[int], List[int], np.ndarray]:
    """Run inference on test data and return true labels, predictions, and confusion matrix."""
    model.eval()
    all_true_labels, all_predicted_labels = [], []

    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_inputs)
            predicted_classes = logits.argmax(dim=1)

            all_true_labels.extend(batch_labels.cpu().tolist())
            all_predicted_labels.extend(predicted_classes.cpu().tolist())

    confusion_mat = confusion_matrix(all_true_labels, all_predicted_labels)
    return all_true_labels, all_predicted_labels, confusion_mat


def build_classification_report(
    y_true: List[int], y_pred: List[int], id_to_activity: Dict[int, str]
) -> str:
    """Create a text classification report with readable class names."""
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