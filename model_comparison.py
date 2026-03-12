"""Model comparison utility for training and evaluating different architectures."""

from typing import Dict, List, Tuple
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from model_architecture import CNNLSTM_1, CNNLSTM_2
from train_eval import (
    build_classification_report,
    build_training_objects,
    compute_class_weights,
    evaluate_model,
    get_device,
    train_model,
)


def create_model(
    architecture: str,
    num_features: int,
    num_classes: int,
    device: torch.device = None,
    **kwargs,
):
    """Instantiate a model by architecture name.

    Args:
        architecture: Model name ('CNNLSTM_1', 'CNNLSTM_2', or 'RandomForest').
        num_features: Number of input channels (IMU sensors).
        num_classes: Number of activity classes.
        device: PyTorch device (CPU or CUDA). Ignored for sklearn models.
        **kwargs: Additional hyperparameters for the model.

    Returns:
        Initialized model (PyTorch nn.Module or sklearn estimator).
    """
    if architecture == "CNNLSTM_1":
        model = CNNLSTM_1(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=kwargs.get("hidden_dim", 64),
            lstm_layers=kwargs.get("lstm_layers", 2),
        )
        return model.to(device)
    elif architecture == "CNNLSTM_2":
        model = CNNLSTM_2(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=kwargs.get("hidden_dim", 64),
            lstm_layers=kwargs.get("lstm_layers", 2),
        )
        return model.to(device)
    elif architecture == "RandomForest":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 20),
            min_samples_split=kwargs.get("min_samples_split", 5),
            random_state=42,
            n_jobs=-1, # To utilize all CPU cores for training
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


class ModelComparison:
    """Train and compare multiple model architectures."""

    def __init__(self, output_dir: str = "model_comparison_results"):
        """Initialize comparison tracker.

        Args:
            output_dir: Directory to save results and checkpoints.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: Dict[str, Dict] = {}

    def train_model_variant(
        self,
        architecture: str,
        train_loader,
        test_loader,
        train_dataset,
        num_features: int,
        num_classes: int,
        num_epochs: int = 50,
        learning_rate: float = 2e-4,
        clip_grad_norm: float = 1.0,
        patience: int = 7,
        device: torch.device = None,
        model_kwargs: Dict = None,
        is_sklearn: bool = False,
    ) -> Dict[str, any]:
        """Train a single model variant and track metrics.

        Args:
            architecture: Model architecture name.
            train_loader: Training DataLoader.
            test_loader: Testing DataLoader.
            train_dataset: Training dataset (for class weight computation).
            num_features: Number of input features.
            num_classes: Number of classes.
            num_epochs: Maximum number of training epochs.
            learning_rate: Learning rate for optimizer.
            clip_grad_norm: Gradient clipping threshold.
            patience: Early stopping patience.
            device: PyTorch device.
            model_kwargs: Additional architecture-specific kwargs.

        Returns:
            Dictionary containing model, metrics, and training history.
        """
        if device is None:
            device = get_device()

        model_kwargs = model_kwargs or {}

        print(f"\n{'='*60}")
        print(f"Training {architecture}")
        print(f"Config: {model_kwargs}")
        print(f"{'='*60}")

        # Create model.
        model = create_model(
            architecture=architecture,
            num_features=num_features,
            num_classes=num_classes,
            device=device if not is_sklearn else None,
            **model_kwargs,
        )

        if is_sklearn:
            # Handle sklearn models (RandomForest)
            # Convert dataloaders to numpy arrays
            X_train, y_train = [], []
            for batch_inputs, batch_labels in train_loader:
                X_train.append(batch_inputs.numpy())
                y_train.append(batch_labels.numpy())
            
            X_train = np.concatenate(X_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)
            
            # Flatten: (batch, time, features) -> (batch, time*features)
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            
            print(f"Training data shape: {X_train_flat.shape}")
            
            # Train the model
            model.fit(X_train_flat, y_train)
            
            # Evaluate on test set
            X_test, y_test = [], []
            for batch_inputs, batch_labels in test_loader:
                X_test.append(batch_inputs.numpy())
                y_test.append(batch_labels.numpy())
            
            X_test = np.concatenate(X_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            y_pred = model.predict(X_test_flat)
            y_true = y_test
            cm = np.zeros((num_classes, num_classes))
            for true_label, pred_label in zip(y_true, y_pred):
                cm[true_label, pred_label] += 1
            
            # Count parameters differently for sklearn
            param_count = sum(
                tree.tree_.node_count for tree in model.estimators_
            )
            
            # For sklearn models, we don't have training history
            history = {
                "train_losses": [],
                "train_accuracies": [],
                "val_losses": [0],
                "val_accuracies": [100 * (y_pred == y_true).sum() / len(y_true)],
            }
        else:
            # Handle PyTorch models
            # Print model size.
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {param_count:,}")

            # Prepare training.
            class_weights = compute_class_weights(
                train_dataset=train_dataset,
                num_classes=num_classes,
                device=device,
            )
            criterion, optimizer, scheduler = build_training_objects(
                model=model,
                class_weights=class_weights,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
            )

            # Train.
            best_model_path = self.output_dir / f"best_{architecture.lower()}.pt"
            history = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=num_epochs,
                clip_grad_norm=clip_grad_norm,
                patience=patience,
                best_model_path=str(best_model_path),
            )

            # Load best checkpoint.
            model.load_state_dict(torch.load(best_model_path, map_location=device))

            # Evaluate.
            y_true, y_pred, cm = evaluate_model(model, test_loader, device)

        # Compute metrics.
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }
        
        if is_sklearn:
            metrics["best_val_loss"] = 0.0
            metrics["best_val_acc"] = float(max(history["val_accuracies"]))
        else:
            metrics["best_val_loss"] = float(min(history["val_losses"]))
            metrics["best_val_acc"] = float(max(history["val_accuracies"]))
        
        best_model_path = self.output_dir / f"best_{architecture.lower()}.pt" if not is_sklearn else None
        
        result = {
            "model": model,
            "architecture": architecture,
            "config": model_kwargs,
            "param_count": param_count,
            "best_model_path": str(best_model_path) if best_model_path else None,
            "history": history,
            "y_true": y_true,
            "y_pred": y_pred,
            "confusion_matrix": cm.tolist() if isinstance(cm, np.ndarray) else cm,
            "metrics": metrics,
        }

        self.results[architecture] = result
        return result

    def compare_architectures(self) -> str:
        """Generate comparison summary of all trained models.

        Returns:
            Formatted comparison string.
        """
        if not self.results:
            return "No models trained yet."

        comparison_str = "\n" + "=" * 80 + "\n"
        comparison_str += "MODEL COMPARISON SUMMARY\n"
        comparison_str += "=" * 80 + "\n\n"

        # Sort by F1 score (descending).
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]["metrics"]["f1_score"],
            reverse=True,
        )

        for rank, (arch_name, result) in enumerate(sorted_models, 1):
            metrics = result["metrics"]
            comparison_str += f"{rank}. {arch_name}\n"
            comparison_str += f"   Parameters: {result['param_count']:,}\n"
            comparison_str += f"   Accuracy:  {metrics['accuracy']:.4f}\n"
            comparison_str += f"   Precision: {metrics['precision']:.4f}\n"
            comparison_str += f"   Recall:    {metrics['recall']:.4f}\n"
            comparison_str += f"   F1 Score:  {metrics['f1_score']:.4f}\n"
            comparison_str += f"   Best Val Acc: {metrics['best_val_acc']:.2f}%\n"
            comparison_str += f"   Best Val Loss: {metrics['best_val_loss']:.4f}\n"
            comparison_str += "\n"

        comparison_str += "=" * 80 + "\n"
        return comparison_str

    def save_results(self, filename: str = "comparison_results.json"):
        """Save results to JSON file.

        Args:
            filename: Output filename.
        """
        results_path = self.output_dir / filename

        # Prepare serializable results (exclude model objects).
        serializable_results = {}
        for arch_name, result in self.results.items():
            serializable_results[arch_name] = {
                "architecture": result["architecture"],
                "config": result["config"],
                "param_count": result["param_count"],
                "metrics": result["metrics"],
                "confusion_matrix": result["confusion_matrix"],
            }

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to {results_path}")

    def get_best_model(self) -> Tuple[str, torch.nn.Module]:
        """Get the best performing model by F1 score.

        Returns:
            Tuple of (architecture_name, model).
        """
        if not self.results:
            raise ValueError("No models trained yet.")

        best_arch = max(
            self.results.items(),
            key=lambda x: x[1]["metrics"]["f1_score"],
        )
        return best_arch[0], best_arch[1]["model"]
