import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from data_pipeline import (
    clean_imu_columns,
    encode_activities,
    load_mmfit_wrist_recordings,
    make_train_test_loaders,
)
from model_architecture import CNNLSTM
from train_eval import (
    build_classification_report,
    build_training_objects,
    compute_class_weights,
    evaluate_model,
    get_device,
    train_model,
)


def main() -> None:
    dataset_root = "mmfit_npy_data"

    feature_columns = [
        "sw_r_acc_x", "sw_r_acc_y", "sw_r_acc_z",
        "sw_r_gyr_x", "sw_r_gyr_y", "sw_r_gyr_z",
        "sw_l_acc_x", "sw_l_acc_y", "sw_l_acc_z",
        "sw_l_gyr_x", "sw_l_gyr_y", "sw_l_gyr_z",
    ]

    data = load_mmfit_wrist_recordings(
        dataset_root=dataset_root,
        sample_rate_hz=100.0,
        labels_time_unit="milliseconds",
        min_recordings_per_activity=0,
    )

    activity_to_id = encode_activities(data)
    clean_imu_columns(data, feature_columns)

    train_loader, test_loader, train_dataset, _ = make_train_test_loaders(
        data=data,
        imu_features=feature_columns,
        window_size=300,
        step_size=100,
        train_split=0.8,
        batch_size_train=32,
        batch_size_test=1,
    )

    device = get_device()
    model = CNNLSTM(
        num_features=len(feature_columns),
        num_classes=len(activity_to_id),
        hidden_dim=64,
        lstm_layers=2,
    ).to(device)

    class_weights = compute_class_weights(
        train_dataset=train_dataset,
        num_classes=len(activity_to_id),
        device=device,
    )

    criterion, optimizer, scheduler = build_training_objects(
        model=model,
        class_weights=class_weights,
        learning_rate=2e-4,
        num_epochs=50,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=50,
        clip_grad_norm=1.0,
        patience=7,
        best_model_path="best_model_CNN+LSTM_npy.pt",
    )

    y_true, y_pred, cm = evaluate_model(model, test_loader, device)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues", xticks_rotation=90)
    plt.show()

    id_to_activity = {idx: activity for activity, idx in activity_to_id.items()}
    print(build_classification_report(y_true, y_pred, id_to_activity))


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
