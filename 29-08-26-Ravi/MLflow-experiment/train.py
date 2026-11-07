import sys

from pathlib import Path

import numpy as np

import torch

from torch.utils.data import DataLoader

import mlflow
import mlflow

from src.config import (
    MLFLOW_DB,
    EXPERIMENT_NAME,

    SEQUENCE_LENGTH,

    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,

    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_LEARNING_RATE,

    XGB_PARAMS
)

from src.data_loader import load_data

from src.validation import validate_dataset

from src.lstm_model import TBMLSTM

from src.xgboost_model import train_xgboost

from src.train_lstm import train_lstm

from src.sequence_dataset import TBMSequenceDataset

# ============================================================
# MLflow
# ============================================================

mlflow.set_tracking_uri(
    MLFLOW_DB
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("TBM FRAUD / SPOOF DETECTION")
    print("=" * 70)

    # ========================================================
    # 1. Load
    # ========================================================

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = load_data()

    # ========================================================
    # 2. Validation
    # ========================================================

    validate_dataset(

        X_train,
        X_val,
        X_test,

        y_train,
        y_val,
        y_test
    )

   

    # ========================================================
    # 5. Convert targets
    # ========================================================

    y_train_np = np.asarray(
        y_train,
        dtype=np.float32
    )

    y_val_np = np.asarray(
        y_val,
        dtype=np.float32
    )

    y_test_np = np.asarray(
        y_test,
        dtype=np.float32
    )

    # ========================================================
    # 6. XGBoost
    # ========================================================

    xgb_model, xgb_metrics = (
        train_xgboost(

            X_train,
            y_train_np,

            X_val,
            y_val_np,

            XGB_PARAMS
        )
    )

    # ========================================================
    # 7. LSTM
    # ========================================================

    print(
        "\nPreparing LSTM sequences..."
    )

    train_dataset = (
        TBMSequenceDataset(

            X_train.values,

            y_train_np,

            sequence_length=
                SEQUENCE_LENGTH
        )
    )

    val_dataset = (
        TBMSequenceDataset(

            X_val.values,

            y_val_np,

            sequence_length=
                SEQUENCE_LENGTH
        )
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=
            LSTM_BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=
            LSTM_BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )

    # ========================================================
    # 8. Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    # ========================================================
    # 9. LSTM
    # ========================================================

    input_size = (
        X_train.shape[1]
    )

    lstm_model = TBMLSTM(

        input_size=input_size,

        hidden_size=
            LSTM_HIDDEN_SIZE,

        num_layers=
            LSTM_NUM_LAYERS,

        dropout=
            LSTM_DROPOUT
    )

    (
        lstm_model,

        quantized_model,

        lstm_pr_auc
    ) = train_lstm(

        lstm_model,

        train_loader,

        val_loader,

        device,

        epochs=
            LSTM_EPOCHS,

        learning_rate=
            LSTM_LEARNING_RATE
    )

    # ========================================================
    # 10. Comparison
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        f"\nXGBoost Validation PR-AUC: "
        f"{xgb_metrics['pr_auc']:.5f}"
    )

    print(
        f"LSTM Validation PR-AUC: "
        f"{lstm_pr_auc:.5f}"
    )

    if (
        xgb_metrics["pr_auc"]
        >
        lstm_pr_auc
    ):

        print(
            "\n🏆 Best model: XGBoost"
        )

    else:

        print(
            "\n🏆 Best model: LSTM"
        )

    print(
        "\nTraining completed."
    )


if __name__ == "__main__":

    main()