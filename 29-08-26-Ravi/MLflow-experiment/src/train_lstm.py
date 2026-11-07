import os
import copy
import time
import tempfile

import numpy as np

import torch
import torch.nn as nn

import mlflow
import mlflow.pytorch

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)


# ============================================================
# MODEL SIZE
# ============================================================

def get_model_size_mb(model):

    with tempfile.NamedTemporaryFile(
        suffix=".pt",
        delete=False
    ) as f:

        path = f.name

    torch.save(
        model.state_dict(),
        path
    )

    size_mb = (
        os.path.getsize(path)
        / (1024 ** 2)
    )

    os.remove(path)

    return size_mb


# ============================================================
# EVALUATION
# ============================================================

def evaluate_lstm(
    model,
    loader,
    device
):

    model.eval()

    all_probabilities = []

    all_targets = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)

            logits = model(X)

            probability = (
                torch.sigmoid(
                    logits
                )
            )

            all_probabilities.append(
                probability
                .cpu()
                .numpy()
            )

            all_targets.append(
                y.numpy()
            )

    y_prob = np.concatenate(
        all_probabilities
    )

    y_true = np.concatenate(
        all_targets
    )

    pr_auc = (
        average_precision_score(
            y_true,
            y_prob
        )
    )

    roc_auc = (
        roc_auc_score(
            y_true,
            y_prob
        )
    )

    return (
        pr_auc,
        roc_auc
    )


# ============================================================
# TRAINING
# ============================================================

def train_lstm(

    model,

    train_loader,

    val_loader,

    device,

    epochs=10,

    learning_rate=1e-3

):

    print("\n")
    print("=" * 60)
    print("LSTM TRAINING")
    print("=" * 60)

    model.to(device)

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_pr_auc = -np.inf

    best_state = None

    with mlflow.start_run(
        run_name="PyTorch_LSTM"
    ):

        mlflow.log_params({

            "model":
                "PyTorch_LSTM",

            "epochs":
                epochs,

            "learning_rate":
                learning_rate,

            "input_size":
                model.lstm.input_size,

            "hidden_size":
                model.lstm.hidden_size,

            "num_layers":
                model.lstm.num_layers,

            "device":
                str(device)
        })

        # ====================================================
        # Epochs
        # ====================================================

        for epoch in range(epochs):

            model.train()

            total_loss = 0

            start_time = time.time()

            # ------------------------------------------------
            # Batches
            # ------------------------------------------------

            for X, y in train_loader:

                X = X.to(device)

                y = y.to(device)

                optimizer.zero_grad()

                logits = model(X)

                loss = criterion(
                    logits,
                    y
                )

                loss.backward()

                optimizer.step()

                total_loss += (
                    loss.item()
                )

            train_loss = (
                total_loss
                / len(train_loader)
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            val_pr_auc, val_roc_auc = (
                evaluate_lstm(
                    model,
                    val_loader,
                    device
                )
            )

            epoch_time = (
                time.time()
                - start_time
            )

            print(
                f"Epoch "
                f"{epoch + 1}/{epochs} | "
                f"Loss={train_loss:.5f} | "
                f"PR-AUC={val_pr_auc:.5f} | "
                f"Time={epoch_time:.2f}s"
            )

            # ------------------------------------------------
            # MLflow
            # ------------------------------------------------

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch
            )

            mlflow.log_metric(
                "val_pr_auc",
                val_pr_auc,
                step=epoch
            )

            mlflow.log_metric(
                "val_roc_auc",
                val_roc_auc,
                step=epoch
            )

            # ------------------------------------------------
            # Best model
            # ------------------------------------------------

            if val_pr_auc > best_pr_auc:

                best_pr_auc = (
                    val_pr_auc
                )

                best_state = (
                    copy.deepcopy(
                        model.state_dict()
                    )
                )

        # ====================================================
        # Restore best
        # ====================================================

        model.load_state_dict(
            best_state
        )

        # ====================================================
        # Save Float32
        # ====================================================

        os.makedirs(
            "models",
            exist_ok=True
        )

        float32_path = (
            "models/pytorch_lstm_best.pt"
        )

        torch.save(
            model.state_dict(),
            float32_path
        )

        float32_size = (
            get_model_size_mb(
                model
            )
        )

        mlflow.log_metric(
            "best_val_pr_auc",
            best_pr_auc
        )

        mlflow.log_metric(
            "float32_model_size_mb",
            float32_size
        )

        mlflow.log_artifact(
            float32_path,
            artifact_path="float32_model"
        )

        # ====================================================
        # Log PyTorch model
        # ====================================================
        '''
        mlflow.pytorch.log_model(
            model,
            artifact_path="pytorch_lstm"
        )
        '''
        mlflow.pytorch.log_model(
        model,
        artifact_path="pytorch_lstm",
        serialization_format="pickle"

        )
        '''
        # ====================================================
        # INT8
        # ====================================================
        
        print(
            "\nApplying INT8 quantization..."
        )

        quantized_model = (
            torch.quantization.quantize_dynamic(

                model.cpu(),

                {
                    nn.LSTM,
                    nn.Linear
                },

                dtype=torch.qint8
            )
        )

        int8_path = (
            "models/pytorch_lstm_int8.pt"
        )

        torch.save(
            quantized_model.state_dict(),
            int8_path
        )

        int8_size = (
            os.path.getsize(
                int8_path
            )
            / (1024 ** 2)
        )

        compression_ratio = (
            float32_size
            / int8_size
        )

        size_reduction = (
            1
            - int8_size
            / float32_size
        ) * 100

        print(
            f"Float32: "
            f"{float32_size:.2f} MB"
        )

        print(
            f"INT8: "
            f"{int8_size:.2f} MB"
        )

        print(
            f"Compression: "
            f"{compression_ratio:.2f}x"
        )

        print(
            f"Reduction: "
            f"{size_reduction:.2f}%"
        )

        mlflow.log_metrics({

            "int8_model_size_mb":
                int8_size,

            "compression_ratio":
                compression_ratio,

            "size_reduction_percent":
                size_reduction
        })

        mlflow.log_artifact(
            int8_path,
            artifact_path="int8_model"
        )
        '''
        # ====================================================
        # INT8 QUANTIZATION
        # ====================================================

        print(
            "\nApplying INT8 quantization..."
        )

        quantized_model = (
            torch.quantization.quantize_dynamic(

                model.cpu(),

                {
                    nn.LSTM,
                    nn.Linear
                },

                dtype=torch.qint8
            )
        )

        os.makedirs(
            "models",
            exist_ok=True
        )

        int8_path = (
            "models/pytorch_lstm_int8.pt"
        )

        torch.save(
            quantized_model.state_dict(),
            int8_path
        )

        int8_size = (
            os.path.getsize(
                int8_path
            )
            / (1024 ** 2)
        )

        compression_ratio = (
            float32_size
            / int8_size
        )

        size_reduction = (
            1
            - int8_size
            / float32_size
        ) * 100

        print(
            f"Float32: "
            f"{float32_size:.2f} MB"
        )

        print(
            f"INT8: "
            f"{int8_size:.2f} MB"
        )

        print(
            f"Compression: "
            f"{compression_ratio:.2f}x"
        )

        print(
            f"Reduction: "
            f"{size_reduction:.2f}%"
        )

        mlflow.log_metrics({

            "int8_model_size_mb":
                int8_size,

            "compression_ratio":
                compression_ratio,

            "size_reduction_percent":
                size_reduction
        })

        mlflow.log_artifact(
            int8_path,
            artifact_path="int8_model"
        )
        # ====================================================
        # Registry
        # ====================================================
        '''
        mlflow.pytorch.log_model(

            model,

            artifact_path="best_model",

            registered_model_name=
                "TBM_PyTorch_LSTM"
        )
        '''
        ## Avaoide Crashing 
        mlflow.pytorch.log_model(
        model,
        artifact_path="best_model",
        registered_model_name="TBM_PyTorch_LSTM",
        serialization_format="pickle"
    )

        print(
            "\n✓ LSTM model trained"
        )

    return (
        model,
        quantized_model,
        best_pr_auc
    )