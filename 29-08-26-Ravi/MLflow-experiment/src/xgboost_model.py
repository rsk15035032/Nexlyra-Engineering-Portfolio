import os

import mlflow
import mlflow.xgboost

import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


def calculate_metrics(
    y_true,
    y_prob
):

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    return {

        "pr_auc":
            average_precision_score(
                y_true,
                y_prob
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_prob
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
    }


def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    params
):

    print("\n")
    print("=" * 60)
    print("XGBOOST TRAINING")
    print("=" * 60)

    with mlflow.start_run(
        run_name="XGBoost_TBM"
    ):

        mlflow.log_params(
            params
        )

        model = xgb.XGBClassifier(
            **params
        )

        model.fit(

            X_train,

            y_train,

            eval_set=[
                (
                    X_train,
                    y_train
                ),
                (
                    X_val,
                    y_val
                )
            ],

            verbose=True
        )

        # ====================================================
        # Predictions
        # ====================================================

        train_prob = (
            model
            .predict_proba(
                X_train
            )[:, 1]
        )

        val_prob = (
            model
            .predict_proba(
                X_val
            )[:, 1]
        )

        # ====================================================
        # Metrics
        # ====================================================

        train_metrics = calculate_metrics(
            y_train,
            train_prob
        )

        val_metrics = calculate_metrics(
            y_val,
            val_prob
        )

        print("\nValidation Metrics")

        for key, value in val_metrics.items():

            print(
                f"{key}: {value:.5f}"
            )

        # ====================================================
        # MLflow metrics
        # ====================================================

        mlflow.log_metrics({

            "train_pr_auc":
                train_metrics["pr_auc"],

            "train_roc_auc":
                train_metrics["roc_auc"],

            "train_precision":
                train_metrics["precision"],

            "train_recall":
                train_metrics["recall"],

            "train_f1":
                train_metrics["f1"],

            "val_pr_auc":
                val_metrics["pr_auc"],

            "val_roc_auc":
                val_metrics["roc_auc"],

            "val_precision":
                val_metrics["precision"],

            "val_recall":
                val_metrics["recall"],

            "val_f1":
                val_metrics["f1"]
        })

        # ====================================================
        # Save
        # ====================================================

        os.makedirs(
            "models",
            exist_ok=True
        )

        model_path = (
            "models/xgboost_best.json"
        )

        model.save_model(
            model_path
        )

        mlflow.log_artifact(
            model_path,
            artifact_path="xgboost_model"
        )

        # ====================================================
        # MLflow Model Registry
        # ====================================================

        mlflow.xgboost.log_model(

            model,

            artifact_path="best_model",

            registered_model_name=
                "TBM_XGBoost"
        )

        print(
            "\n✓ XGBoost model trained"
        )

    return model, val_metrics