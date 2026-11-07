import pandas as pd

from .config import (
    X_TRAIN_PATH,
    X_VAL_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_VAL_PATH,
    Y_TEST_PATH,
)


def load_data():

    print("Loading datasets...")

    X_train = pd.read_parquet(
        X_TRAIN_PATH
    )

    X_val = pd.read_parquet(
        X_VAL_PATH
    )

    X_test = pd.read_parquet(
        X_TEST_PATH
    )

    y_train = pd.read_parquet(
        Y_TRAIN_PATH
    )

    y_val = pd.read_parquet(
        Y_VAL_PATH
    )

    y_test = pd.read_parquet(
        Y_TEST_PATH
    )

    # Convert single-column DataFrames to Series
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    if isinstance(y_val, pd.DataFrame):
        y_val = y_val.iloc[:, 0]

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    print("\nDataset shapes")

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_val:",
        X_val.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "y_val:",
        y_val.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )