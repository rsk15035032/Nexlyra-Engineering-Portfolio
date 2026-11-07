from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# DATA
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

X_TRAIN_PATH = DATA_DIR / "X_train.parquet"
X_VAL_PATH = DATA_DIR / "X_val.parquet"
X_TEST_PATH = DATA_DIR / "X_test.parquet"

Y_TRAIN_PATH = DATA_DIR / "y_train.parquet"
Y_VAL_PATH = DATA_DIR / "y_val.parquet"
Y_TEST_PATH = DATA_DIR / "y_test.parquet"


# ============================================================
# OUTPUT
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"


# ============================================================
# MLflow
# ============================================================

MLFLOW_DB = (
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)

EXPERIMENT_NAME = "TBM_Fraud_Detection"


# ============================================================
# TARGET
# ============================================================

TARGET = "Is_Spoofed"


# ============================================================
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42


# ============================================================
# LSTM
# ============================================================

SEQUENCE_LENGTH = 30

LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2

LSTM_BATCH_SIZE = 64

LSTM_EPOCHS = 10

LSTM_LEARNING_RATE = 1e-3


# ============================================================
# XGBOOST
# ============================================================

XGB_PARAMS = {

    "objective": "binary:logistic",

    "eval_metric": "aucpr",

    "max_depth": 8,

    "learning_rate": 0.05,

    "n_estimators": 500,

    "subsample": 0.8,

    "colsample_bytree": 0.8,

    "random_state": RANDOM_STATE,

    "n_jobs": -1
}