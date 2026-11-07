"""Production Configuration Settings for Nexlyra MicroFracture Service."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "strain_vision_net.onnx"

# Dual Scaler Artifact Paths
SPATIAL_SCALER_PATH = BASE_DIR / "artifacts" / "spatial_scaler.pkl"
TABULAR_SCALER_PATH = BASE_DIR / "artifacts" / "tabular_scaler.pkl"

DRIFT_BASELINE_PATH = BASE_DIR / "artifacts" / "strain_drift_baseline.json"
API_KEY_NAME = "X-API-Key"
API_KEY_VALUE = os.getenv("NEXLYRA_API_KEY", "nexlyra-prod-secret-key-2026")

CLASSIFICATION_THRESHOLD = 0.85

# Exact Dataset Columns & Feature Schema
TARGET_COLUMN = "MicroFracture_Triggered"
TABULAR_COLUMNS = [
    "Axial_Load_Tons",
    "Wind_Shear_MPH",
    "Symmetry_V",
    "Symmetry_H",
    "Center_to_Edge_Ratio",
]