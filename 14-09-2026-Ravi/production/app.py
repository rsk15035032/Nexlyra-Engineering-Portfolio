"""
Nexlyra StrainVisionNet
Production FastAPI Inference Microservice

Architecture:
    Client Payload (Base64 Matrix + Raw Tabular)
         │
         ├──► Decode Base64 ──► 8x8 Spatial Tensor ──► Drift Engine
         │                                 │
         │                                 ▼
         ├──► Calculate Engineered Spatial Features
         │                                 │
         ▼                                 ▼
    Tabular Features                Spatial Features
         │                                 │
         ▼                                 ▼
   Tabular Scaler                   Spatial Scaler
         │                                 │
         └───────────────┬─────────────────┘
                         ▼
             Concat [1, 5] Feature Vector
                         │
                         ▼
                ONNX Dual Inference Engine
                         │
                         ▼
              Sigmoid Probability & Output
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Dict, Final, Tuple

import joblib
import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings

from drift import SpatialDriftDetector

# CONFIGURATION & SETTINGS


class Settings(BaseSettings):
    """Application level environment settings and constants."""
    
    app_name: str = "Nexlyra Strain Vision Microfracture API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Security
    api_key_secret: str = Field(default="nexlyra-prod-secret-key-2026", validation_alias="API_KEY_SECRET")
    
    # Directory Layout
    root_dir: Path = Path(__file__).resolve().parent
    models_dir: Path = root_dir / "models"
    artifacts_dir: Path = root_dir / "artifacts"
    
    # Model Artifact File Paths
    onnx_model_path: Path = models_dir / "strain_vision_net.onnx"
    spatial_scaler_path: Path = artifacts_dir / "spatial_feature_standard_scaler.pkl"
    tabular_scaler_path: Path = artifacts_dir / "tabular_standard_scaler.pkl"
    baseline_entropy_path: Path = artifacts_dir / "strain_drift_baseline.json"

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings: Final[Settings] = Settings()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("strain_vision_api")



# CONSTANTS & CONTRACTS

MATRIX_SIZE: Final[int] = 8
EXPECTED_VALUES: Final[int] = 64

TABULAR_COLUMNS: Final[list[str]] = ["Axial_Load_Tons", "Wind_Shear_MPH"]
SPATIAL_COLUMNS: Final[list[str]] = ["Symmetry_V", "Symmetry_H", "Center_to_Edge_Ratio"]
FEATURE_COLUMNS: Final[list[str]] = TABULAR_COLUMNS + SPATIAL_COLUMNS



# SCHEMAS


class StrainInferenceRequest(BaseModel):
    """Client request schema. Engineered spatial features are calculated server-side."""
    
    model_config = ConfigDict(extra="forbid")

    matrix_b64: str = Field(
        ...,
        min_length=1,
        description="Base64 encoded JSON array containing exactly 64 float values.",
        examples=["WzAuMSwgMC4yLCAuLi4sIDAuOF0="],
    )
    Axial_Load_Tons: float = Field(
        ...,
        description="Raw axial load input in tons.",
        examples=[12.5],
    )
    Wind_Shear_MPH: float = Field(
        ...,
        description="Raw wind shear input speed in MPH.",
        examples=[45.2],
    )


class StrainInferenceResponse(BaseModel):
    """Client response contract."""
    
    MicroFracture_Triggered: bool = Field(..., description="Binary classification flag.")
    MicroFracture_Probability: float = Field(..., description="Sigmoid normalized output [0.0 - 1.0].")
    Spatial_Entropy: float = Field(..., description="Calculated Shannon entropy of strain tensor.")
    Drift_Detected: bool = Field(..., description="Flag indicating statistical drift from baseline.")


# FEATURE ENGINEERING & PREPROCESSING PIPELINE


class PipelinePreprocessor:
    """Stateless unit for decoding tensors and computing engineered domain features."""

    @staticmethod
    def decode_base64_matrix(b64_str: str) -> np.ndarray:
        """Decode Base64 string -> UTF-8 JSON -> 8x8 Spatial Matrix (1, 1, 8, 8)."""
        if not isinstance(b64_str, str) or not b64_str.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="matrix_b64 must be a non-empty string.",
            )

        try:
            raw_bytes = base64.b64decode(b64_str.strip(), validate=True)
            decoded_text = raw_bytes.decode("utf-8")
            values = json.loads(decoded_text)
        except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="matrix_b64 payload must be valid Base64-encoded UTF-8 JSON array.",
            ) from exc

        if not isinstance(values, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Decoded payload must be a list of numbers.",
            )

        try:
            array = np.asarray(values, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Decoded array contains non-numeric values.",
            ) from exc

        if array.size != EXPECTED_VALUES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected {EXPECTED_VALUES} values for 8x8 matrix, received {array.size}.",
            )

        if not np.isfinite(array).all():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Matrix payload contains infinite or NaN elements.",
            )

        return array.reshape(1, 1, MATRIX_SIZE, MATRIX_SIZE)

    @staticmethod
    def calculate_spatial_features(spatial_tensor: np.ndarray) -> np.ndarray:
        """Derives Vertical Symmetry, Horizontal Symmetry, and Center-to-Edge Ratio."""
        matrix = spatial_tensor[0, 0]

        # Symmetries
        symmetry_v = float(np.mean(np.abs(matrix - np.flip(matrix, axis=0))))
        symmetry_h = float(np.mean(np.abs(matrix - np.flip(matrix, axis=1))))

        # Center vs Edge Intensity Ratio
        center_region = matrix[3:5, 3:5]
        center_mean = float(np.mean(np.abs(center_region)))

        edge_values = np.concatenate([
            matrix[0, :],   # Top
            matrix[-1, :],  # Bottom
            matrix[:, 0],   # Left
            matrix[:, -1]   # Right
        ])
        edge_mean = float(np.mean(np.abs(edge_values)))
        
        center_to_edge_ratio = center_mean / (edge_mean + 1e-8)

        spatial_features = np.array(
            [[symmetry_v, symmetry_h, center_to_edge_ratio]], 
            dtype=np.float32
        )

        if not np.isfinite(spatial_features).all():
            raise ValueError("Spatial feature extraction calculated non-finite metrics.")

        return spatial_features



# STATE CONTAINER & LIFESPAN


class ModelContainer:
    """In-memory runtime container for operational artifacts."""
    
    onnx_session: ort.InferenceSession | None = None
    spatial_scaler: Any = None
    tabular_scaler: Any = None
    drift_detector: SpatialDriftDetector | None = None

    @classmethod
    def is_healthy(cls) -> bool:
        return all([
            cls.onnx_session is not None,
            cls.spatial_scaler is not None,
            cls.tabular_scaler is not None,
            cls.drift_detector is not None,
        ])


model_container = ModelContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifespan state management."""
    logger.info("Initializing StrainVisionNet inference service...")

    # File Existence Verification
    for path in [settings.onnx_model_path, settings.spatial_scaler_path, settings.tabular_scaler_path]:
        if not path.exists():
            logger.critical("Required artifact path missing: %s", path)
            raise FileNotFoundError(f"Missing required service artifact: {path}")

    # Load Models & Scalers
    model_container.onnx_session = ort.InferenceSession(
        str(settings.onnx_model_path), 
        providers=["CPUExecutionProvider"]
    )
    model_container.spatial_scaler = joblib.load(settings.spatial_scaler_path)
    model_container.tabular_scaler = joblib.load(settings.tabular_scaler_path)

    # Validate Scalers
    if getattr(model_container.spatial_scaler, "n_features_in_", None) != len(SPATIAL_COLUMNS):
        raise RuntimeError("Loaded spatial scaler feature dimension mismatch.")
    if getattr(model_container.tabular_scaler, "n_features_in_", None) != len(TABULAR_COLUMNS):
        raise RuntimeError("Loaded tabular scaler feature dimension mismatch.")

    # Drift Detection
    model_container.drift_detector = SpatialDriftDetector(baseline_path=settings.baseline_entropy_path)

    logger.info("✓ Application artifacts successfully loaded.")
    yield
    logger.info("Shutting down service and releasing resources...")
    model_container.onnx_session = None
    model_container.spatial_scaler = None
    model_container.tabular_scaler = None
    model_container.drift_detector = None


# FASTAPI INSTANCE


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production inference microservice for Nexlyra StrainVisionNet.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# AUTHENTICATION DEPENDENCY

async def verify_api_key(x_api_key: Annotated[str, Header(alias="X-API-Key")] = "") -> str:
    """Header authentication gatekeeper."""
    if x_api_key != settings.api_key_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header credential.",
        )
    return x_api_key


# ROUTE ENDPOINTS

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """System readiness check."""
    healthy = model_container.is_healthy()
    return {
        "status": "healthy" if healthy else "unhealthy",
        "model_loaded": healthy,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
    }


@app.get("/model-info", status_code=status.HTTP_200_OK)
async def model_info():
    """Returns ONNX engine graph specifications and runtime topology."""
    if not model_container.onnx_session:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model uninitialized.")

    return {
        "model": "StrainVisionNet",
        "format": "ONNX",
        "provider": "CPUExecutionProvider",
        "spatial_input_shape": [1, 1, MATRIX_SIZE, MATRIX_SIZE],
        "tabular_input_shape": [1, len(FEATURE_COLUMNS)],
        "feature_columns": FEATURE_COLUMNS,
        "tabular_columns": TABULAR_COLUMNS,
        "spatial_columns": SPATIAL_COLUMNS,
        "vendor_is_model_feature": False,
        "onnx_inputs": [
            {"name": node.name, "shape": node.shape, "type": node.type}
            for node in model_container.onnx_session.get_inputs()
        ],
    }


@app.post("/predict", response_model=StrainInferenceResponse, status_code=status.HTTP_200_OK)
async def predict_microfracture(
    payload: StrainInferenceRequest,
    _: Annotated[str, Depends(verify_api_key)],
):
    """Executes microfracture classification pipeline with strict input processing."""
    
    # 1. Input Validation
    if not math.isfinite(payload.Axial_Load_Tons) or not math.isfinite(payload.Wind_Shear_MPH):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Tabular inputs must be finite numeric scalars.",
        )

    # 2. Decode Payload & Engineer Spatial Features
    spatial_tensor = PipelinePreprocessor.decode_base64_matrix(payload.matrix_b64)
    raw_spatial_features = PipelinePreprocessor.calculate_spatial_features(spatial_tensor)

    # 3. Spatial Drift Evaluation
    try:
        drift_flag, current_entropy = model_container.drift_detector.check_drift(spatial_tensor)
    except Exception as exc:
        logger.exception("Drift processing failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error evaluating statistical drift.",
        ) from exc

    # 4. Feature Vector Preprocessing & Concatenation
    try:
        raw_tabular = np.array([[payload.Axial_Load_Tons, payload.Wind_Shear_MPH]], dtype=np.float32)
        scaled_tabular = model_container.tabular_scaler.transform(raw_tabular).astype(np.float32)
        scaled_spatial = model_container.spatial_scaler.transform(raw_spatial_features).astype(np.float32)

        final_features = np.concatenate([scaled_tabular, scaled_spatial], axis=1)
    except Exception as exc:
        logger.exception("Preprocessing transformation failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error during feature standardization.",
        ) from exc

    # 5. ONNX Dynamic Inputs Mapping
    try:
        input_nodes = model_container.onnx_session.get_inputs()
        ort_inputs: Dict[str, np.ndarray] = {}

        for node in input_nodes:
            name_lower = node.name.lower()
            if any(k in name_lower for k in ["spatial", "strain", "image"]):
                ort_inputs[node.name] = spatial_tensor
            elif any(k in name_lower for k in ["tabular", "feature", "numeric"]):
                ort_inputs[node.name] = final_features

        ort_outputs = model_container.onnx_session.run(None, ort_inputs)
        raw_logit = float(np.asarray(ort_outputs[0]).squeeze())
    except Exception as exc:
        logger.exception("ONNX model execution failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference runtime failed to process tensor vectors.",
        ) from exc

    if not math.isfinite(raw_logit):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model computed non-finite outputs.",
        )

    # 6. Post-Processing & Response Construct
    probability = float(1.0 / (1.0 + math.exp(-raw_logit))) if raw_logit >= 0 else float(math.exp(raw_logit) / (1.0 + math.exp(raw_logit)))

    return StrainInferenceResponse(
        MicroFracture_Triggered=bool(probability >= 0.5),
        MicroFracture_Probability=round(probability, 4),
        Spatial_Entropy=round(float(current_entropy), 4),
        Drift_Detected=bool(drift_flag),
    )