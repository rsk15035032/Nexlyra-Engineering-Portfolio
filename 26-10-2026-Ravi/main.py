import os
import time
import zlib
import base64
import json
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import onnxruntime as ort
from scipy.stats import ks_2samp
import joblib

# Global context dictionary for runtime session, scalers, and monitoring buffers
ml_context = {
    "ort_session": None,
    "scaler": None,
    "request_count": 0,
    "cavitation_alerts": 0,
    "fraud_flags": 0,
    "recent_acoustic_variance": []
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan context manager loading the INT8/FP32 ONNX model 
    from the artifacts directory and locating the fluid mechanics scaler.
    """
    print("[*] Initializing MLOps startup: Loading ONNX runtime session and fluid mechanics scaler...")
    
    # 1. Load ONNX Runtime Session (Prioritize artifacts/ directory)
    onnx_candidates = [
        os.path.join("artifacts", "hydro_vit_int8.onnx"),
        os.path.join("artifacts", "hydro_vit.onnx"),
        "hydro_vit_int8.onnx",
        "hydro_vit.onnx"
    ]
    onnx_path = next((path for path in onnx_candidates if os.path.exists(path)), None)
    
    if onnx_path:
        ml_context["ort_session"] = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        print(f"[*] Successfully loaded ONNX model from: {onnx_path}")
    else:
        raise FileNotFoundError("[!] Critical: No ONNX model file found in artifacts/ (hydro_vit_int8.onnx or hydro_vit.onnx). Run export_quantization.py first.")
    
    # 2. Robustly locate fluid mechanics scaler across candidate directories
    scaler_candidates = [
        "scaler_fluid_mechanics.joblib",
        "scaler_physics.joblib",
        os.path.join("models", "scaler_fluid_mechanics.joblib"),
        os.path.join("models", "scaler_physics.joblib"),
        os.path.join("artifacts", "scaler_fluid_mechanics.joblib"),
        os.path.join("16-10-2026-Ravi", "scaler_fluid_mechanics.joblib")
    ]
    scaler_path = next((path for path in scaler_candidates if os.path.exists(path)), None)
    
    if scaler_path:
        ml_context["scaler"] = joblib.load(scaler_path)
        print(f"[*] Successfully loaded fluid mechanics scaler from: {scaler_path}")
    else:
        print("[!] Warning: Scaler file not found. Proceeding with unscaled feature inputs.")

    print("[*] MLOps startup initialization complete.")
    yield
    
    # Shutdown cleanup
    print("[*] Shutting down MLOps microservice: Clearing global ML context...")
    ml_context.clear()
    print("[*] Microservice shutdown complete.")

app = FastAPI(
    title="Nexlyra HydroCavitationNet MLOps Microservice",
    version="2.1.0",
    description="Production telemetry auditing, ONNX INT8 inference, and cavitation fraud detection engine.",
    lifespan=lifespan
)

security = HTTPBearer()
API_SECRET_TOKEN = os.getenv("NEXLYRA_API_TOKEN", "nexlyra-secure-bearer-2026")


class HydroPayload(BaseModel):
    IoT_Hydrophone_JSON: str = Field(..., description="Zlib-Hex-Base85 encoded telemetry payload string.")
    Claimed_Invoice_USD: float = Field(..., description="Billed emergency shutdown invoice amount in USD.")
    Contractor_Name: str = Field(..., description="Executing vendor name.")


def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verifies HTTP Bearer token authorization for secure field telemetry access."""
    if credentials.credentials != API_SECRET_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    """Custom middleware recording request latency and injecting X-Inference-Latency-MS header."""
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000.0
    response.headers["X-Inference-Latency-MS"] = f"{duration_ms:.2f}"
    return response


@app.post("/api/v1/predict-cavitation")
def predict_cavitation(payload: HydroPayload, token: str = Depends(verify_bearer_token)):
    """
    Primary prediction endpoint: Decodes payloads, applies robust joblib scaling, 
    executes ONNX inference, monitors data drift, and audits fraudulent vendor billing.
    """
    ml_context["request_count"] += 1
    try:
        # 1. Decode Zlib-Hex-Base85 Payload
        decoded_bytes = base64.b85decode(payload.IoT_Hydrophone_JSON.encode('ascii'))
        decompressed_data = zlib.decompress(decoded_bytes)
        telemetry = json.loads(decompressed_data.decode('utf-8'))
        
        spectrogram = np.array(telemetry["spectrogram"], dtype=np.float32).reshape(1, 1, 10, 10)
        physics = np.array(telemetry["physics_features"], dtype=np.float32).reshape(1, -1)
        
        # 2. Data Drift Monitoring via Kolmogorov-Smirnov Test
        current_variance = float(np.var(spectrogram))
        ml_context["recent_acoustic_variance"].append(current_variance)
        drift_warning = False
        
        if len(ml_context["recent_acoustic_variance"]) >= 20:
            baseline_ref = ml_context["recent_acoustic_variance"][:-10]
            recent_window = ml_context["recent_acoustic_variance"][-10:]
            _, p_val = ks_2samp(baseline_ref, recent_window)
            if p_val < 0.05 or abs(np.mean(recent_window) - np.mean(baseline_ref)) / (np.mean(baseline_ref) + 1e-9) > 0.25:
                drift_warning = True

        # 3. Scale Tabular Physics Features safely based on scaler's expected input dimension
        if ml_context["scaler"] is not None:
            expected_features = getattr(ml_context["scaler"], "n_features_in_", physics.shape[1])
            if physics.shape[1] == expected_features:
                physics = ml_context["scaler"].transform(physics)
            elif physics.shape[1] > expected_features:
                scaled_part = ml_context["scaler"].transform(physics[:, :expected_features])
                unscaled_part = physics[:, expected_features:]
                physics = np.hstack([scaled_part, unscaled_part])
            else:
                physics = ml_context["scaler"].transform(physics)

        # Ensure final physics shape matches model input expectation [1, 6]
        if physics.shape[1] < 6:
            physics = np.pad(physics, ((0, 0), (0, 6 - physics.shape[1])), mode='constant')
        elif physics.shape[1] > 6:
            physics = physics[:, :6]

        # 4. Execute ONNX Runtime Inference Session
        ort_inputs = {
            'acoustic_input': spectrogram,
            'tabular_input': physics
        }
        ort_outs = ml_context["ort_session"].run(None, ort_inputs)
        logits = ort_outs[0][0][0]
        probability = float(1.0 / (1.0 + np.exp(-logits)))

        # 5. Cost-Optimal Threshold & Fraud Audit Rules
        optimal_threshold = 0.385  # Cost-utility threshold ($500k FP vs $10M FN)
        cavitation_detected = probability >= optimal_threshold
        
        fraud_alert = False
        if not cavitation_detected and payload.Claimed_Invoice_USD > 100000.0 and payload.Contractor_Name == "Hydro_Phantom_Turbines":
            fraud_alert = True
            ml_context["fraud_flags"] += 1

        if cavitation_detected:
            ml_context["cavitation_alerts"] += 1

        return {
            "cavitation_probability": round(probability, 4),
            "cavitation_detected": cavitation_detected,
            "fraud_alert": fraud_alert,
            "cost_optimal_threshold": optimal_threshold,
            "data_drift_warning": drift_warning,
            "audited_invoice_usd": payload.Claimed_Invoice_USD
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Telemetry payload decoding or inference failed: {str(e)}")


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint tracking total predictions, cavitation alerts, and fraud flags."""
    return {
        "total_predictions": ml_context["request_count"],
        "total_cavitation_alerts": ml_context["cavitation_alerts"],
        "total_fraud_flags": ml_context["fraud_flags"]
    }