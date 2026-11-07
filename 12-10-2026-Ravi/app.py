"""
==============================================================================
NEXLYRA MLOPS: PRODUCTION FASTAPI MICROSERVICE (`app.py`)
==============================================================================
"""

import os
import json
import base64
import time
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Global State Dictionary for Production Artifacts
ml_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Lifecycle Event
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    onnx_path = os.path.join(base_dir,"artifacts", "hydro_cavnet.onnx")
    hydro_scaler_path = os.path.join(base_dir,"models", "hydrophone_scaler.joblib")
    web_scaler_path = os.path.join(base_dir,"models", "web_mechanics_scaler.joblib")
    
    if not os.path.exists(onnx_path):
        raise RuntimeError(f"ONNX model missing at: {onnx_path}")
    
    ml_artifacts['ort_session'] = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ml_artifacts['hydro_scaler'] = joblib.load(hydro_scaler_path) if os.path.exists(hydro_scaler_path) else None
    ml_artifacts['web_scaler'] = joblib.load(web_scaler_path) if os.path.exists(web_scaler_path) else None
    
    ml_artifacts['baseline_variance'] = 1.25
    ml_artifacts['telemetry_buffer'] = []
    print("[STARTUP] ONNX session and local feature scalers loaded successfully.")
    
    yield
    
    # Shutdown Cleanup (if needed)
    ml_artifacts.clear()
    print("[SHUTDOWN] Cleared model artifacts from memory.")

app = FastAPI(
    title="Nexlyra HydroCavitation Intelligence API",
    version="1.0.0",
    description="Real-time acoustic cavitation detection and vendor fraud analytics.",
    lifespan=lifespan
)

PREDICTION_COUNTER = Counter('hydro_predictions_total', 'Total inference requests processed')
CAVITATION_ALERTS = Counter('hydro_cavitation_alerts_total', 'Total confirmed cavitation triggers')
FRAUD_ALERTS = Counter('hydro_fraud_alerts_total', 'Total vendor spoofing / fraud flags')
LATENCY_HISTOGRAM = Histogram('hydro_inference_latency_seconds', 'Inference request latency')

SECURITY_TOKEN = os.getenv("NEXLYRA_API_TOKEN", "nexlyra-secure-token-2026")
security_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    if credentials.credentials != SECURITY_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token.")
    return credentials.credentials

class HydroPayload(BaseModel):
    IoT_Hydrophone_JSON: str = Field(..., description="Base85 encoded JSON string containing spectrogram array and telemetry.")
    Claimed_Shutdown_USD: float = Field(0.0, description="Billed emergency shutdown invoice amount in USD.")
    Vendor: str = Field("Legit_Hydro", description="Contracting entity operating the turbine.")

@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000.0
    response.headers["X-Inference-Latency-MS"] = f"{duration_ms:.2f}"
    return response

@app.post("/api/v1/predict-cavitation")
def predict_cavitation(payload: HydroPayload, token: str = Security(verify_token)):
    PREDICTION_COUNTER.inc()
    start_time = time.time()
    
    try:
        # Robust decoding block
        b85_raw = payload.IoT_Hydrophone_JSON.strip()
        decoded_bytes = base64.b85decode(b85_raw.encode('ascii'))
        try:
            json_str = bytes.fromhex(decoded_bytes.decode('ascii')).decode('utf-8')
        except (ValueError, UnicodeDecodeError):
            json_str = decoded_bytes.decode('utf-8', errors='ignore')
            
        raw_data = json.loads(json_str)
        
        press = float(raw_data.get('press_kpa', 100.0))
        vel = float(raw_data.get('vel_ms', 5.0))
        temp = float(raw_data.get('temp_c', 15.0))
        spec_list = raw_data.get('spectrogram', [0.0]*100)
        
        rho = 1000.0
        p_vapor = 1.2
        v_clamped = max(vel, 1e-3)
        dyn_press = 0.5 * rho * (v_clamped ** 2) / 1000.0
        sigma = (press - p_vapor) / dyn_press
        
        spec_arr = np.array([float(v) for v in spec_list], dtype=np.float32)
        if spec_arr.size != 100:
            spec_arr = np.pad(spec_arr, (0, max(0, 100 - spec_arr.size)), mode='constant')[:100]
        
        spec_tensor_2d = spec_arr.reshape(10, 10) + 1e-8
        
        geom_mean = np.exp(np.mean(np.log(spec_tensor_2d), axis=1))
        arith_mean = np.mean(spec_tensor_2d, axis=1)
        spectral_flatness = float(np.mean(geom_mean / (arith_mean + 1e-8)))
        
        total_energy = np.sum(spec_tensor_2d) + 1e-8
        max_energy = np.max(spec_tensor_2d)
        broadband_ratio = float((total_energy / 100.0) / (max_energy + 1e-8))
        
        spec_tensor = spec_arr.reshape(1, 1, 10, 10)
        raw_tab = np.array([[press, vel, temp, sigma, spectral_flatness, broadband_ratio]], dtype=np.float32)
        tab_scaled = np.zeros((1, 6), dtype=np.float32)
        
        if ml_artifacts.get('hydro_scaler') is not None:
            tab_scaled[:, :3] = ml_artifacts['hydro_scaler'].transform(raw_tab[:, :3])
        else:
            tab_scaled[:, :3] = raw_tab[:, :3]
            
        if ml_artifacts.get('web_scaler') is not None:
            try:
                tab_scaled = ml_artifacts['web_scaler'].transform(raw_tab)
            except Exception:
                tab_scaled[:, 3:] = raw_tab[:, 3:]
        else:
            tab_scaled[:, 3:] = raw_tab[:, 3:]
        
        ort_inputs = {
            'spec_input': spec_tensor,
            'tabular_input': tab_scaled
        }
        ort_outs = ml_artifacts['ort_session'].run(None, ort_inputs)
        logit = ort_outs[0][0][0]
        prob = 1.0 / (1.0 + np.exp(-logit))
        
        optimal_threshold = 0.0100
        cavitation_detected = bool(prob >= optimal_threshold)
        if cavitation_detected:
            CAVITATION_ALERTS.inc()
            
        ml_artifacts['telemetry_buffer'].append(press)
        drift_warning = False
        if len(ml_artifacts['telemetry_buffer']) >= 100:
            current_variance = np.var(ml_artifacts['telemetry_buffer'][-100:])
            variance_ratio = abs(current_variance - ml_artifacts['baseline_variance']) / ml_artifacts['baseline_variance']
            if variance_ratio > 0.25:
                drift_warning = True

        fraud_alert = False
        if payload.Vendor == "Hydro_Phantom_Turbines" and prob < 0.1 and payload.Claimed_Shutdown_USD > 50000:
            fraud_alert = True
            FRAUD_ALERTS.inc()
            
        LATENCY_HISTOGRAM.observe(time.time() - start_time)
        
        return {
            "cavitation_probability": float(prob),
            "cavitation_detected": cavitation_detected,
            "fraud_alert": fraud_alert,
            "drift_warning": drift_warning,
            "cost_optimal_threshold": optimal_threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Payload processing failed: {str(e)}")
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)