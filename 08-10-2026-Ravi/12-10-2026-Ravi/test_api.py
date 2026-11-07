"""
NEXLYRA MLOPS: ASYNCHRONOUS PYTEST SUITE (`test_api.py`)
"""

import pytest
from fastapi.testclient import TestClient
from app import app
import base64
import json

client = TestClient(app)
AUTH_HEADER = {"Authorization": "Bearer nexlyra-secure-token-2026"}

def test_health_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200

def test_predict_genuine_cavitation():
    payload_dict = {
        "press_kpa": 120.0,
        "vel_ms": 12.5,
        "temp_c": 14.0,
        "spectrogram": [float(i) for i in range(100)]
    }
    hex_data = json.dumps(payload_dict).encode('utf-8').hex()
    b85_payload = base64.b85encode(bytes.fromhex(hex_data)).decode('utf-8')
    
    response = client.post(
        "/api/v1/predict-cavitation",
        headers=AUTH_HEADER,
        json={"IoT_Hydrophone_JSON": b85_payload, "Claimed_Shutdown_USD": 150000.0, "Vendor": "Legit_Hydro"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "cavitation_probability" in data
    assert "fraud_alert" in data

def test_predict_fraudulent_spoof():
    payload_dict = {
        "press_kpa": 500.0,
        "vel_ms": 2.0,
        "temp_c": 11.0,
        "spectrogram": [0.0]*100
    }
    hex_data = json.dumps(payload_dict).encode('utf-8').hex()
    b85_payload = base64.b85encode(bytes.fromhex(hex_data)).decode('utf-8')
    
    response = client.post(
        "/api/v1/predict-cavitation",
        headers=AUTH_HEADER,
        json={"IoT_Hydrophone_JSON": b85_payload, "Claimed_Shutdown_USD": 150000.0, "Vendor": "Hydro_Phantom_Turbines"}
    )
    assert response.status_code == 200
    data = response.json()
    print(response)
    assert data["fraud_alert"] is True

def test_corrupted_payload():
    response = client.post(
        "/api/v1/predict-cavitation",
        headers=AUTH_HEADER,
        json={"IoT_Hydrophone_JSON": "INVALID_BASE85_STRING!!!", "Claimed_Shutdown_USD": 1000.0}
    )
    assert response.status_code == 400