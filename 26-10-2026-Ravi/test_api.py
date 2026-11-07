import pytest
import zlib
import base64
import json
from fastapi.testclient import TestClient
from main import app, API_SECRET_TOKEN

client = TestClient(app)

def encode_payload(spectrogram_grid, physics_vals):
    raw_dict = {
        "spectrogram": spectrogram_grid,
        "physics_features": physics_vals
    }
    json_str = json.dumps(raw_dict)
    compressed = zlib.compress(json_str.encode('utf-8'))
    b85_encoded = base64.b85encode(compressed).decode('ascii')
    return b85_encoded


def test_true_cavitation_payload():
    spec = [[0.1]*10 for _ in range(10)]
    spec[4][4] = 0.95  # Simulate vertical broadband shockwave
    physics = [1013.25, 12.5, 1.8, 45.0, 1.1, 285.0]
    
    payload = {
        "IoT_Hydrophone_JSON": encode_payload(spec, physics),
        "Claimed_Invoice_USD": 450000.0,
        "Contractor_Name": "Nexlyra_Internal_Sensors"
    }
    
    headers = {"Authorization": f"Bearer {API_SECRET_TOKEN}"}
    response = client.post("/api/v1/predict-cavitation", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "cavitation_probability" in data


def test_hydro_phantom_spoofed_payload():
    # Spoofed flat acoustic spectrum with high static pressure override
    spec = [[0.1]*10 for _ in range(10)]
    physics = [2500.0, 5.0, 3.2, 10.0, 0.4, 290.0] # High pressure neutralizing vibration
    
    payload = {
        "IoT_Hydrophone_JSON": encode_payload(spec, physics),
        "Claimed_Invoice_USD": 450000.0,
        "Contractor_Name": "Hydro_Phantom_Turbines"
    }
    
    headers = {"Authorization": f"Bearer {API_SECRET_TOKEN}"}
    response = client.post("/api/v1/predict-cavitation", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["fraud_alert"] is True


def test_corrupted_payload():
    payload = {
        "IoT_Hydrophone_JSON": "CORRUPTED_BASE85_STRING!!!",
        "Claimed_Invoice_USD": 10000.0,
        "Contractor_Name": "Untrusted_Vendor"
    }
    headers = {"Authorization": f"Bearer {API_SECRET_TOKEN}"}
    response = client.post("/api/v1/predict-cavitation", json=payload, headers=headers)
    assert response.status_code == 400