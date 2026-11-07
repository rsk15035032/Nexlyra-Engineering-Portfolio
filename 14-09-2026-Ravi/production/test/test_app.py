"""Integration test suite for Nexlyra StrainVisionNet API."""

import base64
import json
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

API_KEY = "nexlyra-prod-secret-key-2026"


def create_valid_b64_matrix() -> str:
    """Helper to generate base64-encoded JSON array with 64 elements."""
    raw_list = [0.5] * 64
    json_str = json.dumps(raw_list)
    return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["feature_count"] == 5


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["vendor_is_model_feature"] is False
    assert len(data["feature_columns"]) == 5


def test_predict_unauthorized():
    payload = {
        "matrix_b64": create_valid_b64_matrix(),
        "Axial_Load_Tons": 2.5,
        "Wind_Shear_MPH": 12.0,
    }
    response = client.post("/predict", json=payload, headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_predict_success():
    payload = {
        "matrix_b64": create_valid_b64_matrix(),
        "Axial_Load_Tons": 2.5,
        "Wind_Shear_MPH": 12.0,
    }
    response = client.post("/predict", json=payload, headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    data = response.json()
    assert "MicroFracture_Triggered" in data
    assert "MicroFracture_Probability" in data
    assert "Spatial_Entropy" in data
    assert "Drift_Detected" in data