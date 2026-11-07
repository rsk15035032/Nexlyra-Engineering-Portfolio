"""
MANUAL API TEST SCRIPT (`test_manual.py`)
"""

import requests
import base64
import zlib
import json

# API Endpoint URL
URL = "http://127.0.0.1:8000/api/v1/predict-cavitation"

# Bearer Token (Matches API_SECRET_TOKEN in main.py)
HEADERS = {
    "Authorization": "Bearer nexlyra-secure-bearer-2026",
    "Content-Type": "application/json"
}

# Construct Sample Telemetry Payload matching the 6 physics features expected by the model
payload_dict = {
    "spectrogram": [float(i % 10) for i in range(100)],
    "physics_features": [1013.25, 12.5, 1.8, 45.0, 1.1, 285.0]
}

# Compress using zlib and encode to Base85 ASCII string
json_bytes = json.dumps(payload_dict).encode('utf-8')
compressed_bytes = zlib.compress(json_bytes)
b85_payload = base64.b85encode(compressed_bytes).decode('ascii')
print(f"Generated Base85 Payload length: {len(b85_payload)}")

# Request Body matching HydroPayload schema in main.py
request_body = {
    "IoT_Hydrophone_JSON": b85_payload,
    "Claimed_Invoice_USD": 450000.0,
    "Contractor_Name": "Hydro_Phantom_Turbines"  # Will trigger fraud alert if probability < optimal_threshold
}

print("[INFO] Sending manual POST request to Nexlyra MLOps microservice...")
response = requests.post(URL, headers=HEADERS, json=request_body)

print(f"Status Code : {response.status_code}")
print(f"Response Headers (Latency): X-Inference-Latency-MS = {response.headers.get('X-Inference-Latency-MS')}")
print("Response JSON:")
print(json.dumps(response.json(), indent=4))