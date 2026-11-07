"""
MANUAL API TEST SCRIPT (`test_manual.py`)
"""

import requests
import base64
import json

# API Endpoint URL
URL = "http://127.0.0.1:8000/api/v1/predict-cavitation"

# Bearer Token (Matches SECURITY_TOKEN in app.py)
HEADERS = {
    "Authorization": "Bearer nexlyra-secure-token-2026",
    "Content-Type": "application/json"
}

# Construct Sample Telemetry Payload
payload_dict = {
    "press_kpa": 115.5,
    "vel_ms": 14.2,
    "temp_c": 13.8,
    "spectrogram": [float(i % 10) for i in range(100)]
}

# Encode to JSON bytes and then Base85 ASCII string
json_bytes = json.dumps(payload_dict).encode('utf-8')
b85_payload = base64.b85encode(json_bytes).decode('ascii')

# Request Body matching HydroPayload schema
request_body = {
    "IoT_Hydrophone_JSON": b85_payload,
    "Claimed_Shutdown_USD": 75000.0,
    "Vendor": "Hydro_Phantom_Turbines"  # Will trigger fraud alert if prob < 0.1
}

print("[INFO] Sending manual POST request to Nexlyra MLOps microservice...")
response = requests.post(URL, headers=HEADERS, json=request_body)

print(f"Status Code : {response.status_code}")
print(f"Response Headers (Latency): X-Inference-Latency-MS = {response.headers.get('X-Inference-Latency-MS')}")
print("Response JSON:")
print(json.dumps(response.json(), indent=4))