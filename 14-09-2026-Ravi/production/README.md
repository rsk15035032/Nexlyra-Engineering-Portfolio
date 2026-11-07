# StrainVisionNet: Real-Time Dual-Input Microfracture Detection Service

StrainVisionNet is a high-throughput, low-latency microservice engineered to predict structural microfracture triggers (`MicroFracture_Triggered`). It processes dual-input telemetry data combining spatial $8 \times 8$ strain matrices with 5-feature tabular sensor vectors using an optimized, INT8-quantized ONNX model hosted via FastAPI.

---

## Key Architecture Features

* **Dual-Input Neural Network:** Combines a 2D CNN spatial processing branch with an MLP tabular feature network via late fusion.
* **Quantized ONNX Engine:** Uses INT8 dynamic quantization for CPU deployment, delivering an **~8.5x compression ratio** ($1.81\text{ MB} \rightarrow 0.21\text{ MB}$) with $<1.0\%$ AUROC degradation.
* **Entropy-Based Drift Detection:** Real-time Shannon entropy monitoring on spatial strain data to detect domain/distribution shifts ($>20\%$ threshold).
* **Production-Grade API:** Async FastAPI microservice with authentication, lifespan lifecycle management, and pre-load validation.

---

## Input Specification & Features

### 1. Spatial Input (`spatial_input`)
* **Shape:** `[batch_size, 1, 8, 8]`
* **Format:** Base64-encoded `float32` vector (64 elements).

### 2. Tabular Inputs (`tabular_input`)
* **Shape:** `[batch_size, 5]`
* **Features:**
  1. `Axial_Load_Tons`
  2. `Wind_Shear_MPH`
  3. `Symmetry_V`
  4. `Symmetry_H`
  5. `Center_to_Edge_Ratio`

---

## Service Architecture & Directory Structure

```text
├── artifacts/
│   ├── spatial_scaler.pkl        # StandardScaler for spatial strain tensors
│   ├── tabular_scaler.pkl        # StandardScaler for tabular features
│   ├── strain_drift_baseline.json# Baseline Shannon entropy metric
│   └── quantization_report.json  # INT8 parity and compression audit
├── models/
│   ├── best_architec.py          # PyTorch StrainVisionNet architecture
│   ├── best_model.pt             # Trained PyTorch checkpoint
│   └── strain_vision_net.onnx    # Quantized ONNX model artifact
├── tests/
│   └── test_app.py               # Integration & endpoint test suite
├── app.py                        # FastAPI microservice with Lifespan context
├── benchmark.py                  # ONNX Runtime latency/throughput test script
├── drift.py                      # Shannon entropy drift detection module
├── export_onnx.py                # PyTorch to ONNX graph exporter
├── quantization.py               # INT8 dynamic quantization script
└── validate_onnx.py              # ONNX graph interface and sanity validator
```

Execution & Pipeline Deployment
1. Environment Setup
```Bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Export, Quantize & Validate Model
Execute the end-to-end model optimization pipeline:

Bash
# Export PyTorch checkpoint to ONNX format
python export_onnx.py

# Perform INT8 Dynamic Quantization and evaluate parity
python quantization.py

# Verify ONNX tensor interface integrity
python validate_onnx.py

# Benchmark latency (P50, P95, P99) and throughput
python benchmark.py
3. Launch the API Service
Start the FastAPI microservice using Uvicorn:

```Bash
export NEXLYRA_API_KEY="nexlyra-prod-secret-key-2026"
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
API Reference
POST /predict
Executes dual-input inference, calculates spatial entropy, checks for drift, and returns bounded microfracture probabilities.

Request Headers
Content-Type: application/json
X-API-Key: nexlyra-prod-secret-key-2026
```

Example Request Payload
```JSON
{
  "matrix_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
  "Axial_Load_Tons": 0.689,
  "Wind_Shear_MPH": 0.735,
  "Symmetry_V": 0.270,
  "Symmetry_H": 0.104,
  "Center_to_Edge_Ratio": 1.000
}
```
# Example Response
```JSON
{
  "MicroFracture_Triggered": false,
  "MicroFracture_Probability": 0.0421,
  "Spatial_Entropy": 0.0,
  "Drift_Detected": false
}
```
# Containerization (Docker)
To containerize and run the microservice in production:

Bash
# Build Docker image
``` bash
docker build -t strainvisionnet-api:latest .
```

# Run container
```bash
docker run -d \
  --name strainvisionnet_service \
  -p 8000:8000 \
  -e NEXLYRA_API_KEY="nexlyra-prod-secret-key-2026" \
  strainvisionnet-api:latest

```

# Verification & Testing
- Run automated tests using pytest:
```bash
pytest tests/test_app.py -v
```