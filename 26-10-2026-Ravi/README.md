# Nexlyra HydroCavitationNet MLOps Pipeline

Production-grade MLOps deployment pipeline for industrial dam turbine cavitation detection and vendor billing audit.

## Key Features & Architecture
* **Dual-Input ViT Model**: Fuses acoustic spectrograms (`[1, 10, 10]`) with physical telemetry features via `HydroTransformerNet`.
* **ONNX & INT8 Optimization**: Exports models to ONNX (Opset 17) with dynamic axes and applies INT8 dynamic quantization, achieving sub-millisecond inference (~0.41 ms).
* **Hardened FastAPI Microservice**: Implements Bearer token security, Zlib-Hex-Base85 payload decoding, Kolmogorov-Smirnov (KS) data drift detection, and cost-optimal thresholding.
* **Telemetry & Observability**: Built-in request latency injection headers (`X-Inference-Latency-MS`) and Prometheus metrics tracking.

## Project Structure
* `artchitecture.py`: Core PyTorch multi-modal transformer architecture.
* `export_quantization.py`: Script for FP32 ONNX export, runtime benchmarking, and INT8 dynamic quantization.
* `main.py`: Production FastAPI microservice with asynchronous lifespan management.
* `test_manual.py`: End-to-end test script verifying predictions, latencies, and fraud-detection rules.

## Quick Start Guide
* **1. Export and Quantize Model**:
  ```bash
  1. python export_quantization.py

  2. Launch FastAPI Microservice:
   uvicorn main:app --reload
  3. Run Manual Verification:
   python test_manual.py

```