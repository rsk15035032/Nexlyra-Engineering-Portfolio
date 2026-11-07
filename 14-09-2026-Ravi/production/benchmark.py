"""Inference latency and throughput benchmarking module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmarker")


def benchmark_engines(
    onnx_path: Path,
    report_path: Path,
    num_warmup: int = 20,
    num_runs: int = 200,
) -> dict:
    """Benchmarks ONNX Runtime inference latency (P50, P95, P99) and throughput."""
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model missing at: {onnx_path}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_nodes = session.get_inputs()

    # Dynamic input dictionary matching exact ONNX input node names
    ort_inputs = {}
    for node in input_nodes:
        if "spatial" in node.name.lower() or "strain" in node.name.lower():
            ort_inputs[node.name] = np.random.randn(1, 1, 8, 8).astype(np.float32)
        elif "tabular" in node.name.lower():
            ort_inputs[node.name] = np.random.randn(1, 5).astype(np.float32)
        else:
            # Fallback based on shape rank
            if len(node.shape) == 4:
                ort_inputs[node.name] = np.random.randn(1, 1, 8, 8).astype(np.float32)
            else:
                ort_inputs[node.name] = np.random.randn(1, 5).astype(np.float32)

    # 1. Warmup Runs
    for _ in range(num_warmup):
        _ = session.run(None, ort_inputs)

    # 2. Latency Benchmarking
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, ort_inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # Convert to ms

    latencies = np.array(latencies)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean_latency = float(np.mean(latencies))
    throughput_qps = float(1000.0 / mean_latency)

    report = {
        "num_samples": num_runs,
        "mean_latency_ms": round(mean_latency, 4),
        "p50_latency_ms": round(p50, 4),
        "p95_latency_ms": round(p95, 4),
        "p99_latency_ms": round(p99, 4),
        "throughput_qps": round(throughput_qps, 2),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info("Benchmark report saved to %s", report_path)
    logger.info("Performance Summary: %s", json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    model_path = root / "models" / "strain_vision_net.onnx"
    output_report = root / "artifacts" / "benchmark_report.json"
    benchmark_engines(model_path, output_report)