"""Validation module to verify ONNX model integrity, node names, and shape consistency."""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("onnx_validator")


def validate_onnx_export(onnx_path: Path) -> bool:
    """Validates structural integrity and runs inference pass on exported ONNX model."""
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model file not found at: {onnx_path}")

    logger.info("Validating ONNX artifact: %s", onnx_path)

    # 1. Structural Check
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("✓ ONNX structural integrity check PASSED.")

    # 2. Inspect Inputs and Outputs
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    input_nodes = session.get_inputs()
    output_nodes = session.get_outputs()

    logger.info("\n--- Model Interface Specification ---")
    for idx, inp in enumerate(input_nodes):
        logger.info("Input %d: Name='%s', Shape=%s, Type=%s", idx, inp.name, inp.shape, inp.type)
    for idx, out in enumerate(output_nodes):
        logger.info("Output %d: Name='%s', Shape=%s, Type=%s", idx, out.name, out.shape, out.type)

    # 3. Dynamic Execution Feed (Automatically maps to whatever input names the ONNX model defines)
    ort_inputs = {}
    for node in input_nodes:
        if "spatial" in node.name.lower() or "strain" in node.name.lower():
            ort_inputs[node.name] = np.random.randn(1, 1, 8, 8).astype(np.float32)
        elif "tabular" in node.name.lower():
            ort_inputs[node.name] = np.random.randn(1, 5).astype(np.float32)
        else:
            # Fallback based on shape dimensions
            if len(node.shape) == 4:
                ort_inputs[node.name] = np.random.randn(1, 1, 8, 8).astype(np.float32)
            else:
                ort_inputs[node.name] = np.random.randn(1, 5).astype(np.float32)

    # 4. Run Inference Pass
    ort_outputs = session.run(None, ort_inputs)
    pred_prob = float(ort_outputs[0].squeeze())

    logger.info("\nSample Prediction Output: %.4f", pred_prob)
    assert 0.0 <= pred_prob <= 1.0, f"Probability output {pred_prob} out of bounds [0.0, 1.0]"
    logger.info("✓ Sanity execution test PASSED.")
    return True


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    model_file = root / "models" / "strain_vision_net.onnx"
    validate_onnx_export(model_file)