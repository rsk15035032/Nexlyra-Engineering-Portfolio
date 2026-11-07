"""Inference Module for Dual-Input StrainVisionNet ONNX Engine."""

import numpy as np
import onnxruntime as ort


class NexlyraONNXInference:
    def __init__(self, model_path: str):
        """Initializes ONNX Runtime session and inspects input node bindings."""
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.spatial_input_name = self.session.get_inputs()[0].name
        self.tabular_input_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, spatial_tensor: np.ndarray, tabular_tensor: np.ndarray) -> float:
        """Executes dual-input ONNX inference.

        Args:
            spatial_tensor: Formatted as shape (1, 1, 8, 8), dtype float32.
            tabular_tensor: Formatted as shape (1, 5), dtype float32.

        Returns:
            float: Sigmoid probability for MicroFracture_Triggered.
        """
        assert spatial_tensor.shape == (
            1,
            1,
            8,
            8,
        ), f"Expected spatial shape (1, 1, 8, 8), got {spatial_tensor.shape}"
        assert tabular_tensor.shape == (
            1,
            5,
        ), f"Expected tabular shape (1, 5), got {tabular_tensor.shape}"

        outputs = self.session.run(
            [self.output_name],
            {
                self.spatial_input_name: spatial_tensor,
                self.tabular_input_name: tabular_tensor,
            },
        )
        return float(outputs[0].squeeze())