"""ONNX Export Module for Nexlyra StrainVisionNet.

Exports trained PyTorch dual-input architecture to optimized ONNX graph.
"""

from __future__ import annotations

import logging
from pathlib import Path
import torch

from models.best_architec import StrainVisionNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("onnx_exporter")


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    tabular_dim: int = 5,
) -> Path:
    """Exports a trained PyTorch model checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch model checkpoint (.pt/.pth).
        output_path: Target path for the exported .onnx model.
        tabular_dim: Number of tabular feature inputs (default: 5).

    Returns:
        Path to the exported ONNX artifact.

    Raises:
        FileNotFoundError: If the PyTorch checkpoint does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading PyTorch model checkpoint from %s...", checkpoint_path)
    model = StrainVisionNet(tabular_dim=tabular_dim)
    loaded = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if isinstance(loaded, torch.nn.Module):
        model = loaded
    elif isinstance(loaded, dict):
        state_dict = loaded.get("model_state_dict", loaded.get("state_dict", loaded))
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Exact batch dimensions: Spatial (1, 1, 8, 8) and Tabular (1, 5)
    dummy_spatial = torch.randn(1, 1, 8, 8, dtype=torch.float32)
    dummy_tabular = torch.randn(1, tabular_dim, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting ONNX graph to %s...", output_path)
    torch.onnx.export(
        model,
        (dummy_spatial, dummy_tabular),
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["spatial_input", "tabular_input"],
        output_names=["microfracture_probability"],
        dynamic_axes={
            "spatial_input": {0: "batch_size"},
            "tabular_input": {0: "batch_size"},
            "microfracture_probability": {0: "batch_size"},
        },
        dynamo=False,
    )

    logger.info("Successfully exported ONNX model to %s", output_path)
    return output_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    ckpt = root / "models" / "best_model.pth"
    out = root / "models" / "strain_vision_net.onnx"
    export_to_onnx(ckpt, out)