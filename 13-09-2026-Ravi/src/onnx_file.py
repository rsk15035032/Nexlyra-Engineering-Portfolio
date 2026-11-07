from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn

def export_dual_input_onnx(
    model: nn.Module,
    image_example: torch.Tensor,
    tabular_example: torch.Tensor,
    out_path: str | Path,
) -> Path:
    model = model.eval().cpu()
    image_example = image_example.cpu().float()
    tabular_example = tabular_example.cpu().float()

    if image_example.ndim == 2:
        image_example = image_example.unsqueeze(0).unsqueeze(0)
    elif image_example.ndim == 3:
        image_example = image_example.unsqueeze(0)

    if tabular_example.ndim == 1:
        tabular_example = tabular_example.unsqueeze(0)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (image_example, tabular_example),
        f=str(out_file),
        input_names=["strain_matrix", "tabular_features"],
        output_names=["logit"],
        dynamic_axes={
            "strain_matrix": {0: "batch_size"},
            "tabular_features": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
        opset_version=17,
    )
    return out_file