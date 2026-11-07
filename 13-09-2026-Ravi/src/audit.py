from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from gradcam import run_gradcam
from utils import ensure_dir, save_png_base64

def audit_explanation(
    model: nn.Module,
    image: torch.Tensor,
    tabular: torch.Tensor,
    vendor: str,
    sample_id: int,
    top_shap_feature: str,
    out_dir: str | Path = "day5_artifacts/audit",
) -> tuple[dict[str, Any], Path]:
    out_path = ensure_dir(out_dir)

    cam_data = run_gradcam(model, image, tabular)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cam_data["cam"], cmap="jet")
    ax.axis("off")
    b64_img = save_png_base64(fig)

    audit_payload = {
        "sample_id": sample_id,
        "vendor": vendor,
        "predicted_probability": cam_data["probability"],
        "target_layer": cam_data["layer_name"],
        "top_shap_feature": top_shap_feature,
        "gradcam_base64": f"data:image/png;base64,{b64_img}",
    }

    json_file = out_path / f"audit_sample_{sample_id}.json"
    json_file.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")

    return audit_payload, json_file