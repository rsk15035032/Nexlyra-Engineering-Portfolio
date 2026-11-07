from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from utils import save_svg

def compute_shap(
    model: nn.Module,
    images: torch.Tensor,
    tabulars: torch.Tensor,
    n: int = 100,
) -> tuple[np.ndarray, list[int], shap.KernelExplainer]:
    model.eval()
    device = next(model.parameters()).device

    bg_tabular = tabulars[: min(10, len(tabulars))].to(device)
    bg_image = images[0:1].to(device)

    def model_predict(tabular_np: np.ndarray) -> np.ndarray:
        tab_t = torch.tensor(tabular_np, dtype=torch.float32, device=device)
        img_t = bg_image.repeat(len(tab_t), 1, 1, 1)
        with torch.no_grad():
            out = model(img_t, tab_t)
            if isinstance(out, (tuple, list)):
                out = out[0]
            elif isinstance(out, dict):
                out = out.get("logit", out.get("logits", out.get("output")))
            return torch.sigmoid(torch.as_tensor(out).reshape(-1)).cpu().numpy()

    explainer = shap.KernelExplainer(model_predict, bg_tabular.cpu().numpy())
    eval_tabular = tabulars[:n].cpu().numpy()
    shap_values = explainer.shap_values(eval_tabular)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    return shap_values, list(range(len(eval_tabular))), explainer

def feature_means(
    shap_values: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    abs_means = np.abs(shap_values).mean(axis=0)
    raw_means = shap_values.mean(axis=0)
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": abs_means,
            "mean_shap": raw_means,
        }
    )
    return df.sort_values(by="mean_abs_shap", ascending=False).reset_index(
        drop=True
    )

def summary_plot(
    shap_values: np.ndarray, feature_names: list[str], out_svg: str | Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        features=feature_names,
        feature_names=feature_names,
        show=False,
    )
    save_svg(fig, out_svg)