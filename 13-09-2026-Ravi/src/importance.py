from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import CFG
from utils import model_logit

def permutation_importance_tabular(
    model: nn.Module,
    images: torch.Tensor,
    tabulars: torch.Tensor,
    targets: np.ndarray | torch.Tensor,
) -> pd.DataFrame:
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)
    tabulars = tabulars.to(device)

    with torch.no_grad():
        base_logits = model_logit(model, images, tabulars)
        base_probs = torch.sigmoid(base_logits).cpu().numpy()

    y = np.asarray(targets)
    base_mse = float(np.mean((base_probs - y) ** 2))

    scores = []
    for col_idx, feature_name in enumerate(CFG.tabular_features):
        tab_perm = tabulars.clone()
        perm_indices = torch.randperm(len(tabulars))
        tab_perm[:, col_idx] = tabulars[perm_indices, col_idx]

        with torch.no_grad():
            perm_logits = model_logit(model, images, tab_perm)
            perm_probs = torch.sigmoid(perm_logits).cpu().numpy()

        perm_mse = float(np.mean((perm_probs - y) ** 2))
        importance = perm_mse - base_mse
        scores.append(
            {"feature": feature_name, "importance_mse_drop": importance}
        )

    df = pd.DataFrame(scores)
    return df.sort_values(by="importance_mse_drop", ascending=False).reset_index(
        drop=True
    )

def rank_spatial_vs_tabular(
    model: nn.Module,
    images: torch.Tensor,
    tabulars: torch.Tensor,
    targets: np.ndarray | torch.Tensor,
) -> pd.DataFrame:
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)
    tabulars = tabulars.to(device)
    y = np.asarray(targets)

    with torch.no_grad():
        base_probs = (
            torch.sigmoid(model_logit(model, images, tabulars)).cpu().numpy()
        )
        base_mse = float(np.mean((base_probs - y) ** 2))

        # Permute spatial inputs
        perm_img = images[torch.randperm(len(images))]
        spatial_probs = (
            torch.sigmoid(model_logit(model, perm_img, tabulars)).cpu().numpy()
        )
        spatial_mse_drop = float(np.mean((spatial_probs - y) ** 2)) - base_mse

        # Permute tabular inputs
        perm_tab = tabulars[torch.randperm(len(tabulars))]
        tabular_probs = (
            torch.sigmoid(model_logit(model, images, perm_tab)).cpu().numpy()
        )
        tabular_mse_drop = float(np.mean((tabular_probs - y) ** 2)) - base_mse

    df = pd.DataFrame(
        [
            {"modality": "Spatial (Strain Matrix)", "mse_drop": spatial_mse_drop},
            {"modality": "Tabular Features", "mse_drop": tabular_mse_drop},
        ]
    )
    return df.sort_values(by="mse_drop", ascending=False).reset_index(drop=True)