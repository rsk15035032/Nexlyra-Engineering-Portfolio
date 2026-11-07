from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from utils import model_logit
except ImportError:
    model_logit = None

class SpatialOnlyWrapper(nn.Module):
    def __init__(self, dual_model: nn.Module, tabular_baseline: torch.Tensor):
        super().__init__()
        self.model = copy.deepcopy(dual_model)
        self.model.eval()

        if tabular_baseline.ndim == 1:
            tabular_baseline = tabular_baseline.unsqueeze(0)

        self.register_buffer("tabular_baseline", tabular_baseline.detach().clone())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tab = self.tabular_baseline.to(device=image.device, dtype=image.dtype)
        if tab.shape[0] != image.shape[0]:
            tab = tab[:1].expand(image.shape[0], -1)

        if model_logit is not None:
            out = model_logit(self.model, image, tab)
        else:
            out = self.model(image, tab)

        if isinstance(out, (tuple, list)):
            out = out[0]
        elif isinstance(out, dict):
            out = out.get("logit", out.get("logits", out.get("output")))

        return out

@dataclass
class TrainResult:
    model: nn.Module
    best_val_auc: float
    test_auc: float
    test_ap: float
    history: list[dict]

def _logit(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    elif isinstance(out, dict):
        out = out.get("logit", out.get("logits", out.get("output")))
    return torch.as_tensor(out).reshape(-1)

def train_spatial_only(
    model_factory: Callable[[], nn.Module],
    train_loader,
    val_loader,
    test_loader,
    device: str | torch.device = "cpu",
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: float | None = None,
    checkpoint_path: str | Path = "day5_artifacts/spatial_only_best.pt",
) -> TrainResult:
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    weight_tensor = (
        torch.tensor([pos_weight], device=device, dtype=torch.float32)
        if pos_weight is not None
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    best_auc = -float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0

        for batch in train_loader:
            image, _, y = batch[:3]
            image = image.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32).reshape(-1)

            optimizer.zero_grad(set_to_none=True)
            logits = _logit(model, image)
            loss = criterion(logits, y)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(y)
            n += len(y)

        model.eval()
        val_p, val_y = [], []
        with torch.no_grad():
            for batch in val_loader:
                image, _, y = batch[:3]
                p = torch.sigmoid(_logit(model, image.to(device, dtype=torch.float32))).cpu().numpy()
                val_p.append(p)
                val_y.append(y.detach().cpu().numpy().reshape(-1))

        val_p = np.concatenate(val_p)
        val_y = np.concatenate(val_y)
        val_auc = float(roc_auc_score(val_y, val_p)) if len(np.unique(val_y)) > 1 else 0.5

        history.append({"epoch": epoch, "train_loss": train_loss / max(n, 1), "val_auc": val_auc})

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_auc": val_auc},
                str(ckpt_path),
            )

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_p, test_y = [], []
    with torch.no_grad():
        for batch in test_loader:
            image, _, y = batch[:3]
            p = torch.sigmoid(_logit(model, image.to(device, dtype=torch.float32))).cpu().numpy()
            test_p.append(p)
            test_y.append(y.detach().cpu().numpy().reshape(-1))

    test_p = np.concatenate(test_p)
    test_y = np.concatenate(test_y)

    test_auc = float(roc_auc_score(test_y, test_p)) if len(np.unique(test_y)) > 1 else 0.5
    test_ap = float(average_precision_score(test_y, test_p)) if len(np.unique(test_y)) > 1 else 0.0

    return TrainResult(
        model=model,
        best_val_auc=float(best_auc),
        test_auc=test_auc,
        test_ap=test_ap,
        history=history,
    )