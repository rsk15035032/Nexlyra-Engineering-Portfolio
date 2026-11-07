from __future__ import annotations
import base64
import gc
import hashlib
import io
import random
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def scalar_probability(logit: Any) -> float:
    if isinstance(logit, torch.Tensor):
        x = logit.detach().cpu().to(torch.float32)
    else:
        x = torch.tensor(logit, dtype=torch.float32)
    return float(torch.sigmoid(x).reshape(-1)[0].item())

def model_logit(
    model: torch.nn.Module, matrix: torch.Tensor, tabular: torch.Tensor
) -> torch.Tensor:
    out = model(matrix, tabular)
    if isinstance(out, (tuple, list)):
        out = out[0]
    elif isinstance(out, dict):
        for key in ("logit", "logits", "output", "prediction"):
            if key in out:
                out = out[key]
                break
    out = torch.as_tensor(out)
    if out.ndim == 2 and out.shape[-1] == 1:
        return out[:, 0]
    if out.ndim == 2 and out.shape[-1] == 2:
        return out[:, 1] - out[:, 0]
    return out.reshape(-1)

def save_svg(fig: plt.Figure, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    fig.savefig(p, format="svg", bbox_inches="tight")
    plt.close(fig)

def save_png_base64(fig: plt.Figure, dpi: int = 180) -> str:
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    finally:
        buf.close()
        plt.close(fig)

def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def clear_memory(*objects: Any) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()