from __future__ import annotations
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import save_svg

def plot_first_layer_filters(
    model: nn.Module, out_svg: str
) -> tuple[str, nn.Conv2d]:
    first_name, first_layer = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_name, first_layer = name, module
            break

    if first_layer is None:
        raise ValueError("No Conv2d module found in the architecture.")

    weights = first_layer.weight.detach().cpu().numpy()
    num_filters = min(16, weights.shape[0])
    cols = 4
    rows = (num_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if num_filters > 1 else [axes]

    for i in range(num_filters):
        kernel = weights[i, 0] if weights.ndim == 4 else weights[i]
        axes[i].imshow(kernel, cmap="coolwarm")
        axes[i].set_title(f"F{i+1}", fontsize=8)
        axes[i].axis("off")

    for i in range(num_filters, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"First Conv2d Layer Filters ({first_name})", fontsize=10)
    fig.tight_layout()
    save_svg(fig, out_svg)

    return first_name, first_layer