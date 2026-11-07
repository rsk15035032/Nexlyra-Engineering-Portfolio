from __future__ import annotations
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from utils import model_logit, save_svg

def find_last_conv2d(model: nn.Module) -> tuple[str, nn.Conv2d]:
    last_name, last_module = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name, last_module = name, module
    if last_module is None:
        raise ValueError("No Conv2d module found in the architecture.")
    return last_name, last_module

def run_gradcam(
    model: nn.Module, image: torch.Tensor, tabular: torch.Tensor
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device

    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)

    if tabular.ndim == 1:
        tabular = tabular.unsqueeze(0)

    image = image.to(device).requires_grad_(True)
    tabular = tabular.to(device)

    layer_name, target_layer = find_last_conv2d(model)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    logit = model_logit(model, image, tabular)
    logit.backward()

    h1.remove()
    h2.remove()

    act = activations[0].detach().cpu().numpy()[0]
    grad = gradients[0].detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    prob = float(torch.sigmoid(logit).detach().cpu().item())
    return {
        "cam": cam,
        "probability": prob,
        "layer_name": layer_name,
    }

def plot_cam(
    image: np.ndarray, cam: np.ndarray, title: str, out_svg: str
) -> None:
    img = np.squeeze(image)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img, cmap="viridis")
    axes[0].set_title("Original Input")
    axes[0].axis("off")

    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(cam, cmap="jet", alpha=0.5)
    axes[1].set_title(f"Grad-CAM: {title}")
    axes[1].axis("off")

    fig.tight_layout()
    save_svg(fig, out_svg)