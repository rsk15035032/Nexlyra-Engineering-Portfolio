from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from utils import save_svg

def quadrant_segmentation(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    seg = np.zeros((h, w), dtype=np.int32)
    seg[h // 2 :, : w // 2] = 0
    seg[h // 2 :, w // 2 :] = 1
    seg[: h // 2, : w // 2] = 2
    seg[: h // 2, w // 2 :] = 3
    return seg

def make_lime_explanation(
    predict_proba, matrix_8x8, num_samples: int = 1000
):
    if hasattr(matrix_8x8, "detach"):
        img = matrix_8x8.detach().cpu().numpy().squeeze()
    else:
        img = np.squeeze(matrix_8x8)

    img = np.asarray(img, dtype=np.float32)
    lo, hi = float(img.min()), float(img.max())
    denom = hi - lo if hi > lo else 1.0
    img = (img - lo) / denom

    if img.ndim == 2:
        rgb = np.repeat(img[..., None], 3, axis=2)
    else:
        rgb = img

    explainer = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        rgb,
        classifier_fn=predict_proba,
        top_labels=2,
        hide_color=0.0,
        num_samples=num_samples,
        segmentation_fn=quadrant_segmentation,
    )
    return explanation, rgb

def plot_lime(explanation, rgb: np.ndarray, out_svg: str) -> None:
    if hasattr(explanation, "top_labels") and len(explanation.top_labels) > 0:
        label = explanation.top_labels[0]
    else:
        label = 1

    temp, mask = explanation.get_image_and_mask(
        label, positive_only=False, num_features=4, hide_rest=False
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mark_boundaries(rgb, mask))
    ax.set_title("LIME — 4-Quadrant Explanation", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    save_svg(fig, out_svg)