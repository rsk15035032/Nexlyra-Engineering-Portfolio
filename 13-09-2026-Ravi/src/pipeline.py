from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from audit import audit_explanation
from config import CFG
from explainibility import compute_shap, feature_means, summary_plot
from filters import plot_first_layer_filters
from gradcam import plot_cam, run_gradcam
from importance import permutation_importance_tabular, rank_spatial_vs_tabular
from lime_main import make_lime_explanation, plot_lime
from onnx_file import export_dual_input_onnx
from report import build_evidence_report
from utils import ensure_dir, seed_everything

def normalize_sample(sample):
    image, tab, y = sample[:3]
    image = torch.as_tensor(image).float()
    tab = torch.as_tensor(tab).float()
    y = int(torch.as_tensor(y).item())
    if image.ndim == 2:
        image = image.unsqueeze(0)
    return image, tab, y

def find_true_positive(dataset, model, max_scan=None):
    model.eval()
    device = next(model.parameters()).device
    n = len(dataset) if max_scan is None else min(len(dataset), max_scan)
    for i in range(n):
        image, tab, y = normalize_sample(dataset[i])
        if y != 1:
            continue
        with torch.no_grad():
            logit = model(image.unsqueeze(0).to(device), tab.unsqueeze(0).to(device))
            if isinstance(logit, (tuple, list)):
                logit = logit[0]
            if isinstance(logit, dict):
                logit = logit.get("logit", logit.get("logits", logit.get("output")))
            p = torch.sigmoid(torch.as_tensor(logit).reshape(-1)[0]).item()
        if p >= 0.5:
            return i, p
    return None, None

def find_vendor_example(dataset, vendors, model=None, spoof_only=False, n_scan=None):
    limit = len(dataset) if n_scan is None else min(len(dataset), n_scan)
    for i in range(limit):
        if vendors[i] != CFG.vendor_name:
            continue
        image, tab, y = normalize_sample(dataset[i])
        if spoof_only and y != 1:
            continue
        if model is None:
            return i
        device = next(model.parameters()).device
        with torch.no_grad():
            logit = model(image.unsqueeze(0).to(device), tab.unsqueeze(0).to(device))
            if isinstance(logit, (tuple, list)):
                logit = logit[0]
            if isinstance(logit, dict):
                logit = logit.get("logit", logit.get("logits", logit.get("output")))
            p = torch.sigmoid(torch.as_tensor(logit).reshape(-1)[0]).item()
        if (not spoof_only) or p >= 0.5:
            return i
    return None

def run_flow(model, test_dataset, test_vendor, test_y=None, out_dir="day5_artifacts"):
    seed_everything(CFG.seed)
    out = ensure_dir(out_dir)
    model.eval()

    # True Positive Grad-CAM
    tp_idx, tp_prob = find_true_positive(test_dataset, model)
    if tp_idx is None:
        raise RuntimeError("No verified true positive found in dataset.")
    tp_img, tp_tab, tp_y = normalize_sample(test_dataset[tp_idx])
    tp_cam = run_gradcam(model, tp_img, tp_tab)
    plot_cam(
        tp_img.numpy(),
        tp_cam["cam"],
        f"True Positive | p={tp_prob:.4f}",
        out / "01_true_positive_gradcam.svg",
    )

    # Vendor Spoofed Grad-CAM
    vendor_idx = find_vendor_example(test_dataset, test_vendor, model, spoof_only=True)
    if vendor_idx is None:
        raise RuntimeError(f"No spoofed {CFG.vendor_name} sample found.")
    v_img, v_tab, v_y = normalize_sample(test_dataset[vendor_idx])
    v_cam = run_gradcam(model, v_img, v_tab)
    plot_cam(
        v_img.numpy(),
        v_cam["cam"],
        f"Titan Alloy spoof | p={v_cam['probability']:.4f}",
        out / "02_titan_spoof_gradcam.svg",
    )

    # SHAP Feature Evaluation
    images, tabs, labels = [], [], []
    for i in range(min(len(test_dataset), CFG.n_shap)):
        im, tb, yy = normalize_sample(test_dataset[i])
        images.append(im)
        tabs.append(tb)
        labels.append(yy)
    images_t = torch.stack(images)
    tabs_t = torch.stack(tabs)

    shap_values, shap_idx, explainer = compute_shap(model, images_t, tabs_t, n=CFG.n_shap)
    shap_summary = feature_means(shap_values, list(CFG.tabular_features))
    shap_summary.to_csv(out / "shap_feature_summary.csv", index=False)
    summary_plot(shap_values, list(CFG.tabular_features), out / "03_shap_summary.svg")

    # Partial Dependence Diagnostic
    from sklearn.inspection import PartialDependenceDisplay
    from sklearn.linear_model import LogisticRegression

    X = tabs_t.numpy()
    y = np.asarray(labels)
    proxy = LogisticRegression(max_iter=1000).fit(X, y)
    fig, ax = plt.subplots(figsize=(7, 5))
    PartialDependenceDisplay.from_estimator(proxy, X, [0], ax=ax)
    fig.tight_layout()
    fig.savefig(out / "04_axial_load_partial_dependence.svg", format="svg")
    plt.close(fig)

    # Feature Interaction Visual Diagnostic
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.35)
    ax.set_xlabel("Axial_Load_Tons")
    ax.set_ylabel("Wind_Shear_MPH")
    ax.set_title("Wind shear × axial load interaction diagnostic")
    fig.tight_layout()
    fig.savefig(out / "05_wind_axial_interaction.svg", format="svg")
    plt.close(fig)

    # First Layer Filter Inspection
    filter_name, _ = plot_first_layer_filters(model, out / "06_first_layer_filters.svg")

    # Permutation Importance Analysis
    pi = permutation_importance_tabular(model, images_t, tabs_t, y)
    pi.to_csv(out / "permutation_importance.csv", index=False)
    rank = rank_spatial_vs_tabular(model, images_t, tabs_t, y)
    rank.to_csv(out / "spatial_vs_tabular.csv", index=False)

    # Generate Audit Payload
    audit, audit_path = audit_explanation(
        model,
        v_img,
        v_tab,
        vendor=CFG.vendor_name,
        sample_id=vendor_idx,
        top_shap_feature=shap_summary.iloc[0]["feature"],
        out_dir=out / "audit",
    )

    # Export ONNX model
    onnx_path = export_dual_input_onnx(model, v_img, v_tab, out / "model.onnx")

    manifest = {
        "seed": CFG.seed,
        "true_positive_index": tp_idx,
        "true_positive_probability": tp_prob,
        "titan_spoof_index": vendor_idx,
        "titan_spoof_probability": v_cam["probability"],
        "gradcam_target_layer": tp_cam["layer_name"],
        "first_conv2d": filter_name,
        "shap_features": shap_summary.to_dict(orient="records"),
        "audit_json": str(audit_path),
        "onnx_model": str(onnx_path),
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    build_evidence_report(out)
    return manifest