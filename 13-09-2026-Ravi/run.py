from __future__ import annotations

import json
import sys
from pathlib import Path
import torch
import torch.nn as nn

# 1. SHAP Patching for Version Compatibility
import shap

if not hasattr(shap, "Explainer"):
    if hasattr(shap, "KernelExplainer"):
        def _explainer_fallback(model_fn, maskers=None, algorithm=None, feature_names=None, **kwargs):
            return shap.KernelExplainer(model_fn, maskers, **kwargs)
        shap.Explainer = _explainer_fallback
    else:
        from shap.explainers import Permutation
        shap.Explainer = Permutation

# 2. Path Setup
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models"))

from utils import seed_everything
from pipeline import run_flow
from report import build_evidence_report


class StrainVisionNet(nn.Module):
    """Dual-branch CNN + MLP fusion architecture."""

    def __init__(self, tabular_dim: int = 5):
        super().__init__()

        # Spatial CNN Branch
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        # Tabular MLP Branch
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )

        # Fusion Head
        self.fusion = nn.Sequential(
            nn.Linear(1024 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1),
        )

    def forward(self, spatial_x: torch.Tensor, tabular_x: torch.Tensor) -> torch.Tensor:
        if spatial_x.ndim == 3:
            spatial_x = spatial_x.unsqueeze(1)

        spatial_features = self.spatial_branch(spatial_x)
        tabular_features = self.tabular_branch(tabular_x)

        combined_features = torch.cat([spatial_features, tabular_features], dim=1)
        logits = self.fusion(combined_features)

        return logits


def extract_dict_key(data_dict: dict, candidate_keys: list[str]):
    """Utility to extract data from a dictionary using flexible key variations."""
    for key in candidate_keys:
        if key in data_dict:
            return data_dict[key]
    return None


class TensorDatasetWrapper(torch.utils.data.Dataset):
    """Wraps dictionary, tuple, or PyTorch dataset structures loaded from .pt files."""

    def __init__(self, data, vendors=None):
        self.vendors = vendors

        if isinstance(data, dict):
            self.images = extract_dict_key(
                data,
                ["images", "spatial", "spatial_x", "matrix", "X_img", "image", "x_spatial", "inputs"],
            )
            self.tabular = extract_dict_key(
                data,
                ["tabular", "tabular_x", "tab", "X_tab", "x_tabular", "features"],
            )
            self.labels = extract_dict_key(
                data,
                ["labels", "label", "targets", "target", "y", "ground_truth"],
            )

            if self.images is None and len(data) >= 3:
                vals = list(data.values())
                self.images, self.tabular, self.labels = vals[0], vals[1], vals[2]

            if self.vendors is None:
                self.vendors = extract_dict_key(data, ["vendors", "vendor", "vendor_metadata", "names"])

        elif isinstance(data, (list, tuple)):
            self.images = data[0]
            self.tabular = data[1]
            self.labels = data[2]
            if self.vendors is None and len(data) > 3:
                self.vendors = data[3]

        elif hasattr(data, "tensors"):
            tensors = data.tensors
            self.images = tensors[0]
            self.tabular = tensors[1]
            self.labels = tensors[2]
            if self.vendors is None and len(tensors) > 3:
                self.vendors = tensors[3]

        else:
            self.dataset = data

    def __len__(self) -> int:
        if hasattr(self, "images") and self.images is not None:
            return len(self.images)
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if hasattr(self, "images") and self.images is not None:
            return self.images[idx], self.tabular[idx], self.labels[idx]
        return self.dataset[idx]


def parse_vendor_object(data):
    """Unwraps string lists, numpy arrays, or dict tensors into a standard Python list."""
    if data is None:
        return None
    if isinstance(data, dict):
        for k in ["test", "vendors", "vendor", "vendor_metadata", "names", "labels"]:
            if k in data:
                return parse_vendor_object(data[k])
        return parse_vendor_object(list(data.values())[0])
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().tolist()
    if hasattr(data, "tolist"):
        return data.tolist()
    return list(data)


def load_resources():
    tensor_dir = ROOT / "tensors" if (ROOT / "tensors").exists() else ROOT / "tensor"
    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Vendor Metadata Discovery
    vendor_file = tensor_dir / "vendor_metadata.pt"
    test_vendors = None

    if vendor_file.exists():
        print(f"📦 Loaded vendor metadata from: {vendor_file.name}")
        raw_vendor_data = torch.load(str(vendor_file), map_location="cpu", weights_only=False)
        test_vendors = parse_vendor_object(raw_vendor_data)
    else:
        print("⚠️ vendor_metadata.pt not found. Attempting fallback extraction.")

    # 2. Test Dataset Discovery
    possible_test_paths = [
        tensor_dir / "final_test_dual_input.pt",
        tensor_dir / "test.pt",
        tensor_dir / "test_data.pt",
        tensor_dir / "test_dataset.pt",
        ROOT / "final_test_dual_input.pt",
        ROOT / "test.pt",
    ]

    test_path = None
    for path in possible_test_paths:
        if path.exists():
            test_path = path
            break

    if test_path is None and tensor_dir.exists():
        for file in tensor_dir.glob("*.pt"):
            if "vendor" not in file.stem.lower():
                test_path = file
                break

    if test_path is None or not test_path.exists():
        raise FileNotFoundError(f"Test tensor dataset not found in {tensor_dir} or {ROOT}.")

    print(f"📦 Loaded test dataset from: {test_path.name}")
    test_data = torch.load(str(test_path), map_location="cpu", weights_only=False)
    test_dataset = TensorDatasetWrapper(test_data, vendors=test_vendors)

    if test_vendors is None:
        test_vendors = test_dataset.vendors

    first_sample = test_dataset[0]
    sample_tabular = first_sample[1]
    tabular_dim = (
        sample_tabular.shape[-1]
        if hasattr(sample_tabular, "shape")
        else len(sample_tabular)
    )

    # 3. Model & Weights Loading
    model = None
    py_path = models_dir / "best_architec.py"

    if py_path.exists():
        try:
            import best_architec

            model_cls = getattr(
                best_architec,
                "StrainVisionNet",
                getattr(best_architec, "DualInputModel", None),
            )
            if model_cls is not None:
                model = model_cls(tabular_dim=tabular_dim)
        except Exception as e:
            print(f"⚠️ Could not instantiate model from best_architec.py: {e}")

    if model is None:
        model = StrainVisionNet(tabular_dim=tabular_dim)

    possible_ckpt_paths = [
        models_dir / "best_model.pt",
        models_dir / "best_model.pth",
        models_dir / "model.pt",
        models_dir / "checkpoint.pt",
        ROOT / "best_model.pt",
        ROOT / "best_model.pth",
        ROOT / "checkpoints" / "best_model.pt",
    ]

    ckpt_path = None
    for path in possible_ckpt_paths:
        if path.exists():
            ckpt_path = path
            break

    if ckpt_path is None:
        search_dirs = [models_dir, ROOT, ROOT / "checkpoints"]
        for s_dir in search_dirs:
            if s_dir.exists():
                for file in list(s_dir.glob("*.pt")) + list(s_dir.glob("*.pth")):
                    if "test" not in file.stem.lower() and "vendor" not in file.stem.lower():
                        ckpt_path = file
                        break
            if ckpt_path:
                break

    if ckpt_path is None or not ckpt_path.exists():
        raise RuntimeError(f"Model weights file missing in {models_dir} or {ROOT}.")

    print(f"📦 Loaded model weights from: {ckpt_path.name}")
    loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(loaded, torch.nn.Module):
        model = loaded
    elif isinstance(loaded, dict):
        state_dict = loaded.get("model_state_dict", loaded.get("state_dict", loaded))
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, test_dataset, test_vendors


def main() -> None:
    seed_everything(42)
    print("🚀 Initializing model evaluation and explainability flow...")

    model, test_dataset, test_vendors = load_resources()

    manifest = run_flow(
        model=model,
        test_dataset=test_dataset,
        test_vendor=test_vendors,
        out_dir="reports",
    )

    build_evidence_report(out_dir="reports", project="Nexlyra")
    print("📊 Audit report successfully generated at 'reports/AI_Evidence_Report.md'")


if __name__ == "__main__":
    main()