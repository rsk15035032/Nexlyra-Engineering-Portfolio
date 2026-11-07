from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

def build_evidence_report(
    out_dir: str | Path = "day5_artifacts", project: str = "Nexlyra"
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}

    lines = [
        f"# {project} AI Evidence Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "Explainability and model-audit evidence for the 8×8 strain-matrix "
        "micro-fracture classifier.",
        "",
        "## Evidence",
        f"- Verified true-positive index: {manifest.get('true_positive_index', 'N/A')}",
        f"- Titan Alloy spoof index: {manifest.get('titan_spoof_index', 'N/A')}",
        f"- Grad-CAM target layer: {manifest.get('gradcam_target_layer', 'N/A')}",
        f"- First Conv2d: {manifest.get('first_conv2d', 'N/A')}",
        "",
        "## SHAP",
    ]

    shap_features = manifest.get("shap_features", [])
    if isinstance(shap_features, list) and shap_features:
        for row in shap_features:
            feature_name = row.get("feature", "Unknown")
            mean_abs = float(row.get("mean_abs_shap", 0.0))
            mean_val = float(row.get("mean_shap", 0.0))
            lines.append(
                f"- {feature_name}: mean |SHAP|={mean_abs:.6f}, mean SHAP={mean_val:.6f}"
            )
    else:
        lines.append("- No SHAP features recorded in manifest.")

    lines += [
        "",
        "## Scientific Interpretation",
        "Grad-CAM identifies spatial regions used by the CNN; SHAP quantifies "
        "tabular contribution. Neither method alone proves causality.",
        "The claim that perfect spatial symmetry under high wind forces a "
        "negative prediction must be supported by measured attribution and "
        "controlled ablation results, not assumed from the task statement.",
        "",
        "## Production Controls",
        "- Fixed seed and deterministic evaluation.",
        "- Exact Day-4 preprocessing must be reused.",
        "- SVG evidence artifacts are retained for auditability.",
        "- Audit JSON embeds the Grad-CAM image.",
        "- ONNX export must be verified against PyTorch before deployment.",
    ]

    report = out / "AI_Evidence_Report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report