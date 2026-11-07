# Nexlyra AI Evidence Report

Generated: 2026-09-13T15:13:07.881545+00:00

## Scope
Explainability and model-audit evidence for the 8×8 strain-matrix micro-fracture classifier.

## Evidence
- Verified true-positive index: 8
- Titan Alloy spoof index: 220
- Grad-CAM target layer: spatial_branch.3
- First Conv2d: spatial_branch.0

## SHAP
- Wind_Shear_MPH: mean |SHAP|=0.005107, mean SHAP=0.001207
- Axial_Load_Tons: mean |SHAP|=0.004335, mean SHAP=0.001776
- Vessel_Age_Years: mean |SHAP|=0.001394, mean SHAP=0.000311
- Ambient_Temp_C: mean |SHAP|=0.000474, mean SHAP=0.000049
- Vibration_RMS: mean |SHAP|=0.000209, mean SHAP=0.000001

## Scientific Interpretation
Grad-CAM identifies spatial regions used by the CNN; SHAP quantifies tabular contribution. Neither method alone proves causality.
The claim that perfect spatial symmetry under high wind forces a negative prediction must be supported by measured attribution and controlled ablation results, not assumed from the task statement.

## Production Controls
- Fixed seed and deterministic evaluation.
- Exact Day-4 preprocessing must be reused.
- SVG evidence artifacts are retained for auditability.
- Audit JSON embeds the Grad-CAM image.
- ONNX export must be verified against PyTorch before deployment.