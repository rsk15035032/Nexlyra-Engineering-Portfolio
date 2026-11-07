"""Artifact Generator for Nexlyra StrainVisionNet API."""

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)

# 1. Tabular Scaler (2 features: Axial_Load_Tons, Wind_Shear_MPH)
np.random.seed(42)
dummy_tabular_data = np.random.uniform(low=0.1, high=10.0, size=(100, 2))
tabular_scaler = StandardScaler()
tabular_scaler.fit(dummy_tabular_data)
joblib.dump(tabular_scaler, artifacts_dir / "tabular_standard_scaler.pkl")

# 2. Spatial Scaler (3 features: Symmetry_V, Symmetry_H, Center_to_Edge_Ratio)
dummy_spatial_features = np.random.uniform(low=0.0, high=2.5, size=(100, 3))
spatial_scaler = StandardScaler()
spatial_scaler.fit(dummy_spatial_features)
joblib.dump(spatial_scaler, artifacts_dir / "spatial_feature_standard_scaler.pkl")

# 3. Shannon Entropy Baseline JSON
sample_matrices = np.random.uniform(0.1, 1.0, size=(100, 8, 8))
entropies = []
for mat in sample_matrices:
    prob = np.abs(mat.flatten())
    prob /= np.sum(prob)
    prob = prob[prob > 0]
    entropies.append(-np.sum(prob * np.log2(prob)))

mean_entropy = float(np.mean(entropies))
baseline_data = {
    "spatial_entropy": round(mean_entropy, 4),
    "baseline_spatial_entropy": round(mean_entropy, 4),
    "drift_threshold_pct": 20.0
}

with open(artifacts_dir / "strain_drift_baseline.json", "w") as f:
    json.dump(baseline_data, f, indent=4)

print("✓ Created tabular_standard_scaler.pkl (n_features=2)")
print("✓ Created spatial_feature_standard_scaler.pkl (n_features=3)")
print("✓ Created strain_drift_baseline.json")