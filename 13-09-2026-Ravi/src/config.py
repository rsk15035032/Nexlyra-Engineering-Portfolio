from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Config:
    seed: int = 42
    vendor_name: str = "Titan_Alloy_Syndicate"
    n_shap: int = 100
    tabular_features: tuple[str, ...] = (
        "Axial_Load_Tons",
        "Wind_Shear_MPH",
        "Ambient_Temp_C",
        "Vibration_RMS",
        "Vessel_Age_Years",
    )

CFG = Config()