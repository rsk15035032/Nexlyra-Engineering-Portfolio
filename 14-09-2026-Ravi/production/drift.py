"""Spatial Data Drift Detection Module for StrainVisionNet."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger("drift_detector")


class SpatialDriftDetector:
    """Calculates spatial statistical entropy and evaluates domain drift."""

    def __init__(self, baseline_path: Path | str) -> None:
        self.baseline_path = Path(baseline_path)
        self.baseline_entropy = self._load_baseline()

    def _load_baseline(self) -> float:
        if self.baseline_path.exists():
            try:
                with open(self.baseline_path, "r") as f:
                    data = json.load(f)
                entropy = (
                    data.get("spatial_entropy")
                    or data.get("baseline_spatial_entropy")
                )
                if entropy is not None:
                    return float(entropy)
            except Exception as err:
                logger.warning("Failed to load baseline JSON: %s", err)
        return 4.5

    @staticmethod
    def calculate_entropy(matrix: np.ndarray) -> float:
        """Calculates Shannon entropy across an 8x8 strain tensor."""
        prob = np.abs(matrix.flatten())
        total = float(np.sum(prob))
        if total == 0.0:
            return 0.0
        prob = prob / total
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob)))

    def check_drift(self, matrix: np.ndarray) -> tuple[bool, float]:
        """Evaluates whether current matrix entropy deviates >20% from baseline."""
        current_entropy = self.calculate_entropy(matrix)
        if self.baseline_entropy == 0:
            return False, current_entropy

        delta = abs(current_entropy - self.baseline_entropy) / self.baseline_entropy
        drift_alert = bool(delta > 0.20)
        return drift_alert, current_entropy