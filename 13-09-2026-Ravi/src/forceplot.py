from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import shap  


def save_force_plot(
    shap_values,
    feature_names: list[str],
    base_value: float | np.ndarray,
    out_html: str,
) -> str:
    """Generates a SHAP force plot for a single sample and saves it to an HTML file."""
    # Extract raw numpy array from SHAP Explanation object if necessary
    if hasattr(shap_values, "values"):
        vals = np.asarray(shap_values.values).reshape(-1)
    else:
        vals = np.asarray(shap_values).reshape(-1)

    # Safely extract scalar float base value
    if hasattr(base_value, "detach"):
        base_value = base_value.detach().cpu().numpy()
    base = float(np.asarray(base_value).reshape(-1)[0])

    # Generate SHAP Force Plot HTML object
    html_obj = shap.force_plot(
        base,
        vals,
        feature_names=feature_names,
        matplotlib=False,
    )

    # Save HTML output using shap helper or native file writer fallback
    if hasattr(shap, "save_html"):
        shap.save_html(out_html, html_obj)
    else:
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(f"<html><body>{html_obj.data}</body></html>")

    return out_html