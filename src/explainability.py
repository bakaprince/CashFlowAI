from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X_train: pd.DataFrame, X_sample: pd.DataFrame):
    try:
        if len(X_train) < 2:
            return None
        explainer = shap.Explainer(model, X_train)
        return explainer(X_sample)
    except Exception:
        return None
