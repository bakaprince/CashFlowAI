from __future__ import annotations

import pandas as pd


def apply_scenario(df: pd.DataFrame, feature_name: str, percent_change: float) -> pd.DataFrame:
    scenario_df = df.copy()
    if feature_name not in scenario_df.columns:
        raise ValueError(f"Feature '{feature_name}' is not available in the dataset.")
    scenario_df[feature_name] = scenario_df[feature_name] * (1 + percent_change / 100)
    return scenario_df


def compare_baseline_vs_scenario(baseline: float, scenario: float) -> tuple[str, str]:
    if scenario > baseline:
        return "Baseline", "Improved scenario"
    if scenario < baseline:
        return "Baseline", "Riskier scenario"
    return "Baseline", "No change"
