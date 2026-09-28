from __future__ import annotations

# pyrefly: ignore [missing-import]
import numpy as np


def run_monte_carlo(base_prediction: float, sigma: float, threshold: float = 0.0, n_samples: int = 2000):
    rng = np.random.default_rng(42)
    effective_sigma = max(float(sigma), 1.0)
    samples = rng.normal(base_prediction, effective_sigma, n_samples)
    return {
        "samples": samples,
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "p10": float(np.percentile(samples, 10)),
        "p25": float(np.percentile(samples, 25)),
        "p75": float(np.percentile(samples, 75)),
        "p90": float(np.percentile(samples, 90)),
        "probability_below_threshold": float(np.mean(samples < threshold)),
    }
