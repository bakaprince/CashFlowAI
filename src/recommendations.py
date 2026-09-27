from __future__ import annotations


def build_recommendation(baseline_prediction: float, scenario_prediction: float, difference: float, risk_range: tuple[float, float], threshold: float) -> str:
    if scenario_prediction < threshold:
        if difference < 0:
            return "The what-if scenario increases cash-flow risk. Review spending, speed up collections, and protect a working-capital buffer."
        return "The scenario remains below the safe threshold. Consider reducing discretionary costs and monitoring receivables closely."

    if difference < 0:
        return "The scenario is still acceptable, but it weakens the cash position. Watch expenses and collections to avoid a riskier period."

    return "The projected cash position is stable. Continue current operations and maintain prudent expense controls."
