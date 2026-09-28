from __future__ import annotations


def build_recommendation(baseline_prediction: float, scenario_prediction: float, difference: float, risk_range: tuple[float, float], threshold: float) -> str:
    p10_risk, p90_risk = risk_range
    if scenario_prediction < threshold:
        if difference < 0:
            return f"⚠️ High Risk: The what-if scenario lowers cash flow to ₹{scenario_prediction:,.0f}, breaching your safe threshold (₹{threshold:,.0f}). Immediate action: defer non-critical expenditures, expedite receivables collection, and prepare credit line buffers."
        return f"⚠️ Below Safe Threshold: Projected scenario cash flow (₹{scenario_prediction:,.0f}) is below your safety threshold (₹{threshold:,.0f}). Consider trimming operating expenses and closely tracking client invoicing."

    if p10_risk < threshold:
        return f"⚡ Downside Volatility Warning: Expected cash flow (₹{scenario_prediction:,.0f}) meets your target, but Monte Carlo simulation indicates a 10% probability of dropping below ₹{p10_risk:,.0f} (under your ₹{threshold:,.0f} threshold). Maintain a safety reserve."

    if difference < 0:
        return f"ℹ️ Moderate Position: The what-if scenario contracts cash flow by ₹{abs(difference):,.0f}, but remains above the safety threshold. Monitor receivables closely to prevent unexpected bottlenecks."

    return f"✅ Strong Financial Health: Projected cash flow (₹{scenario_prediction:,.0f}) is sound and safely exceeds your threshold (₹{threshold:,.0f}). Continue planned operations and strategic investments."
