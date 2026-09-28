import os
from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import shap
# pyrefly: ignore [missing-import]
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from src.data_preprocessing import clean_and_prepare_dataset, derive_cash_flow_target
from src.explainability import compute_shap_values
from src.scenario_analysis import apply_scenario, compare_baseline_vs_scenario
from src.monte_carlo import run_monte_carlo
from src.recommendations import build_recommendation

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_cashflow_data.csv"
CHART_KWARGS = {"width": "stretch"}

st.set_page_config(page_title="CashFlowAI", page_icon="💰", layout="wide")


@st.cache_data
def load_default_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
    return load_default_data(str(SAMPLE_DATA_PATH))


def prepare_model_data(df: pd.DataFrame):
    cleaned = clean_and_prepare_dataset(df)
    feature_cols = [
        col for col in ["Sales", "Expenses", "Receivables", "Payables", "Cash Inflow", "Cash Outflow"]
        if col in cleaned.columns
    ]
    if not feature_cols:
        raise ValueError("No valid financial feature columns found in dataset (expected Sales, Expenses, Receivables, Payables, Cash Inflow, or Cash Outflow).")
    if len(cleaned) < 5:
        raise ValueError("The dataset must contain at least 5 rows for model training and forecasting.")
    if "Cash Flow" not in cleaned.columns:
        cleaned["Cash Flow"] = derive_cash_flow_target(cleaned)

    X = cleaned[feature_cols]
    y = cleaned["Cash Flow"]
    test_size = max(2, int(len(cleaned) * 0.2)) if len(cleaned) >= 10 else 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else 1.0,
    }
    return cleaned, feature_cols, model, X_train, X_test, y_train, y_test, y_pred, metrics


st.title("💰 AI-Powered Cash Flow Forecasting & Decision Support System")

with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    st.caption("If no file is uploaded, the default sample dataset will be used.")

try:
    df = get_dataset(uploaded_file)
    cleaned_df, feature_cols, model, X_train, X_test, y_train, y_test, y_pred, metrics = prepare_model_data(df)
except Exception as exc:
    st.error(f"The dataset could not be processed: {exc}")
    st.stop()

if "Cash Flow" not in cleaned_df.columns:
    cleaned_df["Cash Flow"] = derive_cash_flow_target(cleaned_df)

latest_row = cleaned_df.iloc[-1].copy()
latest_sample = cleaned_df[feature_cols].iloc[[-1]]
base_prediction = float(model.predict(latest_sample)[0])

slider_min = 0 if base_prediction >= 0 else int(base_prediction * 1.5)
slider_max = max(100000, int(abs(base_prediction) * 1.5))
if slider_min >= slider_max:
    slider_max = slider_min + 10000
default_val = int(base_prediction * 0.7) if base_prediction > 0 else int(slider_min + (slider_max - slider_min) * 0.5)
slider_val = max(slider_min, min(default_val, slider_max))

risk_threshold = st.sidebar.slider(
    "Safe cash threshold (₹)",
    min_value=slider_min,
    max_value=slider_max,
    value=slider_val,
    step=1000,
)

residuals = y_test - y_pred
sigma = float(np.std(residuals)) if len(residuals) > 1 else 1.0
monte_carlo = run_monte_carlo(base_prediction, sigma, threshold=risk_threshold, n_samples=2000)

shap_values = compute_shap_values(model, X_train, latest_sample)

# Initialize scenario variables
scenario_prediction = base_prediction
difference = 0.0

with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Cash Flow", f"₹{latest_row['Cash Flow']:,.0f}")
    col2.metric("Predicted Cash Flow", f"₹{base_prediction:,.0f}")
    col3.metric("MAE", f"₹{metrics['MAE']:,.0f}")
    col4.metric("R²", f"{metrics['R2']:.3f}")

tabs = st.tabs(["📊 Data", "📈 Forecast", "🔍 Explainability", "🔄 What-If", "🎲 Risk Simulation", "💡 Recommendation"])

with tabs[0]:
    st.subheader("Dataset Preview")
    st.dataframe(cleaned_df.head(20), **CHART_KWARGS)
    st.subheader("Dataset Summary Statistics")
    st.write(cleaned_df.describe(include="all").T)

with tabs[1]:
    st.subheader("Historical Cash Flow vs Model Forecast")
    history = cleaned_df.copy()
    history["Predicted Cash Flow"] = np.nan
    history.loc[history.index[-len(y_test):], "Predicted Cash Flow"] = y_pred

    fig = px.line(
        history,
        x="Date" if "Date" in history.columns else history.index,
        y=["Cash Flow", "Predicted Cash Flow"],
        title="Cash Flow Forecast Comparison",
        markers=True,
    )
    st.plotly_chart(fig, **CHART_KWARGS)

    st.subheader("Model Performance Metrics")
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "R² Score"],
            "Value": [f"₹{metrics['MAE']:,.2f}", f"₹{metrics['RMSE']:,.2f}", f"{metrics['R2']:.4f}"],
        }
    )
    st.dataframe(metrics_df, **CHART_KWARGS)

with tabs[2]:
    st.subheader("SHAP Feature Attribution (Explainability)")
    if shap_values is not None and hasattr(shap_values, "values") and len(shap_values.values) > 0:
        raw_vals = np.array(shap_values.values)
        if raw_vals.ndim == 2:
            vals = raw_vals[0]
        elif raw_vals.ndim > 2:
            vals = raw_vals[0].flatten()
        else:
            vals = raw_vals

        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "Absolute Contribution": np.abs(vals),
            "Contribution": vals,
        }).sort_values("Absolute Contribution", ascending=False)
        st.dataframe(shap_df, **CHART_KWARGS)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        shap_df.plot(kind="bar", x="Feature", y="Absolute Contribution", ax=ax, color="#2563eb", legend=False)
        ax.set_title("SHAP Feature Importance (Absolute Impact)", fontsize=13, pad=12)
        ax.set_ylabel("Mean Absolute SHAP Value", fontsize=11)
        ax.set_xlabel("Financial Metric", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("SHAP explanation is unavailable for this dataset.")

with tabs[3]:
    st.subheader("What-If Scenario Analysis")
    change_feature = st.selectbox("Select variable to simulate", feature_cols)
    percent_change = st.slider("Simulated change (%)", min_value=-50, max_value=50, value=-20, step=5)

    scenario_data = apply_scenario(cleaned_df, change_feature, percent_change)
    scenario_sample = scenario_data[feature_cols].iloc[[-1]]
    scenario_prediction = float(model.predict(scenario_sample)[0])
    difference = scenario_prediction - base_prediction

    baseline_label, scenario_label = compare_baseline_vs_scenario(base_prediction, scenario_prediction)

    col_w1, col_w2, col_w3 = st.columns(3)
    col_w1.metric("Baseline Prediction", f"₹{base_prediction:,.0f}")
    col_w2.metric("Scenario Prediction", f"₹{scenario_prediction:,.0f}", delta=f"₹{difference:,.0f}")
    col_w3.metric("Assessment", scenario_label)

    chart_df = pd.DataFrame({
        "Scenario": ["Baseline", f"What-If ({change_feature} {percent_change:+d}%)"],
        "Prediction": [base_prediction, scenario_prediction],
    })
    fig = px.bar(
        chart_df,
        x="Scenario",
        y="Prediction",
        color="Scenario",
        color_discrete_sequence=["#3b82f6", "#10b981" if difference >= 0 else "#ef4444"],
        title="Baseline vs What-If Prediction Comparison",
    )
    st.plotly_chart(fig, **CHART_KWARGS)

with tabs[4]:
    st.subheader("Monte Carlo Risk Simulation")
    st.write(f"**Estimated Uncertainty (Residual Std Dev):** ₹{sigma:,.0f}")
    prob_risk = monte_carlo['probability_below_threshold']
    if prob_risk > 0.3:
        st.warning(f"⚠️ **Probability below safe threshold (₹{risk_threshold:,.0f}):** {prob_risk:.2%}")
    else:
        st.info(f"✅ **Probability below safe threshold (₹{risk_threshold:,.0f}):** {prob_risk:.2%}")

    fig = px.histogram(
        pd.DataFrame({"Simulated Cash Flow": monte_carlo["samples"]}),
        x="Simulated Cash Flow",
        nbins=45,
        title="Monte Carlo Distribution of Projected Cash Flow Outcomes",
        color_discrete_sequence=["#6366f1"],
    )
    fig.add_vline(x=risk_threshold, line_dash="dash", line_color="#ef4444", annotation_text=f"Safe Threshold: ₹{risk_threshold:,.0f}")
    fig.add_vline(x=monte_carlo["median"], line_dash="solid", line_color="#10b981", annotation_text=f"Median: ₹{monte_carlo['median']:,.0f}")
    st.plotly_chart(fig, **CHART_KWARGS)

    summary = pd.DataFrame({
        "Statistic": ["Mean", "Median", "P10 (Downside Risk)", "P25", "P75", "P90 (Upside Potential)"],
        "Value": [
            f"₹{monte_carlo['mean']:,.0f}",
            f"₹{monte_carlo['median']:,.0f}",
            f"₹{monte_carlo['p10']:,.0f}",
            f"₹{monte_carlo['p25']:,.0f}",
            f"₹{monte_carlo['p75']:,.0f}",
            f"₹{monte_carlo['p90']:,.0f}",
        ],
    })
    st.dataframe(summary, **CHART_KWARGS)

with tabs[5]:
    st.subheader("Strategic Recommendations & Decision Support")
    recommendation = build_recommendation(
        baseline_prediction=base_prediction,
        scenario_prediction=scenario_prediction,
        difference=difference,
        risk_range=(monte_carlo["p10"], monte_carlo["p90"]),
        threshold=risk_threshold,
    )
    st.info(recommendation)

st.caption("AI-Powered Cash Flow System: Forecast → Explain → What-If → Risk Simulation → Strategic Recommendation")
