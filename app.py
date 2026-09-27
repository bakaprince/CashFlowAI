import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
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

st.set_page_config(page_title="CashFlowAI", page_icon="💰", layout="wide")


@st.cache_data
def load_default_data(path: str):
    return pd.read_csv(path)


def get_dataset(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
    return load_default_data(str(SAMPLE_DATA_PATH))


def prepare_model_data(df):
    cleaned = clean_and_prepare_dataset(df)
    feature_cols = [
        col for col in ["Sales", "Expenses", "Receivables", "Payables", "Cash Inflow", "Cash Outflow"]
        if col in cleaned.columns
    ]
    if not feature_cols:
        raise ValueError("No valid feature columns were found in the dataset.")
    if "Cash Flow" not in cleaned.columns:
        cleaned["Cash Flow"] = derive_cash_flow_target(cleaned)
    X = cleaned[feature_cols]
    y = cleaned["Cash Flow"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }
    return cleaned, feature_cols, model, X_train, X_test, y_train, y_test, y_pred, metrics


st.title("AI-Powered Cash Flow Forecasting & Decision Support System")

with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])
    st.caption("If no file is uploaded, the demo dataset will be used.")

try:
    df = get_dataset(uploaded_file)
    cleaned_df, feature_cols, model, X_train, X_test, y_train, y_test, y_pred, metrics = prepare_model_data(df)
except Exception as exc:
    st.error(f"The dataset could not be processed: {exc}")
    st.stop()

if "Cash Flow" not in cleaned_df.columns:
    cleaned_df["Cash Flow"] = derive_cash_flow_target(cleaned_df)

latest_row = cleaned_df.iloc[-1].copy()
latest_features = latest_row[feature_cols]
base_prediction = float(model.predict(latest_features.to_frame().T)[0])

risk_threshold = st.sidebar.slider("Safe cash threshold", min_value=0, max_value=max(100000, int(base_prediction * 1.5)), value=max(50000, int(base_prediction * 0.7)), step=1000)

shap_values = compute_shap_values(model, X_train, latest_features.to_frame().T)

with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Cash Flow", f"₹{latest_row['Cash Flow']:,.0f}")
    col2.metric("Predicted Cash Flow", f"₹{base_prediction:,.0f}")
    col3.metric("MAE", f"₹{metrics['MAE']:,.0f}")
    col4.metric("R²", f"{metrics['R2']:.3f}")


tabs = st.tabs(["Data", "Forecast", "Explainability", "What-If", "Risk", "Recommendation"])

with tabs[0]:
    st.subheader("Dataset preview")
    st.dataframe(cleaned_df.head(20), use_container_width=True)
    st.subheader("Dataset information")
    st.write(cleaned_df.describe(include="all").T)

with tabs[1]:
    st.subheader("Historical cash flow vs. model prediction")
    history = cleaned_df.copy()
    history["Predicted Cash Flow"] = np.nan
    history.loc[history.index[-len(y_test):], "Predicted Cash Flow"] = y_pred

    fig = px.line(
        history,
        x="Date" if "Date" in history.columns else history.index,
        y=["Cash Flow", "Predicted Cash Flow"],
        title="Cash Flow Forecast",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model performance")
    metrics_df = pd.DataFrame(
        {
            "Metric": ["MAE", "RMSE", "R²"],
            "Value": [metrics["MAE"], metrics["RMSE"], metrics["R2"]],
        }
    )
    st.dataframe(metrics_df, use_container_width=True)

with tabs[2]:
    st.subheader("SHAP feature contribution")
    if shap_values is not None:
        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "Absolute Contribution": np.abs(shap_values.values[0]),
            "Contribution": shap_values.values[0],
        }).sort_values("Absolute Contribution", ascending=False)
        st.dataframe(shap_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap_df.plot(kind="bar", x="Feature", y="Absolute Contribution", ax=ax, color="steelblue")
        ax.set_title("SHAP feature importance")
        ax.set_ylabel("Mean absolute SHAP value")
        st.pyplot(fig)
    else:
        st.info("SHAP explanation is unavailable for this dataset.")

with tabs[3]:
    st.subheader("What-if scenario analysis")
    change_feature = st.selectbox("Select the variable to change", feature_cols)
    percent_change = st.slider("Percentage change", min_value=-50, max_value=50, value=-20, step=5)

    scenario_data = apply_scenario(cleaned_df, change_feature, percent_change)
    scenario_row = scenario_data[feature_cols].iloc[-1]
    scenario_prediction = float(model.predict(pd.DataFrame([scenario_row], columns=feature_cols))[0])
    difference = scenario_prediction - base_prediction

    baseline_label, scenario_label = compare_baseline_vs_scenario(base_prediction, scenario_prediction)
    st.write(f"Baseline prediction: ₹{base_prediction:,.0f}")
    st.write(f"Scenario prediction: ₹{scenario_prediction:,.0f}")
    st.write(f"Difference: ₹{difference:,.0f}")

    chart_df = pd.DataFrame({
        "Scenario": ["Baseline", "What-If"],
        "Prediction": [base_prediction, scenario_prediction],
    })
    fig = px.bar(chart_df, x="Scenario", y="Prediction", color="Scenario", title="Baseline vs What-If prediction")
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Monte Carlo risk simulation")
    residuals = y_test - y_pred
    sigma = float(np.std(residuals)) if len(residuals) > 1 else 1.0
    monte_carlo = run_monte_carlo(base_prediction, sigma, n_samples=2000)

    st.write(f"Estimated uncertainty (standard deviation): ₹{sigma:,.0f}")
    st.write(f"Probability below threshold: {monte_carlo['probability_below_threshold']:.2%}")

    fig = px.histogram(
        pd.DataFrame({"Simulated Cash Flow": monte_carlo["samples"]}),
        x="Simulated Cash Flow",
        nbins=40,
        title="Monte Carlo distribution of possible cash flow outcomes",
    )
    fig.add_vline(x=monte_carlo["median"], line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    summary = pd.DataFrame({
        "Statistic": ["Mean", "Median", "P10", "P25", "P75", "P90"],
        "Value": [
            monte_carlo["mean"],
            monte_carlo["median"],
            monte_carlo["p10"],
            monte_carlo["p25"],
            monte_carlo["p75"],
            monte_carlo["p90"],
        ],
    })
    st.dataframe(summary, use_container_width=True)

with tabs[5]:
    st.subheader("Recommendation")
    recommendation = build_recommendation(
        baseline_prediction=base_prediction,
        scenario_prediction=scenario_prediction,
        difference=difference,
        risk_range=(monte_carlo["p10"], monte_carlo["p90"]),
        threshold=risk_threshold,
    )
    st.success(recommendation)

st.caption("This dashboard follows the project specification: Forecast -> Explain -> What-If -> Simulate Risk -> Recommend.")
