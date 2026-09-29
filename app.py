from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_preprocessing import (
    clean_and_prepare_dataset,
    derive_cash_flow_target,
    normalize_column_name,
)
from src.explainability import compute_shap_values
from src.monte_carlo import run_monte_carlo
from src.recommendations import build_recommendation
from src.scenario_analysis import apply_scenario, compare_baseline_vs_scenario

CHART_KWARGS = {"width": "stretch"}

st.set_page_config(page_title="CashFlowAI", page_icon="💰", layout="wide")


def format_currency(val: float, symbol: str = "$") -> str:
    if abs(val) >= 1e9:
        return f"{symbol}{val / 1e9:,.2f}B"
    if abs(val) >= 1e6:
        return f"{symbol}{val / 1e6:,.2f}M"
    return f"{symbol}{val:,.0f}"


def prepare_model_data(df: pd.DataFrame):
    cleaned = df.copy()
    if "Cash Flow" not in cleaned.columns:
        cleaned["Cash Flow"] = derive_cash_flow_target(cleaned)

    feature_cols = [
        col for col in ["Sales", "Expenses", "Receivables", "Payables", "Cash Inflow", "Cash Outflow"]
        if col in cleaned.columns
    ]
    if not feature_cols:
        raise ValueError("No valid financial feature columns found in dataset (expected Sales, Expenses, Receivables, Payables, Cash Inflow, or Cash Outflow).")
    if len(cleaned) < 5:
        raise ValueError(f"The dataset must contain at least 5 rows for model training and forecasting (found {len(cleaned)}).")

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

# ==========================================
# Sidebar: Raw Dataset Ingestion
# ==========================================
with st.sidebar:
    st.header("📂 Upload Raw Dataset")
    uploaded_file = st.file_uploader(
        "Upload Raw CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload your unprocessed financial records, accounting export, or operational transactions."
    )
    curr_sym = st.selectbox("Currency Unit", ["$", "₹", "€", "£"], index=0)

if uploaded_file is None:
    st.info("📂 **Upload a raw CSV or Excel dataset to begin.**")
    st.markdown("""
    Please upload your raw operational or cash flow dataset using the sidebar to start the analysis.

    ### 🔄 Processing Flow:
    1. **Upload Raw Dataset**: Provide an unprocessed `.csv` or `.xlsx` file.
    2. **Raw Inspection (Step 1)**: Review the original, unmodified dataset as ingested.
    3. **Automated Preprocessing**: CashFlowAI automatically normalizes column aliases, strips currency symbols and commas, imputes missing values, and derives cash flow targets.
    4. **Cleaned Inspection (Step 2)**: Visually verify the transformed dataset before model ingestion.
    5. **AI Forecasting & Analytics**: Explore ML forecasts, SHAP feature importance, what-if simulations, Monte Carlo risk distributions, and automated strategic recommendations.
    """)
    st.stop()
    import sys
    sys.exit(0)

# ==========================================
# Step 1: Read Raw Dataset
# ==========================================
try:
    if uploaded_file.name.lower().endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)
except Exception as read_err:
    st.error(f"❌ Failed to read uploaded file: {read_err}")
    st.stop()

if raw_df is None or raw_df.empty:
    st.error("❌ The uploaded dataset is empty. Please upload a file with financial records.")
    st.stop()

# ==========================================
# Step 2: Run Existing Preprocessing Pipeline
# ==========================================
try:
    cols_lower = [str(c).lower() for c in raw_df.columns]
    is_cashflow_ready = any(k in cols_lower for k in ["cash flow", "cash_flow", "inflow", "sales", "revenue", "turnover"])
    is_transactional = not is_cashflow_ready and any(k in cols_lower for k in ["order", "invoice", "transaction"])

    if is_transactional:
        from src.data_import import preprocess_transactional_dataset
        processed_df = preprocess_transactional_dataset(raw_df.copy(), freq="M")
    else:
        processed_df = clean_and_prepare_dataset(raw_df.copy())
        if "Cash Flow" not in processed_df.columns:
            processed_df["Cash Flow"] = derive_cash_flow_target(processed_df)

except Exception as prep_err:
    st.error(f"❌ Error during data preprocessing: {prep_err}")
    st.stop()

# ==========================================
# Step 3: Run Model Training on PROCESSED Data
# ==========================================
try:
    cleaned_df, feature_cols, model, X_train, X_test, y_train, y_test, y_pred, metrics = prepare_model_data(processed_df)
except Exception as exc:
    st.error(f"❌ The processed dataset could not be prepared for forecasting: {exc}")
    st.stop()

if "Cash Flow" not in cleaned_df.columns:
    cleaned_df["Cash Flow"] = derive_cash_flow_target(cleaned_df)

latest_row = cleaned_df.iloc[-1].copy()
latest_sample = cleaned_df[feature_cols].iloc[[-1]]
base_prediction = float(model.predict(latest_sample)[0])

# Dynamic threshold range calculation
slider_min = 0 if base_prediction >= 0 else int(base_prediction * 1.5)
slider_max = max(100000, int(abs(base_prediction) * 1.5))
if slider_min >= slider_max:
    slider_max = slider_min + 10000
default_val = int(base_prediction * 0.7) if base_prediction > 0 else int(slider_min + (slider_max - slider_min) * 0.5)
slider_val = max(slider_min, min(default_val, slider_max))

# Dynamic step size based on magnitude
dynamic_step = max(1000, int(10 ** max(2, int(np.log10(max(1.0, float(slider_max - slider_min)))) - 2)))

with st.sidebar:
    st.markdown("---")
    st.header("⚙️ Risk Parameters")
    risk_threshold = st.slider(
        f"Safe cash threshold ({curr_sym})",
        min_value=slider_min,
        max_value=slider_max,
        value=slider_val,
        step=dynamic_step,
    )

residuals = y_test - y_pred
sigma = float(np.std(residuals)) if len(residuals) > 1 else 1.0
monte_carlo = run_monte_carlo(base_prediction, sigma, threshold=risk_threshold, n_samples=2000)

shap_values = compute_shap_values(model, X_train, latest_sample)

# Scenario analysis setup
scenario_prediction = base_prediction
difference = 0.0

# ==========================================
# Main Dashboard KPI Header
# ==========================================
with st.container():
    st.caption(f"📁 **Active Upload:** `{uploaded_file.name}` ({len(raw_df)} raw records ➔ {len(cleaned_df)} model-ready observations)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Cash Flow", format_currency(float(latest_row["Cash Flow"]), curr_sym))
    col2.metric("Predicted Cash Flow", format_currency(base_prediction, curr_sym))
    col3.metric("MAE", format_currency(metrics["MAE"], curr_sym))
    col4.metric("R² Score", f"{metrics['R2']:.3f}")

tabs = st.tabs(["📊 Data & Preprocessing", "📈 Forecast", "🔍 Explainability", "🔄 What-If", "🎲 Risk Simulation", "💡 Recommendation"])

# ==========================================
# Tab 0: Data & Preprocessing
# ==========================================
with tabs[0]:
    st.subheader("Data Engineering & Preprocessing Pipeline")

    # ----------------------------------------------------
    # Step 1: Raw / Unprocessed Data
    # ----------------------------------------------------
    st.markdown("### 📦 Step 1: Raw / Unprocessed Data")
    st.markdown("This is the dataset exactly as uploaded by the user, before CashFlowAI performs any cleaning or transformation.")

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Raw Rows", len(raw_df))
    col_r2.metric("Raw Columns", len(raw_df.columns))
    col_r3.metric("Raw Missing Cells", int(raw_df.isna().sum().sum()))
    col_r4.metric("File Type", uploaded_file.name.split(".")[-1].upper())

    with st.expander("🔍 Inspect Raw Column Headers", expanded=False):
        st.write(list(raw_df.columns))

    st.dataframe(raw_df, **CHART_KWARGS)

    st.markdown("---")

    # ----------------------------------------------------
    # Preprocessing Transformations Summary
    # ----------------------------------------------------
    st.markdown("### ⚙️ Preprocessing Transformations Applied")

    normalized_mappings = {}
    for orig_col in raw_df.columns:
        norm = normalize_column_name(orig_col)
        if norm != orig_col and norm in processed_df.columns:
            normalized_mappings[orig_col] = norm

    tr_col1, tr_col2, tr_col3 = st.columns(3)
    with tr_col1:
        st.markdown("**🔤 Column Normalization**")
        if normalized_mappings:
            for orig, target in normalized_mappings.items():
                st.caption(f"• `{orig}` ➔ **`{target}`**")
        else:
            st.caption("• All headers already canonical")

    with tr_col2:
        st.markdown("**🧹 Data Sanitization & Cleaning**")
        raw_missing = int(raw_df.isna().sum().sum())
        proc_missing = int(processed_df.isna().sum().sum())
        st.caption("• Stripped currency symbols (`$`, commas)")
        st.caption(f"• Missing values imputed: {raw_missing} ➔ {proc_missing}")
        st.caption("• Date ISO temporal alignment & chronological sorting")

    with tr_col3:
        st.markdown("**📐 Target Derivation**")
        if "Cash Flow" in raw_df.columns:
            st.caption("• `Cash Flow` provided in upload")
        else:
            st.caption("• `Cash Flow` derived automatically")

    st.markdown("---")

    # ----------------------------------------------------
    # Step 2: Processed / Cleaned Data
    # ----------------------------------------------------
    st.markdown("### 🧹 Step 2: Processed / Cleaned Data")
    st.markdown("This is the same dataset after CashFlowAI's preprocessing pipeline has cleaned, standardized and prepared it for machine learning.")

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.metric("Processed Rows", len(processed_df))
    col_p2.metric("Processed Columns", len(processed_df.columns))
    if "Date" in processed_df.columns:
        col_p3.metric("Start Date", str(processed_df["Date"].iloc[0])[:10])
        col_p4.metric("End Date", str(processed_df["Date"].iloc[-1])[:10])
    else:
        col_p3.metric("Missing Values", 0)
        col_p4.metric("Status", "Standardized")

    st.dataframe(processed_df, **CHART_KWARGS)

    with st.expander("📊 Statistical Summary (Processed Data)", expanded=False):
        st.write(processed_df.describe().T)

    st.markdown("---")

    # ----------------------------------------------------
    # Step 3: Model-Ready Data
    # ----------------------------------------------------
    st.markdown("### 📈 Step 3: Model-Ready Data")
    st.markdown("The machine learning models ingest the processed data below for regression forecasting:")
    m_info1, m_info2, m_info3 = st.columns(3)
    m_info1.info(f"**Target Variable ($y$):** `Cash Flow`")
    m_info2.info(f"**Feature Variables ($X$):** {', '.join(feature_cols)}")
    m_info3.info(f"**Train / Test Split:** {len(X_train)} train | {len(X_test)} test")

# ==========================================
# Tab 1: Forecast
# ==========================================
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
            "Value": [format_currency(metrics["MAE"], curr_sym), format_currency(metrics["RMSE"], curr_sym), f"{metrics['R2']:.4f}"],
        }
    )
    st.dataframe(metrics_df, **CHART_KWARGS)

# ==========================================
# Tab 2: Explainability (SHAP)
# ==========================================
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

# ==========================================
# Tab 3: What-If Scenario Analysis
# ==========================================
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
    col_w1.metric("Baseline Prediction", format_currency(base_prediction, curr_sym))
    col_w2.metric("Scenario Prediction", format_currency(scenario_prediction, curr_sym), delta=format_currency(difference, curr_sym))
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

# ==========================================
# Tab 4: Monte Carlo Risk Simulation
# ==========================================
with tabs[4]:
    st.subheader("Monte Carlo Risk Simulation")
    st.write(f"**Estimated Uncertainty (Residual Std Dev):** {format_currency(sigma, curr_sym)}")
    prob_risk = monte_carlo["probability_below_threshold"]
    thresh_formatted = format_currency(risk_threshold, curr_sym)
    if prob_risk > 0.3:
        st.warning(f"⚠️ **Probability below safe threshold ({thresh_formatted}):** {prob_risk:.2%}")
    else:
        st.info(f"✅ **Probability below safe threshold ({thresh_formatted}):** {prob_risk:.2%}")

    fig = px.histogram(
        pd.DataFrame({"Simulated Cash Flow": monte_carlo["samples"]}),
        x="Simulated Cash Flow",
        nbins=45,
        title="Monte Carlo Distribution of Projected Cash Flow Outcomes",
        color_discrete_sequence=["#6366f1"],
    )
    fig.add_vline(x=risk_threshold, line_dash="dash", line_color="#ef4444", annotation_text=f"Safe Threshold: {thresh_formatted}")
    fig.add_vline(x=monte_carlo["median"], line_dash="solid", line_color="#10b981", annotation_text=f"Median: {format_currency(monte_carlo['median'], curr_sym)}")
    st.plotly_chart(fig, **CHART_KWARGS)

    summary = pd.DataFrame({
        "Statistic": ["Mean", "Median", "P10 (Downside Risk)", "P25", "P75", "P90 (Upside Potential)"],
        "Value": [
            format_currency(monte_carlo["mean"], curr_sym),
            format_currency(monte_carlo["median"], curr_sym),
            format_currency(monte_carlo["p10"], curr_sym),
            format_currency(monte_carlo["p25"], curr_sym),
            format_currency(monte_carlo["p75"], curr_sym),
            format_currency(monte_carlo["p90"], curr_sym),
        ],
    })
    st.dataframe(summary, **CHART_KWARGS)

# ==========================================
# Tab 5: Recommendations
# ==========================================
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
