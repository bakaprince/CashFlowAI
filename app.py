from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

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
from src.data_import import (
    fetch_sec_cashflow_dataset,
    preprocess_transactional_dataset,
    download_and_preprocess_kaggle,
    SEC_CIK_MAP,
)
from src.explainability import compute_shap_values
from src.scenario_analysis import apply_scenario, compare_baseline_vs_scenario
from src.monte_carlo import run_monte_carlo
from src.recommendations import build_recommendation

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
DATA_DIR = PROCESSED_DATA_DIR  # Backward-compatible alias
UNPROCESSED_DATA_DIR = BASE_DIR / "unprocessed_data"
SAMPLE_DATA_PATH = PROCESSED_DATA_DIR / "sample_cashflow_data.csv"
CHART_KWARGS = {"width": "stretch"}

st.set_page_config(page_title="CashFlowAI", page_icon="💰", layout="wide")


@st.cache_data
def load_csv_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def format_currency(val: float, symbol: str = "$") -> str:
    if abs(val) >= 1e9:
        return f"{symbol}{val / 1e9:,.2f}B"
    if abs(val) >= 1e6:
        return f"{symbol}{val / 1e6:,.2f}M"
    return f"{symbol}{val:,.0f}"


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

# ==========================================
# Sidebar: Dataset Selection & Ingestion
# ==========================================
with st.sidebar:
    st.header("📂 Data Source")

    data_source_options = [
        "🏢 Apple Inc. (Real SEC Filings, 2017–2026)",
        "🏢 Microsoft Corp. (Real SEC Filings, 2017–2026)",
        "🏢 Amazon.com (Real SEC Filings, 2017–2026)",
        "🏢 Alphabet / Google (Real SEC Filings, 2017–2025)",
        "🏢 Tesla Inc. (Real SEC Filings, 2018–2026)",
        "📊 Sample SME Baseline (Processed 2-Year Series)",
        "📦 Raw Retail Transactions (Unprocessed Order Logs)",
        "🧹 Raw Messy SME Operations (Unprocessed Spreadsheet)",
        "📁 Upload Custom File (CSV or Excel)",
        "🌐 Kaggle & Live Data Pipeline",
    ]

    selected_source = st.selectbox("Choose Dataset Source", data_source_options, index=0)

    # Smart currency symbol default
    default_curr = "$" if any(k in selected_source for k in ["Apple", "Microsoft", "Amazon", "Alphabet", "Tesla", "Retail"]) else "₹"
    curr_sym = st.selectbox("Currency Unit", ["$", "₹", "€", "£"], index=0 if default_curr == "$" else 1)

    df_raw: Optional[pd.DataFrame] = None
    data_origin_description = ""
    raw_preview_df: Optional[pd.DataFrame] = None
    raw_preview_title: str = ""

    if "Apple" in selected_source:
        data_path = PROCESSED_DATA_DIR / "real_cashflow_aapl.csv"
        df_raw = load_csv_data(str(data_path))
        data_origin_description = "Official audited SEC EDGAR quarterly cash flow statement (Apple Inc., 2017–2026, CIK: 0000320193)."
        raw_json_path = UNPROCESSED_DATA_DIR / "raw_sec_edgar_aapl.json"
        if raw_json_path.exists():
            raw_preview_title = "Raw SEC EDGAR XBRL JSON (CIK 0000320193, US-GAAP Facts)"

    elif "Microsoft" in selected_source:
        data_path = PROCESSED_DATA_DIR / "real_cashflow_msft.csv"
        df_raw = load_csv_data(str(data_path))
        data_origin_description = "Official audited SEC EDGAR quarterly cash flow statement (Microsoft Corp., 2017–2026, CIK: 0000789019)."

    elif "Amazon" in selected_source:
        data_path = PROCESSED_DATA_DIR / "real_cashflow_amzn.csv"
        df_raw = load_csv_data(str(data_path))
        data_origin_description = "Official audited SEC EDGAR quarterly cash flow statement (Amazon.com Inc., 2017–2026, CIK: 0001018724)."

    elif "Alphabet" in selected_source:
        data_path = PROCESSED_DATA_DIR / "real_cashflow_googl.csv"
        df_raw = load_csv_data(str(data_path))
        data_origin_description = "Official audited SEC EDGAR quarterly cash flow statement (Alphabet Inc., 2017–2025, CIK: 0001652044)."

    elif "Tesla" in selected_source:
        data_path = PROCESSED_DATA_DIR / "real_cashflow_tsla.csv"
        df_raw = load_csv_data(str(data_path))
        data_origin_description = "Official audited SEC EDGAR quarterly cash flow statement (Tesla Inc., 2018–2026, CIK: 0001318605)."

    elif "Sample SME Baseline" in selected_source:
        df_raw = load_csv_data(str(SAMPLE_DATA_PATH))
        data_origin_description = "Pre-packaged synthetic SME 2-year operational cash flow dataset."

    elif "Raw Retail Transactions" in selected_source:
        raw_csv_path = UNPROCESSED_DATA_DIR / "raw_retail_transactions.csv"
        raw_unproc_df = load_csv_data(str(raw_csv_path))
        raw_preview_df = raw_unproc_df
        raw_preview_title = "Unprocessed Granular Transaction Records (Order-Level Logs)"
        df_raw = preprocess_transactional_dataset(raw_unproc_df, freq="M")
        data_origin_description = "Unprocessed order logs (244 transactions) aggregated into monthly cash flow series."

    elif "Raw Messy SME Operations" in selected_source:
        messy_path = UNPROCESSED_DATA_DIR / "raw_uncleaned_sme_cashflow.csv"
        raw_unproc_df = load_csv_data(str(messy_path))
        raw_preview_df = raw_unproc_df
        raw_preview_title = "Unprocessed Messy SME Accounting Spreadsheet (Currency strings, missing values & non-standard headers)"
        df_raw = clean_and_prepare_dataset(raw_unproc_df)
        data_origin_description = "Unprocessed messy spreadsheet cleaned via regex sanitization and median imputation."

    elif "Upload" in selected_source:
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                temp_df = pd.read_csv(uploaded_file)
            else:
                temp_df = pd.read_excel(uploaded_file)

            # Check if dataset is transaction-level (e.g. Kaggle sales/orders logs)
            cols_lower = [c.lower() for c in temp_df.columns]
            is_cashflow_ready = any(k in cols_lower for k in ["cash flow", "cash_flow", "inflow"]) and any(k in cols_lower for k in ["sales", "revenue"])

            if not is_cashflow_ready and any(k in cols_lower for k in ["date", "time", "order"]):
                st.info("💡 Transactional log detected. Choose aggregation frequency:")
                agg_freq = st.selectbox("Aggregation Frequency", ["Monthly (Recommended)", "Weekly", "Daily", "Quarterly"])
                freq_code = {"Monthly (Recommended)": "ME", "Weekly": "W", "Daily": "D", "Quarterly": "QE"}[agg_freq]
                try:
                    df_raw = preprocess_transactional_dataset(temp_df, freq=freq_code)
                    data_origin_description = f"User uploaded transactional dataset (aggregated {agg_freq.lower()})."
                    st.success("Successfully preprocessed transactional dataset into cash flow format!")
                except Exception as e:
                    st.warning(f"Transactional conversion failed: {e}. Trying standard cleaner.")
                    df_raw = temp_df
            else:
                df_raw = temp_df
                data_origin_description = "User uploaded custom dataset."
        else:
            st.info("Please upload a file or choose one of the real SEC / sample datasets above.")
            st.stop()

    elif "Kaggle" in selected_source:
        st.subheader("Kaggle & Live ETL Pipeline")
        pipeline_tab = st.radio("Source Mode", ["Kaggle Dataset", "Live SEC EDGAR Ticker"], horizontal=True)

        if pipeline_tab == "Kaggle Dataset":
            kaggle_slug = st.text_input("Kaggle Dataset Slug", value="thedevastator/superstore-sales", help="Format: owner/dataset-name")
            st.caption("Requires `~/.kaggle/kaggle.json` credentials or Kaggle API keys.")
            if st.button("🚀 Download & Preprocess from Kaggle"):
                with st.spinner("Downloading and processing Kaggle dataset..."):
                    try:
                        df_raw = download_and_preprocess_kaggle(kaggle_slug)
                        data_origin_description = f"Preprocessed Kaggle dataset ({kaggle_slug})."
                        st.success("Kaggle dataset loaded and preprocessed!")
                    except Exception as k_err:
                        st.error(f"Kaggle download failed: {k_err}")
                        st.info("Tip: You can also download the CSV manually from Kaggle and upload via 'Upload Custom File' above.")
                        st.stop()
            else:
                st.info("Enter Kaggle slug and click the button above, or select another source.")
                st.stop()

        else:
            live_ticker = st.text_input("Company Ticker", value="AAPL").upper()
            if st.button("📡 Fetch Live SEC EDGAR Statement"):
                with st.spinner(f"Fetching SEC filings for {live_ticker}..."):
                    try:
                        df_raw = fetch_sec_cashflow_dataset(live_ticker)
                        data_origin_description = f"Live SEC EDGAR financial filings for {live_ticker}."
                        st.success(f"Fetched {len(df_raw)} quarters for {live_ticker}!")
                    except Exception as s_err:
                        st.error(f"Failed to fetch live SEC data: {s_err}")
                        st.stop()
            else:
                st.info("Click button above to fetch live SEC data.")
                st.stop()

# ==========================================
# Model Training & Feature Extraction
# ==========================================
try:
    if df_raw is None or df_raw.empty:
        st.warning("No data selected. Please choose a dataset from the sidebar.")
        st.stop()

    cleaned_df, feature_cols, model, X_train, X_test, y_train, y_test, y_pred, metrics = prepare_model_data(df_raw)
except Exception as exc:
    st.error(f"The dataset could not be processed: {exc}")
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
    if data_origin_description:
        st.caption(f"📌 **Active Dataset:** {data_origin_description}")
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
    st.subheader("Dataset Overview & Operational Metrics")
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("Total Observations", len(cleaned_df))
    if "Date" in cleaned_df.columns:
        c_m2.metric("Start Date", str(cleaned_df["Date"].iloc[0])[:10])
        c_m3.metric("End Date", str(cleaned_df["Date"].iloc[-1])[:10])
    c_m4.metric("Avg Sales/Revenue", format_currency(float(cleaned_df["Sales"].mean()) if "Sales" in cleaned_df.columns else 0.0, curr_sym))

    # Raw vs Processed Data Inspector
    if raw_preview_df is not None:
        st.markdown("---")
        with st.expander(f"📦 Step 1: Raw Unprocessed Data Preview ({raw_preview_title})", expanded=True):
            st.info("💡 **Unprocessed Raw Data:** Displaying raw inputs prior to data cleansing, column normalization, or frequency aggregation:")
            st.dataframe(raw_preview_df.head(15), **CHART_KWARGS)
        st.caption("⬇️ **ETL Pipeline Applied: Temporal aggregation, currency string sanitization, and median imputation** ⬇️")

    elif "Apple" in selected_source and (UNPROCESSED_DATA_DIR / "raw_sec_edgar_aapl.json").exists():
        with st.expander("🏛️ Data Provenance: Inspect Raw SEC EDGAR XBRL Data (Apple Inc.)", expanded=False):
            st.markdown("""
            **Official Source:** U.S. Securities and Exchange Commission (SEC) EDGAR API
            - **Entity:** Apple Inc. (CIK: `0000320193`)
            - **Raw File:** `unprocessed_data/raw_sec_edgar_aapl.json`
            - **Key US-GAAP XBRL Tags Extracted:**
              - Sales: `RevenueFromContractWithCustomerExcludingAssessedTax` / `SalesRevenueNet`
              - Operating Cash Flow: `NetCashProvidedByUsedInOperatingActivities` (Un-cumulated from YTD sums)
              - Working Capital: `AccountsReceivableNetCurrent`, `AccountsPayableCurrent`, `CashAndCashEquivalentsAtCarryingValue`
            """)

    st.markdown("---")
    st.subheader("Cleaned & Preprocessed Data Preview (Model-Ready)")
    st.dataframe(cleaned_df.head(25), **CHART_KWARGS)

    st.subheader("Statistical Summary")
    st.write(cleaned_df.describe().T)

    with st.expander("🛠️ Preprocessing Pipeline & Lineage Details", expanded=False):
        st.markdown(f"""
        - **Active Dataset**: `{selected_source}`
        - **Storage Directory**: `processed_data/` (Processed) | `unprocessed_data/` (Raw Source)
        - **Column Aliases Normalized**: Canonical mapping for Sales, Expenses, Receivables, Payables, Inflow, Outflow.
        - **Sanitization**: Automatic stripping of currency glyphs, commas, and parsing ISO dates.
        - **Discrete Period Accounting**: For SEC quarterly filings, cumulative YTD cash flows are converted into discrete 90-day periods.
        - **Missing Value Handling**: Median-imputation applied across numeric financial series.
        """)

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
