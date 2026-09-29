# 💰 CashFlowAI — AI-Powered Cash Flow Forecasting & Decision Support System

An intelligent financial analytics platform combining machine learning forecasting, explainable AI (XAI), what-if scenario simulations, Monte Carlo probabilistic risk modeling, and strategic decision support.

Built with **Python**, **Scikit-learn**, **SHAP**, **Plotly**, and **Streamlit**.

---

## 📌 Overview

Maintaining a healthy cash flow is critical for business continuity. **CashFlowAI** empowers financial managers, business owners, and analysts to transition from reactive cash management to proactive, data-driven financial planning.

The platform executes a complete end-to-end analytical workflow:

```mermaid
graph TD
    A[Raw Financial Data<br>CSV / Excel] --> B[Data Cleaning & Feature Engineering]
    B --> C[ML Forecasting Model<br>Multiple Linear Regression]
    C --> D[Forecast Evaluation<br>MAE, RMSE, R²]
    C --> E[SHAP Explainability<br>Feature Attribution]
    C --> F[What-If Scenario Simulation<br>Parameter Perturbation]
    C --> G[Monte Carlo Simulation<br>Probabilistic Risk Distribution]
    E & F & G --> H[Strategic Recommendation Engine<br>Tail-Risk & Threshold Alerts]
    H --> I[Streamlit Interactive Dashboard]
```

---

## ✨ Key Features

- **📂 Real-World & Kaggle Data Ingestion**:
  - **Pre-packaged Real SEC Datasets**: Official audited quarterly cash flow filings (2017–2026) for **Apple Inc. (AAPL)**, **Microsoft Corp. (MSFT)**, **Amazon (AMZN)**, **Alphabet/Google (GOOGL)**, and **Tesla (TSLA)**.
  - **Kaggle Automated Ingestion & Preprocessing**: Download and preprocess any financial or e-commerce transaction dataset directly from Kaggle via the Kaggle API.
  - **Transaction-to-CashFlow ETL Pipeline**: Converts raw order/invoice logs (Order Date, Sales, Profit, Cost) into periodic cash flow metrics (Daily, Weekly, Monthly, Quarterly).
  - **Upload Custom Datasets**: Upload `.csv`, `.xlsx`, or `.xls` files with automatic column alias normalization and currency sanitization.
  - **Sample SME Baseline**: 2-year synthetic operational baseline (`processed_data/sample_cashflow_data.csv`).

- **📈 Machine Learning Cash Flow Forecasting**:
  - Multiple Linear Regression trained on historical operating features.
  - Out-of-sample evaluation with comprehensive statistical metrics:
    - **MAE** (Mean Absolute Error)
    - **RMSE** (Root Mean Squared Error)
    - **R² Score** (Coefficient of Determination)
  - Interactive Plotly visualizations comparing actual vs. predicted cash flow.

- **🔍 Explainable AI (SHAP Attribution)**:
  - Decomposes predictions using SHAP (SHapley Additive exPlanations).
  - Pinpoints which operational variables positively or negatively drive the current forecast.
  - Interactive table and bar chart for absolute and directional feature contributions.

- **🔄 What-If Scenario Analysis**:
  - Test hypothetical operational shocks (e.g., -20% drop in sales, +15% increase in expenses).
  - Instantaneous model re-inference comparing baseline vs. simulated projections.
  - Side-by-side delta metrics and impact classification (e.g., Improved, Riskier, Neutral).

- **🎲 Monte Carlo Risk Simulation**:
  - Simulates 2,000 potential outcomes modeled after empirical forecast residuals.
  - Computes exact probability of cash flow breaching the user-defined safety buffer.
  - Quantifies percentile distributions: **Mean**, **Median**, **P10 (downside risk)**, **P25**, **P75**, and **P90 (upside potential)**.

- **💡 Strategic Decision Support & Actionable Recommendations**:
  - Synthesizes forecasts, scenario impacts, and Monte Carlo downside risks (`P10`).
  - Generates clear, context-aware operational guidance (e.g., payment acceleration, debt restructuring, buffer maintenance).

---

## 📁 Project Structure

```
CashFlowAI/
├── app.py                            # Main Streamlit web application & UI tabs
├── unprocessed_data/                 # Raw, unprocessed datasets showing initial states
│   ├── raw_sec_edgar_aapl.json       # Official raw SEC EDGAR XBRL corporate filings (Apple CIK 0000320193)
│   ├── raw_retail_transactions.csv   # Granular order-level retail transaction logs
│   ├── raw_uncleaned_sme_cashflow.csv# Messy accounting spreadsheet with currency symbols & NaNs
│   └── README.md                     # Data provenance, CIK taxonomy & ETL documentation
├── processed_data/                   # Cleaned, normalized, and model-ready cash flow series
│   ├── real_cashflow_aapl.csv        # Apple Inc. quarterly cash flow (2017–2026)
│   ├── real_cashflow_msft.csv        # Microsoft Corp. quarterly cash flow (2017–2026)
│   ├── real_cashflow_amzn.csv        # Amazon.com Inc. quarterly cash flow (2017–2026)
│   ├── real_cashflow_googl.csv       # Alphabet Inc. quarterly cash flow (2017–2025)
│   ├── real_cashflow_tsla.csv        # Tesla Inc. quarterly cash flow (2018–2026)
│   └── sample_cashflow_data.csv      # Default sample SME dataset with 30 periods
├── src/
│   ├── __init__.py
│   ├── data_import.py                # SEC EDGAR XBRL & Kaggle transactional ETL pipelines
│   ├── data_preprocessing.py         # Column alias mapping, regex cleaning & target derivation
│   ├── explainability.py             # SHAP explainer computation
│   ├── monte_carlo.py                # Monte Carlo distribution & threshold risk estimation
│   ├── recommendations.py            # Strategic decision rules incorporating tail-risk
│   └── scenario_analysis.py          # What-if perturbation & baseline comparison
├── SYSTEM_ARCHITECTURE.md            # Detailed system design & component specification
├── requirements.txt                  # Project dependencies
└── README.md                         # Project overview & quickstart
```

> 📖 For an in-depth technical analysis, sequence diagrams, and mathematical formulations, refer to **[SYSTEM_ARCHITECTURE.md](file:///c:/Users/BakaPrince/Documents/githubRepo/CashFlowAI/SYSTEM_ARCHITECTURE.md)**.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10 to 3.14** installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/bakaprince/CashFlowAI.git
cd CashFlowAI
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

Or run directly via your virtual environment python:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application will start and open in your default browser at:
**`http://localhost:8501`**

---

## 📊 Dataset Schema

When uploading custom data, the application automatically handles column variations:

| Canonical Field | Supported Column Aliases | Description |
|---|---|---|
| `Date` | `date`, `month`, `period`, `datetime` | Observation timestamp / interval |
| `Sales` | `sales`, `revenue`, `turnover`, `net sales` | Total top-line revenue |
| `Expenses` | `expenses`, `operating expenses`, `costs` | Operating and overhead costs |
| `Receivables` | `receivables`, `accounts receivable`, `ar` | Outstanding customer payments |
| `Payables` | `payables`, `accounts payable`, `ap` | Short-term obligations to suppliers |
| `Cash Inflow` | `cash inflow`, `inflow`, `cash_inflow` | Direct cash collections |
| `Cash Outflow`| `cash outflow`, `outflow`, `cash_outflow` | Direct cash disbursements |
| `Cash Flow` | `cash flow`, `cash_flow`, `net cash flow` | *Optional*. Derived automatically as `Inflow - Outflow` or `Sales - Expenses` if omitted. |

---

## 🌐 Data Ingestion & Live ETL Pipelines

CashFlowAI includes automated ingestion and preprocessing capabilities accessible directly in the Streamlit application and programmatically via `src/data_import.py`:

### 1. Official SEC EDGAR Corporate Filings (No API Key Required)
- **Built-in Datasets**: Select corporate quarterly filings (2017–2026) for Apple (`AAPL`), Microsoft (`MSFT`), Amazon (`AMZN`), Alphabet (`GOOGL`), or Tesla (`TSLA`) directly from the sidebar.
- **Live Retrieval**: Query any public SEC ticker live via the **Kaggle & Live Data Pipeline** sidebar tab.
- **XBRL Ingestion Module**: Use `fetch_sec_cashflow_dataset(ticker)` from `src/data_import.py`.

### 2. Live Kaggle Ingestion & Preprocessing
- Connect directly to the Kaggle API to pull enterprise datasets (e.g. `thedevastator/superstore-sales`).
- Automatically aggregates granular transaction logs into periodic cash flows.
- **Kaggle Module**: Use `download_and_preprocess_kaggle(slug)` from `src/data_import.py`.

### 3. Local Transactional & Accounting Spreadsheets
- Upload raw transaction logs or messy spreadsheets (`.csv`, `.xlsx`, `.xls`) with automated column aliasing, currency string stripping, and median imputation.
- Select frequency: `Daily`, `Weekly`, `Monthly`, or `Quarterly`.

---

## 🔬 Data Lineage & ETL Processing (Raw vs Processed)

CashFlowAI maintains both raw and processed datasets to demonstrate exact real-world data engineering:
- **`unprocessed_data/`**: Holds raw SEC EDGAR XBRL filings (`raw_sec_edgar_aapl.json`), granular transactional order records (`raw_retail_transactions.csv`), and messy SME bookkeeping spreadsheets (`raw_uncleaned_sme_cashflow.csv`).
- **`processed_data/`**: Holds standardized, normalized, and imputed cash flow time series consumed directly by the machine learning forecasting model.

To inspect how raw data is transformed into model-ready series:
- Launch `streamlit run app.py` and select **`📦 Raw Retail Transactions`** or **`🧹 Raw Messy SME Operations`** in the sidebar.
- Open **Tab 0 ("📊 Data & Preprocessing")** to view the live **Raw Unprocessed Data Preview** alongside the cleaned, model-ready dataset.

---

## 🛠️ Technology Stack

| Library / Tool | Purpose |
|---|---|
| **[Streamlit](https://streamlit.io/)** | Interactive Web Dashboard & UI |
| **[Scikit-Learn](https://scikit-learn.org/)** | Forecasting models & regression evaluation metrics |
| **[SHAP](https://shap.readthedocs.io/)** | Explainable AI & model feature attribution |
| **[Plotly Express](https://plotly.com/python/)** | Interactive financial charts & distributions |
| **[Pandas](https://pandas.pydata.org/)** | Data ingestion, sanitization & manipulation |
| **[NumPy](https://numpy.org/)** | Matrix computations & Monte Carlo sampling |
| **[Matplotlib](https://matplotlib.org/)** | Statistical feature importance visualizations |
| **[OpenPyXL](https://openpyxl.readthedocs.io/)** | Support for Excel spreadsheet uploads |

---

## 📄 License

This project is licensed under the MIT License - see the repository details for more information.
