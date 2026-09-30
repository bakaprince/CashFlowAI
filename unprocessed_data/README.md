# 📂 Unprocessed (Raw) Data Directory — Data Lineage & Provenance

This directory contains the **unprocessed raw data assets** utilized by CashFlowAI before ETL ingestion, column normalization, currency sanitization, missing-value imputation, or temporal aggregation.

---

## 🏛️ 1. SEC EDGAR Audited Corporate Filings

### Where Did We Obtain This Data?
- **Official Source**: **U.S. Securities and Exchange Commission (SEC) EDGAR System**
- **Public API Endpoint**: `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json`
- **Regulation**: Under U.S. federal securities laws, publicly traded companies must submit audited financial disclosures on **Form 10-K** (Annual Report) and **Form 10-Q** (Quarterly Report) in standardized **eXtensible Business Reporting Language (XBRL)** format.
- **Access Method**: Direct programmatic retrieval using python `urllib.request` with standard SEC research user-agent compliance (`CashFlowAIResearch research@cashflowai.internal`). No private API key or paywall required.

### Company Identifiers (CIK Codes):
| Ticker | Entity Name | SEC Central Index Key (CIK) | Primary Filing Forms |
|---|---|---|---|
| **AAPL** | Apple Inc. | `0000320193` | 10-Q & 10-K |
| **MSFT** | Microsoft Corp. | `0000789019` | 10-Q & 10-K |
| **AMZN** | Amazon.com Inc. | `0001018724` | 10-Q & 10-K |
| **GOOGL** | Alphabet Inc. | `0001652044` | 10-Q & 10-K |
| **TSLA** | Tesla Inc. | `0001318605` | 10-Q & 10-K |

### Raw SEC-Derived Files in this Directory:
- **`raw_sec_edgar_aapl.csv`**, **`raw_sec_edgar_amzn.csv`**, **`raw_sec_edgar_googl.csv`**, **`raw_sec_edgar_msft.csv`**, **`raw_sec_edgar_tsla.csv`**:
  Standardized CSV representations extracted directly from the official SEC EDGAR XBRL Company Facts database. Each file contains genuine SEC-reported financial line items: `Date`, `Revenue`, `Operating Cash Flow`, `Accounts Receivable`, `Accounts Payable`, and `Cash and Cash Equivalents`.

### How Was It Processed? (Raw $\to$ `processed_data/real_cashflow_aapl.csv`)
1. **US-GAAP XBRL Taxonomy Extraction**:
   - **Sales / Revenue**: Queried `RevenueFromContractWithCustomerExcludingAssessedTax` and `SalesRevenueNet`. Filtered for 90-day duration quarters ($75 \le \text{days} \le 105$).
   - **Operating Cash Flow**: Queried `NetCashProvidedByUsedInOperatingActivities`.
   - **Balance Sheet & Working Capital**: Queried `AccountsReceivableNetCurrent`, `AccountsPayableCurrent`, and `CashAndCashEquivalentsAtCarryingValue`.
2. **De-cumulation of Year-to-Date (YTD) Cash Flow**:
   - In SEC 10-Q filings, operating cash flows are reported **cumulatively** (Q1 = 3 mos, Q2 = 6 mos, Q3 = 9 mos, Q4 = 12 mos).
   - Our pipeline transforms cumulative amounts into discrete standalone quarter increments:
     $$\text{CashFlow}_{Q_i} = \text{YTD}_{Q_i} - \text{YTD}_{Q_{i-1}}$$
3. **Derived Metrics**:
   - $\text{Cash Inflow} = \text{Sales}$
   - $\text{Expenses} = \max(0, \text{Sales} - \text{Cash Flow})$
   - $\text{Cash Outflow} = \text{Expenses}$

---

## 🛒 2. Granular Transactional E-Commerce / Retail Logs

### Where Did We Obtain This Data?
- **Source Context**: Enterprise transactional logs and Kaggle business transaction benchmarks (e.g. Superstore Sales, Olist Brazilian E-Commerce, Retail Order Databases).
- **Format**: Point-in-time individual customer purchases with invoice numbers, line-item quantities, discounts, and payment settlement statuses.

### Raw File in this Directory:
- **`raw_retail_transactions.csv`**:
  Contains 244 individual transaction records across 2023–2024 with columns:
  `Invoice_ID`, `Transaction_Date`, `Customer_ID`, `Category`, `Quantity`, `Unit_Price`, `Discount_Pct`, `Total_Sales_Amount`, `COGS_Expense`, `Payment_Status`.

### How Was It Processed?
Implemented in `src/data_import.py` via `preprocess_transactional_dataset()`:
1. **Datetime Parsing**: Automatically identifies the temporal column (`Transaction_Date`) and parses it into ISO datetime objects.
2. **Numeric Extraction**: Strips currency glyphs (`$`) and comma formatting from strings.
3. **Temporal Aggregation**: Resamples granular order records into periodic monthly buckets (`freq="ME"`).
4. **Financial Derivations**:
   - Aggregates periodic `Sales` ($\sum \text{Amount}$) and `Expenses` ($\sum \text{Cost}$).
   - Derives working capital estimates: Receivables ($\approx 20\%$ of sales) and Payables ($\approx 25\%$ of expenses).
   - Computes rolling cumulative `Cash Balance` from baseline liquidity.

---

## 📋 3. Messy SME Operational Spreadsheets

### Where Did We Obtain This Data?
- **Source Context**: Modeled on real-world small and medium enterprise (SME) accounting exports (e.g., QuickBooks, Tally, Zoho Books, manual Excel ledgers).
- **Flaws Present**:
  - Non-standard alias headers (`Net Revenue / Sales`, `Operating_Expenses`, `AR (Receivables)`, `Bank Closing Balance`).
  - Formatting noise: Dollar signs (`$`), commas (`125,000.00`), whitespace.
  - Missing entries (`NaN` / null values).
  - Unsorted chronological order.

### Raw File in this Directory:
- **`raw_uncleaned_sme_cashflow.csv`**:
  Contains 30 operational accounting periods in messy spreadsheet format.

### How Was It Processed? (Raw $\to$ `processed_data/sample_cashflow_data.csv`)
Implemented in `src/data_preprocessing.py` via `clean_and_prepare_dataset()`:
1. **Column Alias Resolution**: Uses regex token matching against `COLUMN_ALIASES` to standardize varied headers into canonical names (`Sales`, `Expenses`, `Receivables`, `Payables`, `Cash Balance`, `Date`).
2. **Regex Cleaning**: Strips all currency symbols, commas, and whitespace (`re.sub(r"[^\d.-]", "", val)`).
3. **Missing Value Imputation**: Automatically imputes missing cells using column-wise median values to preserve distribution properties.
4. **Chronological Sorting**: Sorts records chronologically by date.
5. **Target Derivation**: Derives `Cash Flow = Cash Inflow - Cash Outflow` or `Sales - Expenses`.

---

## 🔬 Ingestion & ETL Pipeline Execution

The automated ETL transformation can be inspected interactively in the web dashboard or programmatically via Python:

- **Web Dashboard**: Launch `streamlit run app.py` and select either `📦 Raw Retail Transactions` or `🧹 Raw Messy SME Operations` to see real-time raw-to-processed data transformation in **Tab 0 ("📊 Data & Preprocessing")**.
- **Python / Programmatic**: Import directly from `src/data_import.py` (`fetch_sec_cashflow_dataset`, `preprocess_transactional_dataset`) or `src/data_preprocessing.py` (`clean_and_prepare_dataset`).

