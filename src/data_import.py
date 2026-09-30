from __future__ import annotations

import json
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import numpy as np
import pandas as pd

from src.data_preprocessing import clean_and_prepare_dataset, normalize_column_name

logger = logging.getLogger("CashFlowAI.DataImport")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
DATA_DIR = PROCESSED_DATA_DIR  # Backward-compatible alias
RAW_DATA_DIR = BASE_DIR / "unprocessed_data"

# SEC CIK Mappings for prominent companies
SEC_CIK_MAP = {
    "AAPL": ("0000320193", "Apple Inc."),
    "MSFT": ("0000789019", "Microsoft Corporation"),
    "AMZN": ("0001018724", "Amazon.com Inc."),
    "GOOGL": ("0001652044", "Alphabet Inc."),
    "TSLA": ("0001318605", "Tesla Inc."),
}


def fetch_sec_cashflow_dataset(
    ticker: str = "AAPL",
    user_agent: str = "CashFlowAIResearch research@cashflowai.internal",
    prefer_local_raw: bool = True,
) -> pd.DataFrame:
    """
    Fetches real audited quarterly financial statements from SEC EDGAR XBRL API
    (or loads from local SEC EDGAR-derived raw CSV) and transforms them into a clean,
    unaccumulated cash flow time series.
    """
    ticker_upper = ticker.upper()
    if ticker_upper not in SEC_CIK_MAP:
        raise ValueError(f"Ticker '{ticker}' not found in supported CIK map. Supported: {list(SEC_CIK_MAP.keys())}")

    local_raw_csv = RAW_DATA_DIR / f"raw_sec_edgar_{ticker_upper.lower()}.csv"
    if prefer_local_raw and local_raw_csv.exists():
        raw_df = pd.read_csv(local_raw_csv)
        cleaned = clean_and_prepare_dataset(raw_df)
        cleaned["Date"] = pd.to_datetime(cleaned["Date"]).dt.strftime("%Y-%m-%d")
        final_cols = ["Date", "Sales", "Expenses", "Receivables", "Payables", "Cash Balance", "Cash Inflow", "Cash Outflow", "Cash Flow"]
        for c in final_cols:
            if c != "Date":
                cleaned[c] = cleaned[c].astype(float).round(2)
        return cleaned[final_cols]

    cik, entity_name = SEC_CIK_MAP[ticker_upper]
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"

    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    facts = data.get("facts", {}).get("us-gaap", {})
    if not facts:
        raise ValueError(f"No US-GAAP financial facts found for {entity_name}.")

    # 1. Extract Quarterly Sales / Revenue
    rev_key = next((k for k in ["RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"] if k in facts), None)
    if not rev_key:
        raise ValueError(f"Could not find revenue/sales fact for {entity_name}.")

    df_rev = pd.DataFrame(facts[rev_key]["units"]["USD"])
    df_rev = df_rev[df_rev["form"].isin(["10-Q", "10-K"])].copy()
    df_rev["start"] = pd.to_datetime(df_rev.get("start"))
    df_rev["end"] = pd.to_datetime(df_rev["end"])
    df_rev["days"] = (df_rev["end"] - df_rev["start"]).dt.days
    df_rev_q = df_rev[(df_rev["days"] >= 75) & (df_rev["days"] <= 105)].drop_duplicates(subset=["end"], keep="last")
    df_rev_q = df_rev_q[["end", "val"]].rename(columns={"val": "Sales"})

    # 2. Operating Cash Flow (un-cumulate year-to-date values into discrete quarters)
    cf_key = "NetCashProvidedByUsedInOperatingActivities"
    if cf_key not in facts:
        raise ValueError(f"Could not find Operating Cash Flow fact for {entity_name}.")

    df_cf = pd.DataFrame(facts[cf_key]["units"]["USD"])
    df_cf = df_cf[df_cf["form"].isin(["10-Q", "10-K"])].copy()
    df_cf["end"] = pd.to_datetime(df_cf["end"])
    df_cf = df_cf.sort_values(by=["fy", "end"]).drop_duplicates(subset=["fy", "fp"], keep="last")

    q_cfs: List[Dict[str, Any]] = []
    for fy, group in df_cf.groupby("fy"):
        sorted_g = group.sort_values("end")
        prev_ytd = 0.0
        for _, row in sorted_g.iterrows():
            val = float(row["val"])
            standalone = val - prev_ytd
            prev_ytd = val
            q_cfs.append({"end": row["end"], "Cash Flow": standalone})
    df_cf_standalone = pd.DataFrame(q_cfs).drop_duplicates(subset=["end"], keep="last")

    # 3. Balance Sheet items (Point-in-time instantaneous items)
    def get_instant(tag: str, col_name: str) -> pd.DataFrame:
        if tag not in facts:
            return pd.DataFrame()
        d = pd.DataFrame(facts[tag]["units"]["USD"])
        d = d[d["form"].isin(["10-Q", "10-K"])].copy()
        d["end"] = pd.to_datetime(d["end"])
        d = d.sort_values("end").drop_duplicates(subset=["end"], keep="last")
        return d[["end", "val"]].rename(columns={"val": col_name})

    df_ar = get_instant("AccountsReceivableNetCurrent", "Receivables")
    df_ap = get_instant("AccountsPayableCurrent", "Payables")
    df_cash = get_instant("CashAndCashEquivalentsAtCarryingValue", "Cash Balance")

    # 4. Merge datasets on period end date
    merged = df_rev_q.merge(df_cf_standalone, on="end", how="inner")
    for d in [df_ar, df_ap, df_cash]:
        if not d.empty:
            merged = merged.merge(d, on="end", how="left")

    merged = merged.sort_values("end").dropna(subset=["Sales", "Cash Flow"]).reset_index(drop=True)
    merged["Date"] = merged["end"].dt.strftime("%Y-%m-%d")
    merged["Expenses"] = (merged["Sales"] - merged["Cash Flow"]).clip(lower=0)
    merged["Cash Inflow"] = merged["Sales"]
    merged["Cash Outflow"] = merged["Expenses"]

    final_cols = ["Date", "Sales", "Expenses", "Receivables", "Payables", "Cash Balance", "Cash Inflow", "Cash Outflow", "Cash Flow"]
    for c in final_cols:
        if c not in merged.columns:
            merged[c] = 0.0
        if c != "Date":
            merged[c] = merged[c].fillna(merged[c].median()).astype(float).round(2)

    return merged[final_cols]


def preprocess_transactional_dataset(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    amount_col: Optional[str] = None,
    cost_col: Optional[str] = None,
    freq: str = "M",
) -> pd.DataFrame:
    """
    Transforms raw transactional or e-commerce datasets (e.g. from Kaggle)
    into periodic aggregated cash flow time series (Daily, Weekly, Monthly).
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    cleaned_df = df.copy()

    # Detect Date column if not provided (prioritize date/time/period while excluding ID/Number columns)
    if not date_col:
        date_candidates = [
            c for c in cleaned_df.columns
            if any(k in c.lower() for k in ["date", "datetime", "timestamp", "period"])
            and not any(neg in c.lower() for neg in ["_id", "id", "num", "no"])
        ]
        if not date_candidates:
            date_candidates = [c for c in cleaned_df.columns if any(k in c.lower() for k in ["date", "time", "day", "period"])]
        if not date_candidates:
            raise ValueError(f"Could not automatically detect Date column from: {list(cleaned_df.columns)}")
        date_col = date_candidates[0]

    cleaned_df["_parsed_date"] = pd.to_datetime(cleaned_df[date_col], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=["_parsed_date"]).sort_values("_parsed_date")
    if cleaned_df.empty:
        raise ValueError(f"No valid dates could be parsed from column '{date_col}'.")

    # Detect Sales/Amount column (prioritize total_amount / sales / revenue over unit_price)
    if not amount_col:
        primary_amt = [
            c for c in cleaned_df.columns
            if any(k in c.lower() for k in ["total", "sales", "revenue", "turnover", "gross"])
            and not any(neg in c.lower() for neg in ["unit", "price_per", "discount"])
        ]
        if primary_amt:
            amount_col = primary_amt[0]
        else:
            amt_candidates = [c for c in cleaned_df.columns if any(k in c.lower() for k in ["amount", "sales", "revenue", "price", "turnover"])]
            if not amt_candidates:
                raise ValueError(f"Could not automatically detect Sales/Amount column from: {list(cleaned_df.columns)}")
            amount_col = amt_candidates[0]

    # Clean amount strings
    if cleaned_df[amount_col].dtype == object:
        cleaned_df[amount_col] = cleaned_df[amount_col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
    cleaned_df["_amount"] = pd.to_numeric(cleaned_df[amount_col], errors="coerce").fillna(0.0)

    # Detect Cost/Expenses column (if any)
    if not cost_col:
        cost_candidates = [c for c in cleaned_df.columns if any(k in c.lower() for k in ["cost", "cogs", "expense", "fee", "spending"])]
        cost_col = cost_candidates[0] if cost_candidates else None

    if cost_col:
        if cleaned_df[cost_col].dtype == object:
            cleaned_df[cost_col] = cleaned_df[cost_col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
        cleaned_df["_cost"] = pd.to_numeric(cleaned_df[cost_col], errors="coerce").fillna(0.0)
    else:
        # If profit exists, cost = amount - profit; otherwise estimate industry standard ~65% operating cost
        profit_cands = [c for c in cleaned_df.columns if "profit" in c.lower()]
        if profit_cands:
            p_col = profit_cands[0]
            if cleaned_df[p_col].dtype == object:
                cleaned_df[p_col] = cleaned_df[p_col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
            cleaned_df["_cost"] = cleaned_df["_amount"] - pd.to_numeric(cleaned_df[p_col], errors="coerce").fillna(0.0)
        else:
            cleaned_df["_cost"] = cleaned_df["_amount"] * 0.65

    # Map frequency aliases to modern pandas syntax
    freq_map = {"M": "ME", "Q": "QE", "W": "W", "D": "D"}
    effective_freq = freq_map.get(freq.upper(), freq)

    # Group by period (Monthly 'ME', Weekly 'W', or Daily 'D')
    grouped = cleaned_df.set_index("_parsed_date").resample(effective_freq).agg(
        Sales=("_amount", "sum"),
        Expenses=("_cost", "sum"),
        TransactionCount=("_amount", "count")
    ).reset_index()

    # Filter out empty periods
    grouped = grouped[grouped["TransactionCount"] > 0].reset_index(drop=True)
    grouped["Date"] = grouped["_parsed_date"].dt.strftime("%Y-%m-%d")

    # Financial derivation
    grouped["Cash Inflow"] = grouped["Sales"].round(2)
    grouped["Cash Outflow"] = grouped["Expenses"].round(2)
    grouped["Cash Flow"] = (grouped["Cash Inflow"] - grouped["Cash Outflow"]).round(2)

    # Estimate working capital metrics:
    # Receivables ~ 15-25% of current sales
    grouped["Receivables"] = (grouped["Sales"] * 0.20).round(2)
    # Payables ~ 20-30% of current expenses
    grouped["Payables"] = (grouped["Expenses"] * 0.25).round(2)
    # Rolling Cash Balance starting with baseline liquidity buffer
    base_balance = float(grouped["Sales"].mean() * 0.5)
    grouped["Cash Balance"] = (base_balance + grouped["Cash Flow"].cumsum()).clip(lower=0).round(2)

    final_cols = ["Date", "Sales", "Expenses", "Receivables", "Payables", "Cash Balance", "Cash Inflow", "Cash Outflow", "Cash Flow"]
    return grouped[final_cols]


def download_and_preprocess_kaggle(
    dataset_slug: str,
    target_filename: Optional[str] = None,
    output_csv_name: str = "kaggle_preprocessed_cashflow.csv"
) -> pd.DataFrame:
    """
    Downloads a dataset from Kaggle via the Kaggle API and runs automated preprocessing.
    Requires kaggle credentials (e.g. ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY).
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    extract_path = RAW_DATA_DIR / dataset_slug.replace("/", "_")
    extract_path.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading Kaggle dataset: {dataset_slug}")
        api.dataset_download_files(dataset_slug, path=str(extract_path), unzip=True)
    except Exception as exc:
        raise RuntimeError(
            f"Kaggle API download failed: {exc}.\n"
            "Please ensure you have your Kaggle credentials set in ~/.kaggle/kaggle.json "
            "or download the CSV manually from Kaggle and place it in 'unprocessed_data/'."
        ) from exc

    # Locate CSV file in extracted directory
    csv_files = list(extract_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded Kaggle dataset {dataset_slug}.")

    selected_csv = csv_files[0]
    if target_filename:
        for f in csv_files:
            if target_filename.lower() in f.name.lower():
                selected_csv = f
                break

    raw_df = pd.read_csv(selected_csv)

    # Check if dataset already has cash flow structure or needs transactional transformation
    cols_lower = [c.lower() for c in raw_df.columns]
    if any(k in cols_lower for k in ["cash flow", "cash_flow", "inflow", "cash inflow"]) and any(k in cols_lower for k in ["sales", "revenue", "expenses"]):
        preprocessed = clean_and_prepare_dataset(raw_df)
    else:
        preprocessed = preprocess_transactional_dataset(raw_df)

    output_path = DATA_DIR / output_csv_name
    preprocessed.to_csv(output_path, index=False)
    logger.info(f"Saved preprocessed Kaggle dataset to {output_path}")
    return preprocessed
