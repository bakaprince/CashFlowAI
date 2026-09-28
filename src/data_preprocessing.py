from __future__ import annotations

import pandas as pd


COLUMN_ALIASES = {
    "date": ["date", "month", "period", "datetime"],
    "sales": ["sales", "revenue", "turnover", "net sales"],
    "expenses": ["expenses", "operating expenses", "costs"],
    "receivables": ["receivables", "accounts receivable", "ar"],
    "payables": ["payables", "accounts payable", "ap"],
    "cash balance": ["cash balance", "cash_balance", "closing balance", "balance"],
    "cash inflow": ["cash inflow", "inflow", "cash_inflow"],
    "cash outflow": ["cash outflow", "outflow", "cash_outflow"],
    "cash flow": ["cash flow", "cash_flow", "net cash flow"],
}


def normalize_column_name(name: str) -> str:
    cleaned = str(name).strip().lower().replace("_", " ")
    for canonical, aliases in COLUMN_ALIASES.items():
        if cleaned == canonical or cleaned in aliases:
            return canonical.title()
    return str(name).strip()


def clean_and_prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")

    cleaned = df.copy()
    cleaned.columns = [normalize_column_name(col) for col in cleaned.columns]

    for column in cleaned.columns:
        if any(keyword in column for keyword in ["Date", "Month", "Period"]):
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    numeric_targets = ["Sales", "Expenses", "Receivables", "Payables", "Cash Balance", "Cash Inflow", "Cash Outflow", "Cash Flow"]
    for column in cleaned.columns:
        if column in numeric_targets:
            if cleaned[column].dtype == object:
                cleaned[column] = cleaned[column].astype(str).str.replace(r"[^\d.-]", "", regex=True)
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    metric_cols = [col for col in cleaned.columns if col in numeric_targets]
    if metric_cols:
        cleaned = cleaned.dropna(subset=metric_cols, how="all")

    date_cols = [col for col in cleaned.columns if any(k in col for k in ["Date", "Month", "Period"])]
    if date_cols:
        cleaned = cleaned.sort_values(by=date_cols[0], na_position="last")

    for column in numeric_targets:
        if column in cleaned.columns:
            median_val = cleaned[column].median()
            cleaned[column] = cleaned[column].fillna(median_val if pd.notna(median_val) else 0.0)

    return cleaned.reset_index(drop=True)


def derive_cash_flow_target(df: pd.DataFrame) -> pd.Series:
    if "Cash Flow" in df.columns:
        return df["Cash Flow"]
    if "Cash Inflow" in df.columns and "Cash Outflow" in df.columns:
        return df["Cash Inflow"] - df["Cash Outflow"]
    if "Sales" in df.columns and "Expenses" in df.columns:
        return df["Sales"] - df["Expenses"]
    raise ValueError("Unable to derive a cash flow target. The dataset must contain 'Cash Flow', ('Cash Inflow' and 'Cash Outflow'), or ('Sales' and 'Expenses').")
