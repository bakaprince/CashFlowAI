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
    cleaned = str(name).strip().lower().replace(" ", " ")
    for canonical, aliases in COLUMN_ALIASES.items():
        if cleaned == canonical or cleaned in aliases:
            return canonical.title() if canonical not in ["cash flow", "cash balance", "cash inflow", "cash outflow"] else canonical.title()
    return str(name).strip()


def clean_and_prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")

    cleaned = df.copy()
    cleaned.columns = [normalize_column_name(col) for col in cleaned.columns]

    for column in cleaned.columns:
        if "Date" in column or "Month" in column or "Period" in column:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    for column in cleaned.columns:
        if column in {"Sales", "Expenses", "Receivables", "Payables", "Cash Balance", "Cash Inflow", "Cash Outflow", "Cash Flow"}:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=[col for col in cleaned.columns if col in {"Sales", "Expenses", "Receivables", "Payables", "Cash Flow"}], how="all")
    cleaned = cleaned.sort_values(by=[col for col in cleaned.columns if "Date" in col or "Month" in col or "Period" in col][0], na_position="last") if any("Date" in col or "Month" in col or "Period" in col for col in cleaned.columns) else cleaned

    for column in ["Sales", "Expenses", "Receivables", "Payables", "Cash Balance", "Cash Inflow", "Cash Outflow", "Cash Flow"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    return cleaned.reset_index(drop=True)


def derive_cash_flow_target(df: pd.DataFrame) -> pd.Series:
    if "Cash Inflow" in df.columns and "Cash Outflow" in df.columns:
        return df["Cash Inflow"] - df["Cash Outflow"]
    if "Sales" in df.columns and "Expenses" in df.columns:
        return df["Sales"] - df["Expenses"]
    raise ValueError("Unable to derive a cash flow target from the provided columns.")
