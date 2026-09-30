import re
import pandas as pd


COLUMN_ALIASES = {
    "date": ["date", "month", "period", "datetime", "timestamp", "day"],
    "sales": ["sales", "revenue", "turnover", "net sales", "gross revenue", "invoicing"],
    "expenses": ["expenses", "operating expenses", "costs", "opex", "overhead"],
    "receivables": ["receivables", "accounts receivable", "ar", "debtors"],
    "payables": ["payables", "accounts payable", "ap", "creditors"],
    "cash balance": ["cash balance", "cash_balance", "closing balance", "bank balance", "balance", "cash and cash equivalents", "cash & cash equivalents", "cash and equivalents"],
    "cash inflow": ["cash inflow", "inflow", "cash collections", "collections"],
    "cash outflow": ["cash outflow", "outflow", "disbursements", "cash spent"],
    "cash flow": ["cash flow", "net cash flow", "operating cash flow"],
}


def normalize_column_name(name: str) -> str:
    raw = str(name).strip()
    cleaned = raw.lower().replace("_", " ")

    # 1. Exact canonical or alias match
    for canonical, aliases in COLUMN_ALIASES.items():
        if cleaned == canonical or cleaned in aliases:
            return canonical.title()

    # 2. Tokenized keyword matching (handles punctuation, parentheticals, and slashes)
    clean_tokens = set(re.findall(r"\b[a-z]+\b", cleaned))

    if "cash" in clean_tokens and "flow" in clean_tokens and "inflow" not in clean_tokens and "outflow" not in clean_tokens:
        return "Cash Flow"
    if "inflow" in clean_tokens or ("cash" in clean_tokens and "in" in clean_tokens) or "collections" in clean_tokens:
        return "Cash Inflow"
    if "outflow" in clean_tokens or ("cash" in clean_tokens and "out" in clean_tokens) or "disbursements" in clean_tokens:
        return "Cash Outflow"
    if "balance" in clean_tokens or ("cash" in clean_tokens and any(k in clean_tokens for k in ["equivalents", "equivalent"])):
        return "Cash Balance"
    if "receivable" in clean_tokens or "receivables" in clean_tokens or "ar" in clean_tokens or "debtors" in clean_tokens:
        return "Receivables"
    if "payable" in clean_tokens or "payables" in clean_tokens or "ap" in clean_tokens or "creditors" in clean_tokens:
        return "Payables"
    if any(k in clean_tokens for k in ["sales", "revenue", "turnover", "invoicing"]):
        return "Sales"
    if any(k in clean_tokens for k in ["expenses", "expense", "costs", "cost", "opex", "overhead"]):
        return "Expenses"
    if any(k in clean_tokens for k in ["date", "month", "period", "timestamp", "datetime", "day"]):
        return "Date"

    return raw


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

    # Derive operational flows if dataset reports top-line revenue & operating cash flow
    if "Sales" in cleaned.columns and "Cash Flow" in cleaned.columns and "Expenses" not in cleaned.columns:
        cleaned["Expenses"] = (cleaned["Sales"] - cleaned["Cash Flow"]).clip(lower=0)
    if "Sales" in cleaned.columns and "Cash Inflow" not in cleaned.columns:
        cleaned["Cash Inflow"] = cleaned["Sales"]
    if "Expenses" in cleaned.columns and "Cash Outflow" not in cleaned.columns:
        cleaned["Cash Outflow"] = cleaned["Expenses"]

    return cleaned.reset_index(drop=True)


def derive_cash_flow_target(df: pd.DataFrame) -> pd.Series:
    if "Cash Flow" in df.columns:
        return df["Cash Flow"]
    if "Cash Inflow" in df.columns and "Cash Outflow" in df.columns:
        return df["Cash Inflow"] - df["Cash Outflow"]
    if "Sales" in df.columns and "Expenses" in df.columns:
        return df["Sales"] - df["Expenses"]
    raise ValueError("Unable to derive a cash flow target. The dataset must contain 'Cash Flow', ('Cash Inflow' and 'Cash Outflow'), or ('Sales' and 'Expenses').")
