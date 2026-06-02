from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

INPUT_CSV = Path("csv") / "Brigade_Bangalore_10_April_26 (1)bc6219c.csv"
OUTPUT_DIR = Path("outputs")

SALES_SUMMARY_PATH = OUTPUT_DIR / "sales_summary.json"
BRAND_SUMMARY_PATH = OUTPUT_DIR / "brand_summary.json"
CATEGORY_SUMMARY_PATH = OUTPUT_DIR / "category_summary.json"
SALESPERSON_SUMMARY_PATH = OUTPUT_DIR / "salesperson_summary.json"

NUMERIC_COLUMNS = ["qty", "GMV", "NMV"]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Input path is not a file: {csv_path}")
    return pd.read_csv(csv_path)


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = coerce_numeric(df[column])
    if "brand_name" in df.columns:
        df["brand_name"] = df["brand_name"].fillna("Unknown")
    if "dep_name" in df.columns:
        df["dep_name"] = df["dep_name"].fillna("Unknown")
    if "salesperson_name" in df.columns:
        df["salesperson_name"] = df["salesperson_name"].fillna("Unknown")
    return df


def resolve_order_count(df: pd.DataFrame) -> int:
    for column in ["order_id", "invoice_number"]:
        if column in df.columns:
            return int(df[column].dropna().nunique())
    return int(len(df))


def round_value(value: float, decimals: int = 2) -> float:
    return float(round(value, decimals))


def build_sales_summary(df: pd.DataFrame) -> dict:
    total_orders = resolve_order_count(df)
    total_qty = df["qty"].sum() if "qty" in df.columns else 0.0
    total_gmv = df["GMV"].sum() if "GMV" in df.columns else 0.0
    total_nmv = df["NMV"].sum() if "NMV" in df.columns else 0.0
    average_bill = total_nmv / total_orders if total_orders else 0.0

    return {
        "total_orders": int(total_orders),
        "total_qty": round_value(total_qty),
        "total_gmv": round_value(total_gmv),
        "total_nmv": round_value(total_nmv),
        "average_bill": round_value(average_bill),
    }


def build_group_summary(df: pd.DataFrame, group_column: str, output_key: str) -> pd.DataFrame:
    if group_column not in df.columns:
        raise KeyError(f"Missing required column: {group_column}")

    grouped = (
        df.groupby(group_column, dropna=False)
        .agg(total_qty=("qty", "sum"), total_nmv=("NMV", "sum"))
        .reset_index()
    )

    grouped["total_qty"] = grouped["total_qty"].fillna(0)
    grouped["total_nmv"] = grouped["total_nmv"].fillna(0)
    grouped = grouped.sort_values(by=["total_nmv", group_column], ascending=[False, True])
    grouped["ranking"] = range(1, len(grouped) + 1)
    grouped = grouped.rename(columns={group_column: output_key})

    grouped["total_qty"] = grouped["total_qty"].apply(round_value)
    grouped["total_nmv"] = grouped["total_nmv"].apply(round_value)
    return grouped[[output_key, "total_qty", "total_nmv", "ranking"]]


def write_json(path: Path, payload: Iterable | dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {display_path(path)}")


def main() -> None:
    df = load_data(INPUT_CSV)
    df = normalize_dataframe(df)
    ensure_output_dir(OUTPUT_DIR)

    sales_summary = build_sales_summary(df)
    write_json(SALES_SUMMARY_PATH, sales_summary)

    brand_summary = build_group_summary(df, "brand_name", "brand_name")
    write_json(BRAND_SUMMARY_PATH, brand_summary.to_dict(orient="records"))

    category_summary = build_group_summary(df, "dep_name", "dep_name")
    write_json(CATEGORY_SUMMARY_PATH, category_summary.to_dict(orient="records"))

    salesperson_summary = build_group_summary(df, "salesperson_name", "salesperson_name")
    write_json(SALESPERSON_SUMMARY_PATH, salesperson_summary.to_dict(orient="records"))


if __name__ == "__main__":
    main()
