from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd


def prepare_data_stooq(symbols: list[str], target_dir: str) -> None:
    """
    Prepare data for Stooq retriever by copying selected symbols from the
    full US daily Stooq dataset into a target directory and converting to CSV.
    """
    source_dir = Path(
        r"C:\Users\ikurnosau\Projects\QuantitativeTrading\intraday-portfolio-management"
        r"\data\raw\stooq\bars\us_all\data\daily\us"
    )
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        ticker = symbol.lower()
        source_file = source_dir / f"{ticker}.us.txt"
        target_file = target_path / f"{ticker}_us_d.csv"

        if not source_file.exists():
            logging.warning("Stooq source file not found: %s", source_file)
            continue

        df = pd.read_csv(source_file)
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace(r"[<>]", "", regex=True)
            .str.upper()
        )

        required_cols = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOL"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.warning(
                "Missing columns %s in %s; skipping.",
                ", ".join(missing_cols),
                source_file,
            )
            continue

        df = df[required_cols]
        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.rename(
            columns={
                "DATE": "Date",
                "OPEN": "Open",
                "HIGH": "High",
                "LOW": "Low",
                "CLOSE": "Close",
                "VOL": "Volume",
            }
        )
        df.to_csv(target_file, index=False)
