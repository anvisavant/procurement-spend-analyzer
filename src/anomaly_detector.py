"""
anomaly_detector.py
Z-score based anomaly detection for procurement transactions.

Logic:
    For each spend category, compute the mean and standard deviation
    of transaction amounts. Flag any transaction where:
        |z| > threshold  (default: 2.0)

    Z-score formula:
        z = (amount - category_mean) / category_std

    Transactions in categories with fewer than MIN_SAMPLE_SIZE records
    are excluded from z-score analysis (insufficient baseline).

    Additionally flags:
        - Duplicate invoices: same vendor + amount within 30-day window
        - Round-number transactions: amounts that are exact multiples of
          $1,000 above $10,000 (common in manual/fraudulent entries)

Usage:
    from src.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector(df)
    anomalies = detector.flagged_transactions()   # DataFrame of outliers
    summary   = detector.anomaly_summary()        # counts by category + type
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Literal

AnomalyType = Literal["z-score", "duplicate_invoice", "round_number"]

# Tuning parameters
Z_THRESHOLD       = 2.0     # standard deviations above/below category mean
MIN_SAMPLE_SIZE   = 5       # minimum transactions to compute z-score
DUPLICATE_WINDOW  = 30      # days to look back for duplicate invoice detection
ROUND_NUM_MIN     = 10_000  # minimum amount to check for round-number flag


class AnomalyDetector:
    """Statistical anomaly detection for procurement transaction data."""

    def __init__(self, df: pd.DataFrame,
                 vendor_col:   str = "vendor_name",
                 category_col: str = "category",
                 amount_col:   str = "amount",
                 date_col:     str = "date",
                 z_threshold:  float = Z_THRESHOLD):
        self.df           = df.copy()
        self.vendor_col   = vendor_col
        self.category_col = category_col
        self.amount_col   = amount_col
        self.date_col     = date_col
        self.z_threshold  = z_threshold

        # Ensure date is datetime
        if date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors="coerce")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def flagged_transactions(self) -> pd.DataFrame:
        """
        Returns a DataFrame of all anomalous transactions with columns:
            transaction_id | date | vendor_name | category | amount |
            z_score | anomaly_type | anomaly_detail
        Sorted by |z_score| descending (z-score anomalies first).
        """
        z_flagged  = self._zscore_anomalies()
        dup_flagged = self._duplicate_invoice_anomalies()
        rnd_flagged = self._round_number_anomalies()

        all_flagged = pd.concat([z_flagged, dup_flagged, rnd_flagged], ignore_index=True)

        if all_flagged.empty:
            return all_flagged

        # De-duplicate by transaction_id keeping worst z_score
        all_flagged = (
            all_flagged
            .sort_values("z_score", key=abs, ascending=False)
            .drop_duplicates(subset=["transaction_id"], keep="first")
            .reset_index(drop=True)
        )
        return all_flagged

    def anomaly_summary(self) -> pd.DataFrame:
        """
        Returns a summary DataFrame:
            category | total_transactions | anomaly_count | anomaly_rate_pct | max_z
        """
        flagged = self.flagged_transactions()
        cat_total = self.df.groupby(self.category_col).size().reset_index(name="total_transactions")

        if flagged.empty:
            cat_total["anomaly_count"]    = 0
            cat_total["anomaly_rate_pct"] = 0.0
            cat_total["max_z"]            = 0.0
            return cat_total

        cat_flagged = (
            flagged.groupby(self.category_col)
            .agg(anomaly_count=("transaction_id", "count"),
                 max_z=("z_score", lambda x: x.abs().max()))
            .reset_index()
        )
        summary = cat_total.merge(cat_flagged, on=self.category_col, how="left").fillna(0)
        summary["anomaly_rate_pct"] = (
            summary["anomaly_count"] / summary["total_transactions"] * 100
        ).round(1)
        summary["max_z"] = summary["max_z"].round(2)
        return summary.sort_values("anomaly_count", ascending=False).reset_index(drop=True)

    def category_stats(self) -> pd.DataFrame:
        """
        Returns per-category statistics used in z-score computation:
            category | count | mean | std | min | max | z_threshold_amount
        """
        stats = (
            self.df.groupby(self.category_col)[self.amount_col]
            .agg(count="count", mean="mean", std="std", min="min", max="max")
            .reset_index()
        )
        stats["z_threshold_amount"] = stats["mean"] + self.z_threshold * stats["std"]
        stats = stats.round(2)
        return stats

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _zscore_anomalies(self) -> pd.DataFrame:
        """Flag transactions > z_threshold standard deviations from category mean."""
        rows = []
        for cat, group in self.df.groupby(self.category_col):
            if len(group) < MIN_SAMPLE_SIZE:
                continue
            mean = group[self.amount_col].mean()
            std  = group[self.amount_col].std()
            if std == 0:
                continue
            group = group.copy()
            group["z_score"] = (group[self.amount_col] - mean) / std
            flagged = group[group["z_score"].abs() > self.z_threshold]
            for _, row in flagged.iterrows():
                rows.append(self._build_record(row, "z-score",
                    f"Amount ${row[self.amount_col]:,.0f} is "
                    f"{abs(row['z_score']):.1f}σ from category mean "
                    f"(${mean:,.0f} ± ${std:,.0f})"))
        return pd.DataFrame(rows)

    def _duplicate_invoice_anomalies(self) -> pd.DataFrame:
        """Flag same vendor + same amount within DUPLICATE_WINDOW days."""
        if self.date_col not in self.df.columns:
            return pd.DataFrame()

        df_sorted = self.df.sort_values([self.vendor_col, self.date_col]).copy()
        rows = []
        for vendor, group in df_sorted.groupby(self.vendor_col):
            group = group.sort_values(self.date_col).copy()
            for i, row in group.iterrows():
                window = group[
                    (group[self.date_col] >= row[self.date_col] - pd.Timedelta(days=DUPLICATE_WINDOW)) &
                    (group[self.date_col] < row[self.date_col]) &
                    (group[self.amount_col] == row[self.amount_col])
                ]
                if not window.empty:
                    rows.append(self._build_record(row, "duplicate_invoice",
                        f"Same amount ${row[self.amount_col]:,.0f} from {vendor} "
                        f"within {DUPLICATE_WINDOW}-day window"))
        return pd.DataFrame(rows)

    def _round_number_anomalies(self) -> pd.DataFrame:
        """Flag large round-number transactions (multiples of $1,000 above threshold)."""
        mask = (
            (self.df[self.amount_col] >= ROUND_NUM_MIN) &
            (self.df[self.amount_col] % 1000 == 0)
        )
        flagged = self.df[mask].copy()
        flagged["z_score"] = 0.0
        rows = []
        for _, row in flagged.iterrows():
            rows.append(self._build_record(row, "round_number",
                f"Round-number transaction: ${row[self.amount_col]:,.0f}"))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_record(self, row: pd.Series, anomaly_type: str,
                      detail: str) -> dict:
        return {
            "transaction_id": row.get("transaction_id", ""),
            "date":           row.get(self.date_col, ""),
            "vendor_name":    row.get(self.vendor_col, ""),
            "category":       row.get(self.category_col, ""),
            "amount":         row.get(self.amount_col, 0),
            "z_score":        round(row.get("z_score", 0.0), 2),
            "anomaly_type":   anomaly_type,
            "anomaly_detail": detail,
        }
