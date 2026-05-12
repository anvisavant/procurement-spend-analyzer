"""
risk_scorer.py
Supplier concentration risk analysis per spend category.

Risk logic:
    High   — top vendor >= 35% of category total spend
    Medium — top vendor >= 22% and < 35%
    Low    — top vendor < 22%

Also flags any single vendor that represents >= 10% of TOTAL portfolio spend
as a "Portfolio-Level Risk" regardless of category.

Usage:
    from src.risk_scorer import ConcentrationRiskScorer
    scorer = ConcentrationRiskScorer(df)
    summary  = scorer.category_risk_summary()   # DataFrame, one row per category
    vendors  = scorer.vendor_risk_detail()       # DataFrame, one row per vendor×category
    heatmap  = scorer.risk_heatmap_matrix()      # pivot: categories × risk levels
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Literal

RiskLevel = Literal["High", "Medium", "Low"]

# Thresholds (% of category spend)
HIGH_THRESHOLD   = 35.0
MEDIUM_THRESHOLD = 22.0

# Portfolio-level flag threshold (% of total all-category spend)
PORTFOLIO_THRESHOLD = 10.0


class ConcentrationRiskScorer:
    """Compute supplier concentration risk from a transactions DataFrame."""

    def __init__(self, df: pd.DataFrame,
                 vendor_col: str = "vendor_name",
                 category_col: str = "category",
                 amount_col: str = "amount"):
        self.df = df.copy()
        self.vendor_col   = vendor_col
        self.category_col = category_col
        self.amount_col   = amount_col
        self._total_spend = self.df[amount_col].sum()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def category_risk_summary(self) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per category:
            category | total_spend | top_vendor | top_vendor_spend |
            top_vendor_pct | risk_level | vendor_count
        """
        rows = []
        for cat, group in self.df.groupby(self.category_col):
            vendor_totals = (
                group.groupby(self.vendor_col)[self.amount_col]
                .sum()
                .sort_values(ascending=False)
            )
            total = vendor_totals.sum()
            top_vendor = vendor_totals.index[0]
            top_spend  = vendor_totals.iloc[0]
            top_pct    = top_spend / total * 100 if total > 0 else 0

            rows.append({
                "category":         cat,
                "total_spend":      round(total, 2),
                "top_vendor":       top_vendor,
                "top_vendor_spend": round(top_spend, 2),
                "top_vendor_pct":   round(top_pct, 1),
                "risk_level":       self._classify(top_pct),
                "vendor_count":     len(vendor_totals),
            })

        result = pd.DataFrame(rows).sort_values("total_spend", ascending=False)
        return result.reset_index(drop=True)

    def vendor_risk_detail(self) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per vendor×category combination,
        including share of category spend, share of portfolio spend, and
        portfolio-level risk flag.
        """
        rows = []
        for cat, group in self.df.groupby(self.category_col):
            cat_total = group[self.amount_col].sum()
            vendor_totals = group.groupby(self.vendor_col)[self.amount_col].sum()
            for vendor, spend in vendor_totals.items():
                cat_pct       = spend / cat_total * 100 if cat_total > 0 else 0
                portfolio_pct = spend / self._total_spend * 100 if self._total_spend > 0 else 0
                rows.append({
                    "vendor":          vendor,
                    "category":        cat,
                    "spend":           round(spend, 2),
                    "cat_share_pct":   round(cat_pct, 1),
                    "portfolio_pct":   round(portfolio_pct, 2),
                    "risk_level":      self._classify(cat_pct),
                    "portfolio_flag":  portfolio_pct >= PORTFOLIO_THRESHOLD,
                })

        result = pd.DataFrame(rows).sort_values("spend", ascending=False)
        return result.reset_index(drop=True)

    def risk_heatmap_matrix(self) -> pd.DataFrame:
        """
        Returns a pivot table of shape (n_categories × 3) showing the number
        of vendors at each risk level per category. Suitable for Plotly heatmap.
        """
        detail = self.vendor_risk_detail()
        pivot = (
            detail.groupby(["category", "risk_level"])
            .size()
            .unstack(fill_value=0)
        )
        for col in ["High", "Medium", "Low"]:
            if col not in pivot.columns:
                pivot[col] = 0
        return pivot[["High", "Medium", "Low"]]

    def flagged_vendors(self) -> pd.DataFrame:
        """Return only vendors classified as High concentration risk."""
        detail = self.vendor_risk_detail()
        return detail[detail["risk_level"] == "High"].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(pct: float) -> RiskLevel:
        if pct >= HIGH_THRESHOLD:
            return "High"
        elif pct >= MEDIUM_THRESHOLD:
            return "Medium"
        return "Low"
