"""
savings_engine.py
McKinsey-benchmarked savings opportunity sizer per spend category.

Benchmark bands (sourced from McKinsey PDP/Spendscape methodology):
    Category                  Low    High
    ─────────────────────────────────────
    Raw Materials             4%     10%
    IT Software               6%     12%
    Professional Services     5%     10%
    Logistics/Freight         4%      9%
    Marketing                 5%     12%
    Facilities                3%      8%
    Travel                    8%     15%
    Office Supplies           5%     10%
    HR & Training             4%      9%
    Healthcare & Benefits     3%      7%
    DEFAULT (unknown)         3%      8%

Usage:
    from src.savings_engine import SavingsEngine
    engine = SavingsEngine(df)
    opps   = engine.opportunity_table()   # DataFrame ranked by max savings
    total  = engine.total_savings_range() # (low_total, high_total)
"""

from __future__ import annotations
import pandas as pd

# Category-specific savings benchmark bands (low%, high%)
SAVINGS_BENCHMARKS: dict[str, tuple[float, float]] = {
    "Raw Materials":          (0.04, 0.10),
    "IT Software":            (0.06, 0.12),
    "Professional Services":  (0.05, 0.10),
    "Logistics/Freight":      (0.04, 0.09),
    "Marketing":              (0.05, 0.12),
    "Facilities":             (0.03, 0.08),
    "Travel":                 (0.08, 0.15),
    "Office Supplies":        (0.05, 0.10),
    "HR & Training":          (0.04, 0.09),
    "Healthcare & Benefits":  (0.03, 0.07),
}
DEFAULT_BAND = (0.03, 0.08)


class SavingsEngine:
    """Compute procurement savings opportunities from a transactions DataFrame."""

    def __init__(self, df: pd.DataFrame,
                 category_col: str = "category",
                 amount_col: str = "amount"):
        self.df           = df.copy()
        self.category_col = category_col
        self.amount_col   = amount_col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def opportunity_table(self) -> pd.DataFrame:
        """
        Returns a DataFrame ranked by max savings potential (high end):
            category | total_spend | low_pct | high_pct |
            savings_low | savings_high | savings_midpoint | rank
        """
        cat_spend = (
            self.df.groupby(self.category_col)[self.amount_col]
            .sum()
            .reset_index()
            .rename(columns={self.amount_col: "total_spend"})
        )

        rows = []
        for _, row in cat_spend.iterrows():
            cat   = row["category"]
            spend = row["total_spend"]
            lo_pct, hi_pct = SAVINGS_BENCHMARKS.get(cat, DEFAULT_BAND)
            savings_lo  = spend * lo_pct
            savings_hi  = spend * hi_pct
            rows.append({
                "category":         cat,
                "total_spend":      round(spend, 2),
                "low_pct":          round(lo_pct * 100, 1),
                "high_pct":         round(hi_pct * 100, 1),
                "savings_low":      round(savings_lo, 2),
                "savings_high":     round(savings_hi, 2),
                "savings_midpoint": round((savings_lo + savings_hi) / 2, 2),
            })

        result = (
            pd.DataFrame(rows)
            .sort_values("savings_high", ascending=False)
            .reset_index(drop=True)
        )
        result["rank"] = result.index + 1
        return result

    def total_savings_range(self) -> tuple[float, float]:
        """Return (total_low, total_high) savings across all categories."""
        opps = self.opportunity_table()
        return opps["savings_low"].sum(), opps["savings_high"].sum()

    def top_opportunities(self, n: int = 3) -> pd.DataFrame:
        """Return the top-n categories by maximum savings potential."""
        return self.opportunity_table().head(n)

    def category_savings(self, category: str) -> dict:
        """Return savings band for a single named category."""
        opps = self.opportunity_table()
        row  = opps[opps["category"] == category]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    @staticmethod
    def benchmark_for(category: str) -> tuple[float, float]:
        """Return the (low%, high%) benchmark tuple for a category name."""
        return SAVINGS_BENCHMARKS.get(category, DEFAULT_BAND)
