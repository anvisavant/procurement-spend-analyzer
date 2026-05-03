"""
categorizer.py
Auto-categorizes transaction descriptions into procurement spend categories.

Strategy (two-tier):
  1. Fast keyword matching dictionary — covers ~85% of cases instantly.
  2. Sentence-transformer semantic fallback (all-MiniLM-L6-v2) for unmatched
     descriptions — cosine similarity against category label embeddings.

Usage:
    from src.categorizer import SpendCategorizer
    cat = SpendCategorizer()
    df["category"] = cat.categorize(df["description"])
"""

from __future__ import annotations
import re
from typing import Optional
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Keyword dictionary — order matters (more specific patterns first)
# ---------------------------------------------------------------------------
KEYWORD_MAP: dict[str, list[str]] = {
    "IT Software": [
        "software", "saas", "license", "cloud", "azure", "aws", "gcp",
        "subscription", "platform", "api", "devops", "cybersecurity",
        "analytics platform", "collaboration", "erp", "crm", "data",
    ],
    "Professional Services": [
        "consulting", "advisory", "audit", "legal", "counsel", "strategy",
        "management consult", "tax", "accounting", "professional service",
        "engagement", "outsourc",
    ],
    "Logistics/Freight": [
        "freight", "shipping", "logistics", "warehouse", "delivery",
        "transport", "forwarding", "3pl", "last-mile", "courier",
        "distribution", "cargo",
    ],
    "Raw Materials": [
        "resin", "chemical", "polymer", "feedstock", "material",
        "adhesive", "solvent", "compound", "alloy", "commodity",
        "packaging material", "specialty material",
    ],
    "Facilities": [
        "facilities", "maintenance", "janitorial", "hvac", "building",
        "cafeteria", "security service", "cleaning", "repair",
        "utilities", "real estate", "property",
    ],
    "Marketing": [
        "marketing", "advertising", "media buy", "brand", "creative",
        "campaign", "social media", "pr ", "public relation", "digital ad",
        "content", "seo", "sem",
    ],
    "Travel": [
        "travel", "flight", "airline", "hotel", "accommodation",
        "car rental", "conference", "lodging", "airfare", "per diem",
        "ground transport",
    ],
    "Office Supplies": [
        "office supply", "stationery", "printer", "paper", "toner",
        "desk", "chair", "ergonomic", "whiteboard", "office equipment",
        "supplies bulk",
    ],
    "HR & Training": [
        "training", "learning", "hr ", "human resource", "payroll",
        "recruitment", "benefits admin", "workforce", "onboarding",
        "employee develop",
    ],
    "Healthcare & Benefits": [
        "health insurance", "dental", "vision", "prescription",
        "wellness", "medical", "life insurance", "healthcare",
        "benefits", "pharmacy",
    ],
}

CATEGORIES = list(KEYWORD_MAP.keys())


class SpendCategorizer:
    """Two-tier spend categorizer: keyword-first, NLP fallback."""

    def __init__(self, use_nlp_fallback: bool = True):
        self.use_nlp_fallback = use_nlp_fallback
        self._model = None
        self._cat_embeddings = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def categorize(self, descriptions: pd.Series) -> pd.Series:
        """Categorize a Series of description strings. Returns a Series of category labels."""
        results = descriptions.apply(self._keyword_match)
        unmatched_mask = results == "Uncategorized"

        if unmatched_mask.any() and self.use_nlp_fallback:
            nlp_results = self._nlp_classify(descriptions[unmatched_mask])
            results[unmatched_mask] = nlp_results

        return results

    def categorize_single(self, description: str) -> str:
        """Categorize a single description string."""
        result = self._keyword_match(description)
        if result == "Uncategorized" and self.use_nlp_fallback:
            result = self._nlp_classify(pd.Series([description])).iloc[0]
        return result

    # ------------------------------------------------------------------
    # Tier 1 — keyword matching
    # ------------------------------------------------------------------
    def _keyword_match(self, text: str) -> str:
        if not isinstance(text, str):
            return "Uncategorized"
        lower = text.lower()
        for category, keywords in KEYWORD_MAP.items():
            for kw in keywords:
                if kw in lower:
                    return category
        return "Uncategorized"

    # ------------------------------------------------------------------
    # Tier 2 — sentence-transformer semantic similarity
    # ------------------------------------------------------------------
    def _load_model(self):
        """Lazy-load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._cat_embeddings = self._model.encode(CATEGORIES, normalize_embeddings=True)
        except ImportError:
            self._model = None  # graceful degradation

    def _nlp_classify(self, descriptions: pd.Series) -> pd.Series:
        if self._model is None:
            self._load_model()

        if self._model is None:
            # sentence-transformers not available — return best keyword guess or default
            return descriptions.apply(lambda x: self._keyword_match(x)
                                      if self._keyword_match(x) != "Uncategorized"
                                      else "Professional Services")

        embeddings = self._model.encode(descriptions.tolist(), normalize_embeddings=True)
        # cosine similarity (vectors already L2-normalized → dot product = cosine)
        sims = embeddings @ self._cat_embeddings.T
        best_idx = np.argmax(sims, axis=1)
        return pd.Series([CATEGORIES[i] for i in best_idx], index=descriptions.index)


# ---------------------------------------------------------------------------
# Convenience: enrich a full DataFrame
# ---------------------------------------------------------------------------
def enrich_dataframe(df: pd.DataFrame, description_col: str = "description",
                     category_col: str = "category",
                     use_nlp: bool = True) -> pd.DataFrame:
    """
    Add or overwrite `category_col` on df using SpendCategorizer.
    If `category_col` already exists (pre-labeled data), it is preserved
    and only rows with 'Uncategorized' or NaN are re-inferred.
    """
    cat = SpendCategorizer(use_nlp_fallback=use_nlp)
    df = df.copy()
    if category_col in df.columns:
        mask = df[category_col].isna() | (df[category_col] == "Uncategorized")
        if mask.any():
            df.loc[mask, category_col] = cat.categorize(df.loc[mask, description_col])
    else:
        df[category_col] = cat.categorize(df[description_col])
    return df
