# Procurement Spend Analyzer

> A Streamlit web app that ingests supplier transaction data, auto-categorizes
> spend using NLP, and generates a full procurement analytics dashboard —
> inspired by McKinsey's Spendscape platform.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- ✅ **Auto-Categorization** — Two-tier NLP engine: keyword dictionary (instant, ~85% coverage) + sentence-transformers semantic fallback (`all-MiniLM-L6-v2`) for unmatched descriptions
- ✅ **Spend Dashboard** — KPI cards, total spend by category (bar), monthly spend trends (line), top 10 vendors table, MoM change per category — all via Plotly
- ✅ **Supplier Concentration Risk** — Flags vendors ≥35% of category spend as High Risk; color-coded heatmap (categories × risk level)
- ✅ **Savings Opportunity Sizer** — Category-specific McKinsey benchmarks (3–15%), ranked savings table + stacked range bar chart
- ✅ **Anomaly Detection** — Three detection types: z-score outliers (>2σ), duplicate invoices (same vendor + amount within 30 days), round-number transactions (≥$10K multiples of $1,000)

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/anvisavant/procurement-spend-analyzer
cd procurement-spend-analyzer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

The app loads `data/mock_spend.csv` automatically in demo mode (500 transactions, 10 categories, realistic vendor names).

---

## Project Structure

```
procurement-spend-analyzer/
├── app.py                  # Streamlit entry point — all dashboard sections
├── requirements.txt        # Python dependencies
├── README.md
├── data/
│   └── mock_spend.csv      # 500-row mock transaction dataset
└── src/
    ├── __init__.py
    ├── categorizer.py      # Two-tier NLP spend categorizer
    ├── risk_scorer.py      # Supplier concentration risk scorer
    ├── savings_engine.py   # McKinsey-benchmarked savings opportunity sizer
    └── anomaly_detector.py # Z-score + duplicate + round-number anomaly detection
```

---

## Input Data Schema

| Column | Type | Description |
|---|---|---|
| `transaction_id` | int | Unique transaction identifier |
| `date` | YYYY-MM-DD | Transaction date |
| `vendor_name` | str | Supplier name |
| `description` | str | Line-item description (used for NLP categorization) |
| `amount` | float | Transaction amount (USD) |
| `department` | str | Internal cost center / business unit |

> `category` column is optional — if absent, the app auto-categorizes every row on load.

---

## Module Reference

### `src/categorizer.py` — `SpendCategorizer`
| Method | Returns |
|---|---|
| `categorize(descriptions)` | Series of category labels |
| `categorize_single(text)` | Single category string |
| `enrich_dataframe(df)` | DataFrame with `category` column added |

### `src/risk_scorer.py` — `ConcentrationRiskScorer`
| Method | Returns |
|---|---|
| `category_risk_summary()` | DataFrame — one row per category with risk level |
| `vendor_risk_detail()` | DataFrame — every vendor × category with share % |
| `risk_heatmap_matrix()` | Pivot table ready for Plotly heatmap |
| `flagged_vendors()` | High-risk vendors only |

### `src/savings_engine.py` — `SavingsEngine`
| Method | Returns |
|---|---|
| `opportunity_table()` | DataFrame ranked by max savings potential |
| `total_savings_range()` | `(low_total, high_total)` float tuple |
| `top_opportunities(n)` | Top-n categories by upside |

### `src/anomaly_detector.py` — `AnomalyDetector`
| Method | Returns |
|---|---|
| `flagged_transactions()` | All anomalous transactions with z-score + type |
| `anomaly_summary()` | Count + rate by category |
| `category_stats()` | Mean, std, threshold amount per category |

---

## Savings Benchmarks

Category-specific benchmarks derived from McKinsey PDP methodology:

| Category | Conservative | Upside |
|---|---|---|
| Travel | 8% | 15% |
| IT Software | 6% | 12% |
| Marketing | 5% | 12% |
| Raw Materials | 4% | 10% |
| Professional Services | 5% | 10% |
| Office Supplies | 5% | 10% |
| Logistics/Freight | 4% | 9% |
| HR & Training | 4% | 9% |
| Facilities | 3% | 8% |
| Healthcare & Benefits | 3% | 7% |

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit ≥ 1.32 |
| Charts | Plotly ≥ 5.18 |
| Data | Pandas ≥ 2.0, NumPy ≥ 1.24 |
| NLP | sentence-transformers (`all-MiniLM-L6-v2`) |
| ML | scikit-learn (cosine similarity) |
| Stats | SciPy (z-score, anomaly thresholds) |

---

## License

MIT — © 2026 Anvi Savant
