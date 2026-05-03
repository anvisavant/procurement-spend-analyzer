# Procurement Spend Analyzer

> A Streamlit web app that ingests supplier transaction data, auto-categorizes
> spend using NLP, and generates a full procurement analytics dashboard —
> inspired by McKinsey's Spendscape platform.

---

## Features

- [ ] **Auto-Categorization** — NLP engine (keyword matching + sentence-transformers) maps descriptions to 10 spend categories
- [ ] **Spend Dashboard** — total by category, spend over time, top 10 vendors (Plotly)
- [ ] **Supplier Concentration Risk** — flags vendors >35% of category spend as High Risk
- [ ] **Savings Opportunity Sizer** — applies McKinsey's 3–8% benchmark per category
- [ ] **Anomaly Detection** — z-score flagging for transactions >2σ from category mean

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

The app loads `data/mock_spend.csv` automatically in demo mode.

---

## Project Structure

```
procurement-spend-analyzer/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── README.md
├── data/
│   └── mock_spend.csv      # 500-row mock transaction dataset
└── src/
    ├── __init__.py
    ├── categorizer.py      # NLP categorization engine      [coming soon]
    ├── risk_scorer.py      # Concentration risk logic       [coming soon]
    ├── savings_engine.py   # Savings opportunity sizer      [coming soon]
    └── anomaly_detector.py # Z-score anomaly detection      [coming soon]
```

---

## Input Data Schema

| Column | Type | Description |
|---|---|---|
| `transaction_id` | int | Unique transaction identifier |
| `date` | YYYY-MM-DD | Transaction date |
| `vendor_name` | str | Supplier name |
| `description` | str | Line-item description (used for NLP) |
| `amount` | float | Transaction amount (USD) |
| `department` | str | Internal cost center |

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Charts | Plotly |
| Data | Pandas, NumPy |
| NLP | sentence-transformers (all-MiniLM-L6-v2) |
| ML | scikit-learn (KMeans, cosine similarity) |
| Stats | SciPy |

---

## License

MIT
