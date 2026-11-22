# ChurnBoost AI – Predictive E-Commerce Churn Pipeline

An end-to-end predictive analytics pipeline that identifies high-risk customer churn segments and
forecasts future revenue value from raw e-commerce transaction logs, with an interactive Streamlit
dashboard to explore the results.

**Stack:** Python, Scikit-Learn, XGBoost, Pandas, NumPy, Seaborn, Matplotlib, Streamlit
**Dataset:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) – ~1M line items from a UK online retailer, Dec 2009 – Dec 2011.

---

## 1. What it does

1. **Cleans** the raw transaction log.
2. **Engineers RFM features** (Recency, Frequency, Monetary) plus a few behavioural extras per customer.
3. **Labels churn** using a time split, so the model only ever sees the past.
4. **Trains an XGBoost classifier** to predict churn, tuning the class weight (imbalance) and then the tree/regularisation hyperparameters with a cross-validated randomised search.
5. **Trains an XGBoost regressor** to estimate what a retained customer would spend, then combines the two into an expected-revenue forecast.
6. **Segments customers** into High / Medium / Low risk tiers and flags high-value customers at risk.
7. **Presents everything** in a dashboard: overview, model quality, risk segments, and a per-customer what-if simulator.

## 2. Results (hold-out set, 20% of customers)

| Metric | Value |
|---|---|
| Customers modelled | 5,281 |
| Churn rate | 56.6% |
| ROC-AUC | 0.819 |
| PR-AUC | 0.836 |
| Precision | 0.735 |
| Recall | 0.860 |
| F1 | 0.793 |
| Revenue regressor MAE (retained customers) | ~£510 (~₹56,000) |

Effect of tuning (5-fold CV PR-AUC on the training split): default parameters **0.809** → class weight tuned **0.809** → hyperparameters tuned **0.822**. On the hold-out set this lifted ROC-AUC 0.808 → 0.819 and PR-AUC 0.824 → 0.836 (the earlier hold-out figures were from the untuned model).

---

## 3. Architecture

The project is a staged pipeline that follows a **train once, serve many** pattern: the heavy work runs in one script and writes
its results to `artifacts/`; the dashboard only reads them.

```
 data/online_retail_II.xlsx            (raw UCI data, ~1.07M rows)
          │  src/data_loader.py        drop cancellations / missing IDs / bad rows / duplicates
          ▼
 data/transactions_clean.csv           (cached, so Excel is read once)
          │  src/features.py           time split at cutoff: RFM + extras (past), churn + spend (future)
          ▼
 customer table (1 row / customer: 7 features, churn, future_revenue)
          │  src/train.py
          ├──► XGBClassifier  (class-weight + hyperparameter tuning) ─► churn probability p
          └──► XGBRegressor   (retained customers, log scale)        ─► predicted spend if retained
                         │  combine: expected = (1−p)·spend, at risk = p·spend; tiers; out-of-fold scoring
                         ▼
 artifacts/  models.joblib · customers_scored.csv · metrics.json
          │  read-only
          ▼
 app.py (Streamlit)  ── filters, £→₹ at display time, charts, what-if (calls the saved classifier)

 side branch: notebooks/01_eda.ipynb ─► reports/*.png ─► explainer.html
```

| Stage | Reads | Writes |
|---|---|---|
| `src/data_loader.py` | raw Excel | `data/transactions_clean.csv` |
| `src/features.py` | cleaned transactions | in-memory customer table |
| `src/train.py` | customer table | `artifacts/models.joblib`, `customers_scored.csv`, `metrics.json` |
| `app.py` | the three artifact files | nothing (display only) |
| `notebooks/01_eda.ipynb` | cleaned transactions | `reports/*.png` |

**Design decisions**
- **Train once, serve many** – training takes 1–2 minutes; the dashboard loads saved results and never retrains.
- **Leakage-safe by construction** – the cutoff split lives in `features.py`, so every downstream stage inherits it.
- **Honest scoring** – metrics come from a 20% hold-out untouched by tuning; per-customer probabilities are out-of-fold.
- **Two models, one answer** – the classifier gives *how likely*, the regressor gives *how much*; their product is money at stake.
- **Reproducible** – a fixed seed (42) drives every split, CV fold and model.
- **Currency at the edge** – models and artifacts stay in GBP; the ₹ conversion is display-only and reversed for what-if inputs.
- **One live model call** – everything on the dashboard is precomputed except the what-if simulator, which runs the saved classifier for a single customer.

---

## 4. Project structure

```
churnboost-ai/
├── app.py                  # Streamlit dashboard
├── explainer.html          # plain-English walkthrough of the whole project (open in a browser)
├── assets/
│   ├── style.css           # dashboard theme + animations
│   └── favicon.png
├── .streamlit/config.toml  # light theme tokens
├── data/
│   ├── online_retail_II.xlsx      # raw UCI data (not committed)
│   └── transactions_clean.csv     # cached cleaned data (generated)
├── notebooks/
│   └── 01_eda.ipynb        # exploratory analysis with Seaborn (executed)
├── reports/                # EDA figures saved by the notebook (PNG)
├── src/
│   ├── data_loader.py      # load + clean transactions
│   ├── features.py         # RFM features, churn label, future revenue
│   └── train.py            # class-weight + hyperparameter tuning, training, evaluation, scoring, artifacts
├── artifacts/              # generated by train.py
│   ├── models.joblib       # fitted classifier + regressor
│   ├── customers_scored.csv# every customer with probability, tier, revenue
│   └── metrics.json        # metrics, confusion matrix, feature importance
└── requirements.txt
```

---

## 5. Core logic

### 5.1 Data cleaning – `src/data_loader.py`

The raw file is an Excel workbook with two sheets (2009-10 and 2010-11), which are concatenated. Rows are dropped when:

- `Customer ID` is missing (nothing to attribute the purchase to),
- the invoice starts with `C` (a cancellation / return),
- `Quantity <= 0` or `Price <= 0`,
- the row is an exact duplicate.

A `Revenue = Quantity × Price` column is added. Reading the Excel file is slow, so the cleaned data is
cached to `data/transactions_clean.csv` and reused on later runs.

### 5.2 Churn label without leakage – `src/features.py`

The most important design decision is the **time split**:

```
|<------------- feature window ------------->|<---- label window (90 days) ---->|
first purchase                          cutoff = 2011-09-10                  last date in data
```

- `cutoff = last transaction date − 90 days`.
- **Features** are computed only from purchases *before* the cutoff.
- **Churn = 1** if the customer made **no purchase** in the 90 days *after* the cutoff, otherwise 0.
- **`future_revenue`** is their spend in that label window (0 for churned customers).

Using the same period for features and labels would leak the answer into the model, so they are kept strictly separate.

### 5.3 Features

| Feature | Meaning |
|---|---|
| `recency` | days from the last purchase to the cutoff (**R**) |
| `frequency` | number of distinct orders (**F**) |
| `monetary` | total spend (**M**) |
| `tenure` | days from first purchase to the cutoff |
| `avg_order_value` | mean value per order |
| `avg_days_between_orders` | average gap between orders (tenure for one-time buyers) |
| `n_products` | distinct products bought |

### 5.4 Churn classifier, imbalance handling and tuning – `src/train.py`

Training runs on a stratified 80/20 split. Tuning is done on the 80% training part only, so the 20% hold-out stays untouched for the final scores.

**Stage 1 – class weight.** `scale_pos_weight` is searched over multiples (0.5×, 1×, 1.5×, 2×) of the
negative/positive ratio, scored by 5-fold stratified cross-validated **PR-AUC** (`tune_class_weight`). The best value (1.15) is kept.

**Stage 2 – hyperparameter search.** With the class weight fixed, `RandomizedSearchCV` (`tune_hyperparameters`)
tries 40 random combinations from this space, again scored by 5-fold CV PR-AUC:

| Parameter | Values searched |
|---|---|
| `n_estimators` | 150, 250, 400, 600 |
| `max_depth` | 2, 3, 4, 5, 6 |
| `learning_rate` | 0.02, 0.03, 0.05, 0.08, 0.12 |
| `subsample`, `colsample_bytree` | 0.6, 0.8, 1.0 |
| `min_child_weight` | 1, 3, 5, 10 |
| `gamma` | 0, 0.5, 1, 2 |
| `reg_lambda` | 0.5, 1, 3, 8 |

Best found: `n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.6, colsample_bytree=0.6, min_child_weight=3, gamma=2, reg_lambda=1`
(a smaller, more regularised model). The chosen parameters and CV scores are saved in `artifacts/metrics.json`.

Evaluation uses the hold-out set and reports PR-AUC, recall, precision, F1 and ROC-AUC, which suit this problem better than accuracy.
The same tuned parameters are used for the out-of-fold scoring and the final model.

### 5.5 Revenue forecast

1. An `XGBRegressor` predicts `log1p(future_revenue)` for customers who **did** return (log-scaling tames the heavily skewed spend).
2. For every customer: `predicted_spend_if_retained = expm1(prediction)`.
3. Then:
   - `expected_revenue = (1 − churn_prob) × predicted_spend_if_retained`
   - `revenue_at_risk = churn_prob × predicted_spend_if_retained`

### 5.6 Scoring and segmentation

- Each customer's `churn_prob` comes from **out-of-fold cross-validation predictions**, so the dashboard never shows a probability from a model that trained on that same customer.
- Risk tiers: **High** ≥ 0.70, **Medium** 0.40–0.70, **Low** < 0.40.
- `high_value_at_risk` = High tier **and** top-quartile historical spend.

Everything is saved to `artifacts/` for the dashboard.

---

## 6. Exploratory analysis and the explainer

- **`notebooks/01_eda.ipynb`** (run top to bottom, outputs saved) uses Seaborn/Matplotlib to show the cleaning funnel,
  monthly revenue and top countries, RFM distributions, RFM split by churn, churn rate by recency and frequency buckets,
  a correlation heatmap and the class balance. Figures are also written to `reports/`. Findings: churn climbs from 25% (silent <30 days)
  to 89% (silent >1 year); one-time buyers churn 80% versus 17% for customers with 9+ orders; revenue is strongly seasonal.
- **`explainer.html`** is a self-contained, plain-English walkthrough of the dataset, RFM, the churn definition, the model,
  tuning, results and limitations. Open it directly in a browser (it reads the PNGs from `reports/`).

---

## 7. Dashboard – `app.py`

| Tab | What you see |
|---|---|
| **Overview** | KPI row, customers per risk tier, forecast revenue by tier, churn-probability distribution |
| **Model performance** | ROC-AUC / PR-AUC / precision / recall / F1, feature importance, confusion matrix, precision-recall curve |
| **High-risk segments** | RFM scatter plot coloured by tier, average RFM profile per tier, high-value-at-risk table with CSV download |
| **Customer lookup** | A customer's churn probability on an animated meter, revenue figures, and a what-if simulator (change recency, frequency and spend to see the new probability) |

The sidebar filters by risk tier and minimum historical spend, and every view responds to them.

### Currency (₹)

The UCI dataset is in British pounds. The dashboard displays everything in **Indian rupees (₹)** by
multiplying money columns by `GBP_TO_INR` (set to **110** at the top of `app.py`; edit it to use a different rate).

- Conversion happens **only for display**. Models and `artifacts/` stay in GBP.
- The what-if simulator converts rupee inputs back to GBP before calling the model.
- KPI numbers use Indian digit grouping (e.g. ₹14,14,27,161) and chart axes use crore (Cr).

### Design and motion

- **Theme:** a warm light palette with a petrol accent, in `assets/style.css` and `.streamlit/config.toml`. Risk colours are consistent in every chart.
- **Charts:** Matplotlib/Seaborn restyled to match (no chart frame, palette-matched text, value labels).
- **KPI count-up:** the KPI row counts up from the previously shown value to the new one, so changing a filter animates from old to new.
- **Chart reveal and tab fade:** charts wipe in left to right, and tab panels fade in.
- **Risk meter:** grows from its previous value when you switch customer or move the what-if sliders.
- **Hover:** chart panels lift slightly.
- **Accessibility:** all motion is disabled under `prefers-reduced-motion`.

---

## 8. How to run this project

There are two ways to use the repo. **Option A** opens the dashboard in under a minute using the pre-trained results
that are committed in `artifacts/`. **Option B** reproduces everything from the raw data (download, cleaning, tuning, training).

### Prerequisites
- **Python 3.10 or newer** (developed and tested on Python 3.13, macOS)
- **git**
- An internet connection for the first install, for the dataset (Option B) and for the dashboard's web fonts (it falls back to system fonts offline)
- **OpenMP runtime** (needed by XGBoost):
  - macOS: `brew install libomp`
  - Ubuntu/Debian: `sudo apt-get install libgomp1`
  - Windows: nothing extra

### Step 1 – Clone and set up the environment

```bash
git clone <this-repo-url>     # the URL from the green "Code" button on GitHub
cd ChurnBoost-AI

python3 -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option A – Just see the dashboard (no data download)

```bash
streamlit run app.py
```

Open **http://localhost:8501**. The dashboard reads the committed `artifacts/` (`models.joblib`, `customers_scored.csv`, `metrics.json`).
If Streamlit asks for an email on first launch, press Enter to skip. To read the plain-English walkthrough, open `explainer.html` in a browser.

### Option B – Reproduce everything from scratch

**1. Download the dataset** (about 45 MB) into `data/`:

```bash
curl -L -o data/online_retail_ii.zip "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
unzip -o data/online_retail_ii.zip -d data/     # Windows: tar -xf data/online_retail_ii.zip -C data
rm data/online_retail_ii.zip
ls data/                                         # should show online_retail_II.xlsx
```

(Or download it manually from <https://archive.ics.uci.edu/dataset/502/online+retail+ii> and place `online_retail_II.xlsx` in `data/`.)

**2. Train the models.** This cleans the data, builds features, runs the class-weight and hyperparameter searches, trains both models and
rewrites `artifacts/`:

```bash
python -m src.train
```

The first run reads the Excel file (slow, ~1-2 minutes) and caches `data/transactions_clean.csv`; the whole run takes roughly 2-4 minutes
the first time and 1-2 minutes after that. You should see the churn rate (56.6%), the best class weight (1.15), the hyperparameter
search result and the hold-out metrics printed at the end.

**3. Launch the dashboard:**

```bash
streamlit run app.py
```

**4. (Optional) Re-run the EDA notebook.** It needs the downloaded data and regenerates the PNGs in `reports/`:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 notebooks/01_eda.ipynb
```

Or open it interactively with `jupyter lab notebooks/01_eda.ipynb` and choose the Python kernel of your `.venv` when asked.

### Quick check that it worked
- The terminal after `python -m src.train` ends with `best params: {...}` and a metrics line including `roc_auc` around 0.82 and `pr_auc` around 0.84. Small differences are possible across library versions.
- The dashboard's Overview tab shows **5,281** customers.

### Troubleshooting

| Problem | Fix |
|---|---|
| `XGBoostError: ... libomp.dylib could not be loaded` | macOS: `brew install libomp`, then rerun |
| `FileNotFoundError: ... online_retail_II.xlsx` | Do Option B step 1; the raw data is not stored in git |
| `No trained artifacts found` in the dashboard | Run `python -m src.train` (or `git checkout artifacts/` to restore the committed ones) |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| `ModuleNotFoundError: src` | Run commands from the project root, and use `python -m src.train` (not `python src/train.py`) |
| Fonts look different | The web fonts load from Google Fonts; offline it uses system fonts, which is harmless |
| Amounts look off | The dashboard converts £ to ₹ at a fixed rate; change `GBP_TO_INR` at the top of `app.py` |

### Changing things
- **Different cutoff or label window:** edit `LABEL_WINDOW_DAYS` in `src/features.py`, then retrain.
- **Wider hyperparameter search:** edit `SEARCH_SPACE` or `n_iter` in `src/train.py`, then retrain.
- **Risk tier thresholds:** edit `risk_tier()` in `src/train.py` (0.40 / 0.70 by default), then retrain.

---

## 9. Limitations and notes

- **Class imbalance is mild here.** Churn is 56.6% of customers, and PR-AUC barely changes across the class weights tried (0.808 to 0.809 in CV). The weight tuning is implemented and works, but it has little measurable effect; the hyperparameter search is what moved the score (CV PR-AUC 0.809 → 0.822).
- **CV scores from the search are slightly optimistic** because the best of 40 candidates is picked on the same folds. The untouched hold-out set confirms the gain (PR-AUC 0.824 → 0.836).
- **"Churn" is a definition, not a fact.** It means "no purchase in 90 days after the cutoff". Seasonal or irregular buyers (the data covers a gift retailer) may be labelled churned and return later.
- **One cutoff.** The model is trained and evaluated on a single time split. A rolling-window backtest would give a more robust estimate.
- **Revenue forecast is approximate.** Spend is skewed, so the regressor's error is large relative to typical customers (MAE ≈ £510).
- **Exchange rate is a fixed constant**, not a live rate.
- The dataset covers 2009–2011; it demonstrates the method and is not a statement about current customers.
