"""ChurnBoost AI - Streamlit dashboard. Run: streamlit run app.py"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"

INK, MUTED, LINE, ACCENT = "#1d2433", "#667085", "#e3ded2", "#25586b"
TIER_COLORS = {"Low": "#3b8b6e", "Medium": "#d9962b", "High": "#b4372c"}
TIER_ORDER = ["High", "Medium", "Low"]

# The UCI dataset is in GBP; money is converted for display only (models stay in GBP).
GBP_TO_INR = 110.0
MONEY_COLS = ["monetary", "avg_order_value", "future_revenue", "predicted_spend_if_retained",
              "expected_revenue", "revenue_at_risk"]

st.set_page_config(page_title="ChurnBoost AI", page_icon=str(ROOT / "assets" / "favicon.png"), layout="wide")
st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text()}</style>", unsafe_allow_html=True)

# ---- chart styling: white panel, no chrome, palette-matched text
sns.set_theme(style="white", rc={
    "figure.dpi": 200, "savefig.dpi": 200, "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": LINE, "axes.grid": True, "grid.color": "#efebe1", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.axisbelow": True, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def show(fig):
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def kpi_band(key: str, items: list[dict]):
    """Animated KPI row. Counts up from the value last shown under `key`."""
    prev = st.session_state.get(f"kpi_{key}", {})
    cells = []
    for it in items:
        it["from"] = prev.get(it["label"], 0)
        cells.append(it)
    st.session_state[f"kpi_{key}"] = {it["label"]: it["value"] for it in items}
    html = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700&family=Instrument+Sans:wght@500&display=swap');
html,body{margin:0;background:transparent;font-family:'Instrument Sans',system-ui,sans-serif}
.band{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));background:#fff;border:1px solid #e3ded2;border-radius:14px;overflow:hidden}
.k{padding:16px 22px 15px;border-right:1px solid #efebe1}.k:last-child{border-right:0}
.l{color:#667085;font-size:13px;margin-bottom:6px}
.v{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:26px;color:#1d2433;font-variant-numeric:tabular-nums;letter-spacing:-0.02em}
</style>
<div class="band" id="b"></div>
<script>
const items = __ITEMS__;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const ease = t => t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
const fmt = (it, x) => (it.prefix||'') + x.toLocaleString('en-IN', {minimumFractionDigits: it.decimals, maximumFractionDigits: it.decimals}) + (it.suffix||'');
const b = document.getElementById('b');
items.forEach(it => {
  const d = document.createElement('div'); d.className = 'k';
  d.innerHTML = '<div class="l"></div><div class="v"></div>';
  d.firstChild.textContent = it.label; b.appendChild(d);
  const v = d.lastChild, a = it.from, z = it.value;
  if (reduce || a === z) { v.textContent = fmt(it, z); return; }
  const t0 = performance.now(), dur = 900;
  (function f(t){ const p = Math.min((t - t0) / dur, 1); v.textContent = fmt(it, a + (z - a) * ease(p)); if (p < 1) requestAnimationFrame(f); })(t0);
});
</script>"""
    st.iframe(html.replace("__ITEMS__", json.dumps(cells)), height=96)


def meter(label: str, p: float, key: str, tier: str | None = None):
    """Risk meter whose fill grows from the previously shown probability."""
    frm = st.session_state.get(f"meter_{key}", 0.0)
    st.session_state[f"meter_{key}"] = p
    color = TIER_COLORS["High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"]
    pill = f'<span class="pill {tier}">{tier} risk</span>' if tier else ""
    st.markdown(f"""
<div class="meter-wrap">
  <div class="meter-top"><span class="meter-left"><span class="meter-label">{label}</span>{pill}</span>
  <span class="meter-value" style="color:{color}">{p:.1%}</span></div>
  <div class="meter"><div class="meter-fill" style="--p:{p:.4f};--from:{frm:.4f};background:{color}"></div></div>
  <div class="meter-ticks"><span style="left:40%">40%</span><span style="left:70%">70%</span></div>
</div>""", unsafe_allow_html=True)


@st.cache_data
def load():
    cust = pd.read_csv(ART / "customers_scored.csv")
    metrics = json.loads((ART / "metrics.json").read_text())
    cust[MONEY_COLS] = cust[MONEY_COLS] * GBP_TO_INR
    return cust, metrics


if not (ART / "metrics.json").exists():
    st.error("No trained artifacts found. Run `python -m src.train` first.")
    st.stop()

cust, m = load()
models = joblib.load(ART / "models.joblib")

st.markdown("""
<div class="cb-head">
  <svg width="52" height="52" viewBox="0 0 52 52" aria-hidden="true"><rect width="52" height="52" rx="13" fill="#25586b"/>
  <polyline points="11,17 21,28 28,22 41,37" fill="none" stroke="#f4f1ea" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="41" cy="37" r="4" fill="#e0a43a"/></svg>
  <div><h1 class="cb-title">ChurnBoost AI</h1>
  <p class="cb-sub">Predictive e-commerce churn pipeline · UCI Online Retail II · RFM + XGBoost</p></div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Filters")
    tiers = st.multiselect("Risk tier", TIER_ORDER, default=TIER_ORDER)
    min_m = st.slider("Min. historical spend (₹)", 0, int(cust["monetary"].quantile(0.99)), 0, step=1000)
    st.markdown("---")
    st.markdown(
        f"**Cutoff date:** {m['cutoff']}  \n"
        "Features use purchases *before* the cutoff. A customer is *churned* if they "
        "bought nothing in the 90 days after it.  \n"
        f"Amounts are shown in ₹ (converted from GBP at {GBP_TO_INR:g}).")
view = cust[cust["risk_tier"].isin(tiers) & (cust["monetary"] >= min_m)]

tab_over, tab_model, tab_seg, tab_cust = st.tabs(
    ["Overview", "Model performance", "High-risk segments", "Customer lookup"])

# ------------------------------------------------------------------ Overview
with tab_over:
    kpi_band("overview", [
        {"label": "Customers", "value": len(view), "decimals": 0},
        {"label": "Avg churn probability", "value": float(view["churn_prob"].mean() * 100) if len(view) else 0, "decimals": 1, "suffix": "%"},
        {"label": "High-risk customers", "value": int((view["risk_tier"] == "High").sum()), "decimals": 0},
        {"label": "Revenue at risk (90d)", "value": float(view["revenue_at_risk"].sum()), "decimals": 0, "prefix": "₹"},
    ])
    st.write("")
    l, r = st.columns(2)
    with l, st.container(key="panel_tiers"):
        st.subheader("Customers by risk tier")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        counts = view["risk_tier"].value_counts().reindex(TIER_ORDER).fillna(0)
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=TIER_COLORS,
                    legend=False, ax=ax, width=0.55)
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xlabel(""); ax.set_ylabel("Customers"); ax.grid(axis="x", visible=False)
        show(fig)
    with r, st.container(key="panel_rev"):
        st.subheader("Forecast revenue by risk tier")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        rev = view.groupby("risk_tier")[["expected_revenue", "revenue_at_risk"]].sum().reindex(TIER_ORDER)
        rev.plot(kind="bar", ax=ax, color=[ACCENT, TIER_COLORS["High"]], rot=0, width=0.7)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"₹{v/1e7:.1f}Cr"))
        ax.set_xlabel(""); ax.set_ylabel("Next 90 days (₹ crore)"); ax.grid(axis="x", visible=False)
        ax.legend(["Expected revenue", "Revenue at risk"])
        show(fig)

    st.write("")
    with st.container(key="panel_dist"):
        st.subheader("Churn probability distribution")
        fig, ax = plt.subplots(figsize=(10, 2.8))
        sns.histplot(view["churn_prob"], bins=40, kde=True, ax=ax, color=ACCENT, edgecolor="white", linewidth=0.4)
        for x in (0.4, 0.7):
            ax.axvline(x, ls=(0, (4, 3)), c=MUTED, lw=1)
        ax.set_xlabel("Churn probability"); ax.set_ylabel("Customers"); ax.grid(axis="x", visible=False)
        show(fig)

# ------------------------------------------------------------ Model performance
with tab_model:
    kpi_band("model", [{"label": lab, "value": m[k], "decimals": 3} for lab, k in
                       [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Precision", "precision"),
                        ("Recall", "recall"), ("F1", "f1")]])
    st.caption(f"Hold-out set (20%). Class weight `scale_pos_weight` tuned by CV PR-AUC → {m['scale_pos_weight']}. "
               f"Churn rate in data: {m['churn_rate']:.1%}.")

    l, r = st.columns(2)
    with l, st.container(key="panel_imp"):
        st.subheader("Feature importance")
        imp = pd.Series(m["feature_importance"]).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.barh(imp.index, imp.values, color=ACCENT, height=0.6)
        ax.grid(axis="y", visible=False)
        show(fig)
    with r, st.container(key="panel_cm"):
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 3.6))
        sns.heatmap(np.array(m["confusion_matrix"]), annot=True, fmt="d", cmap=sns.light_palette(ACCENT, as_cmap=True),
                    cbar=False, ax=ax, linewidths=3, linecolor="white", annot_kws={"size": 12})
        ax.set_xticklabels(["Retained", "Churned"]); ax.set_yticklabels(["Retained", "Churned"], rotation=0)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
        show(fig)

    st.write("")
    with st.container(key="panel_pr"):
        st.subheader("Precision-recall curve")
        p, rc, _ = precision_recall_curve(m["test_y"], m["test_proba"])
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(rc, p, color=ACCENT, lw=2.2); ax.fill_between(rc, p, alpha=0.08, color=ACCENT)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.grid(axis="x", visible=False)
        show(fig)

# ---------------------------------------------------------------- Segments
with tab_seg:
    with st.container(key="panel_rfm"):
        st.subheader("RFM view of customers, coloured by risk tier")
        sample = view.sample(min(len(view), 3000), random_state=1)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        sns.scatterplot(data=sample, x="recency", y="frequency", size="monetary", hue="risk_tier",
                        palette=TIER_COLORS, hue_order=TIER_ORDER, sizes=(10, 220), alpha=0.55,
                        edgecolor="white", linewidth=0.3, ax=ax)
        ax.set_yscale("log"); ax.set_xlabel("Recency (days since last purchase)")
        ax.set_ylabel("Frequency (orders, log)")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        show(fig)

    st.write("")
    st.subheader("Average RFM profile per tier")
    st.dataframe(view.groupby("risk_tier")[["recency", "frequency", "monetary", "churn_prob"]]
                 .mean().reindex([t for t in TIER_ORDER if t in view["risk_tier"].unique()]).round(2))

    st.subheader("High-value customers at risk")
    st.caption("Top-quartile historical spend and high churn probability, ranked by revenue at risk.")
    hv = (view[view["high_value_at_risk"]]
          .sort_values("revenue_at_risk", ascending=False)
          [["Customer ID", "recency", "frequency", "monetary", "churn_prob", "revenue_at_risk"]])
    st.dataframe(hv.round(2), width="stretch", hide_index=True)
    st.download_button("Download high-risk list (CSV)", hv.to_csv(index=False), "high_value_at_risk.csv")

# ---------------------------------------------------------------- Lookup
with tab_cust:
    cid = st.selectbox("Customer ID", cust.sort_values("churn_prob", ascending=False)["Customer ID"])
    row = cust[cust["Customer ID"] == cid].iloc[0]
    meter("Churn probability", float(row["churn_prob"]), "cust", tier=row["risk_tier"])
    kpi_band("cust", [
        {"label": "Expected revenue (90d)", "value": float(row["expected_revenue"]), "decimals": 0, "prefix": "₹"},
        {"label": "Revenue at risk", "value": float(row["revenue_at_risk"]), "decimals": 0, "prefix": "₹"},
        {"label": "Historical spend", "value": float(row["monetary"]), "decimals": 0, "prefix": "₹"},
        {"label": "Orders", "value": int(row["frequency"]), "decimals": 0},
    ])
    st.write("")
    st.dataframe(row[models["features"]].to_frame("value").round(2))

    st.subheader("What-if: change this customer's behaviour")
    w = st.columns(3)
    rec = w[0].slider("Recency (days)", 0, int(cust["recency"].max()), int(row["recency"]))
    frq = w[1].slider("Frequency (orders)", 1, int(cust["frequency"].quantile(0.99)), int(row["frequency"]))
    mon = w[2].slider("Monetary (₹)", 0, int(cust["monetary"].quantile(0.99)), int(row["monetary"]), step=500)
    x = row[models["features"]].astype(float).copy()
    # the model was trained on GBP, so convert the rupee inputs back
    x["recency"], x["frequency"], x["monetary"] = rec, frq, mon / GBP_TO_INR
    x["avg_order_value"] = mon / GBP_TO_INR / max(frq, 1)
    p_new = float(models["clf"].predict_proba(pd.DataFrame([x]))[0, 1])
    meter("Simulated churn probability", p_new, "sim")
    d = p_new - float(row["churn_prob"])
    st.markdown(f'Change vs. current: <span class="{"delta-up" if d > 0 else "delta-down"}">{d:+.1%}</span>',
                unsafe_allow_html=True)
