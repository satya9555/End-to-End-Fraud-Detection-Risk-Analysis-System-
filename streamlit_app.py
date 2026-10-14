"""Educational Streamlit dashboard. Run `python run_pipeline.py` first.
    streamlit run streamlit_app.py
Uses artifacts in ./outputs. Manual demos use SYNTHETIC inputs only."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve

import fraud_pipeline as fp

OUT = Path("outputs")
st.set_page_config(page_title="Fraud Risk Dashboard", layout="wide")
st.title("Credit Card Fraud Detection & Risk Analysis (educational)")

if not (OUT / "final_model.joblib").exists():
    st.error("No artifacts found. Run `python run_pipeline.py` first.")
    st.stop()

bundle = joblib.load(OUT / "final_model.joblib")
comp = pd.read_csv(OUT / "model_comparison_all.csv")
ts = pd.read_csv(OUT / "test_scores_final_model.csv")

st.sidebar.header("Risk thresholds (configurable)")
medium = st.sidebar.slider("Medium risk >=", 0.0, 1.0, 0.30, 0.01)
high = st.sidebar.slider("High risk >=", 0.0, 1.0, 0.70, 0.01)
cfg = fp.RiskConfig(medium=medium, high=max(high, medium))
decision_thr = st.sidebar.slider("Alert threshold", 0.0, 1.0,
                                 float(min(max(bundle["threshold"], 0.0), 1.0)), 0.01)
st.sidebar.caption("Scores are ranking scores, not calibrated fraud probabilities.")

pred = (ts.score >= decision_thr).astype(int)
tn, fpos, fn, tp = confusion_matrix(ts.y_true, pred, labels=[0, 1]).ravel()

c = st.columns(4)
c[0].metric("Test transactions analysed", f"{len(ts):,}")
c[1].metric("Flagged for review", f"{int(pred.sum()):,}", f"{100*pred.mean():.3f}% of transactions")
c[2].metric("Actual frauds in test", f"{int(ts.y_true.sum()):,}")
c[3].metric("Frauds caught (recall)", f"{tp/max(tp+fn,1):.1%}")
st.caption(f"Final model: **{bundle['name']}** (selected on validation PR-AUC). "
           "Flagged transactions are NOT confirmed fraud.")

l, r = st.columns(2)
with l:
    st.subheader("Fraud risk distribution (test)")
    cats = ts.score.map(cfg.categorize)
    st.bar_chart(cats.value_counts().reindex(["Low", "Medium", "High"]).fillna(0))
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow([[tn, fpos], [fn, tp]], cmap="Blues")
    for (i, j), v in np.ndenumerate([[tn, fpos], [fn, tp]]):
        ax.text(j, i, int(v), ha="center", va="center")
    ax.set_xticks([0, 1], ["Pred legit", "Pred fraud"]); ax.set_yticks([0, 1], ["Legit", "Fraud"])
    st.pyplot(fig)
with r:
    st.subheader("Precision-recall curve")
    p, rc, _ = precision_recall_curve(ts.y_true, ts.score)
    fig, ax = plt.subplots(figsize=(5, 3.5)); ax.plot(rc, p)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); st.pyplot(fig)
    st.subheader("Model performance (test set)")
    st.dataframe(comp[comp.split == "test"].drop(columns="split").round(4), use_container_width=True)

st.subheader("Important features")
for f in ("permutation_importance.csv", "model_feature_importance.csv"):
    if (OUT / f).exists():
        st.write(f.replace("_", " ").replace(".csv", ""))
        st.dataframe(pd.read_csv(OUT / f).head(10))

st.subheader("Sample transaction risk assessment (synthetic input)")
st.caption("Move the sliders to build a made-up transaction; no real cardholder data is used.")
amt = st.number_input("Amount", 0.0, 50000.0, 120.0)
hour = st.slider("Hour of day", 0, 23, 14)
v_vals = {f: st.sidebar.slider(f, -6.0, 6.0, 0.0, 0.1)
          for f in ["V1", "V3", "V4", "V10", "V14", "V17"]}
row = fp.synthetic_demo_transaction(seed=7) * 0.0
for k, v in v_vals.items():
    row[k] = v
row["log_amount"] = np.log1p(amt)
row["hour_sin"], row["hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
row["is_night"] = int(hour < 6)
row = row[bundle["features"]]
res = fp.score_transactions(bundle["model"], row, cfg).iloc[0]
st.metric("Risk score", f"{res.risk_score:.3f}", res.risk_category)
st.info(f"Suggested action: {res.recommended_action}")
