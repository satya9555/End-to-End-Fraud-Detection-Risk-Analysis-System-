"""
Run the full fraud-detection pipeline and write all artifacts to ./outputs.

Usage:
    python run_pipeline.py --data data/creditcard.csv
    python run_pipeline.py --synthetic --fast      # code smoke test ONLY
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

import fraud_pipeline as fp


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--synthetic", action="store_true",
                    help="use synthetic data (smoke test only; results are NOT real)")
    ap.add_argument("--fast", action="store_true", help="smaller search for quick runs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ---- load & inspect
    df = fp.make_synthetic_data() if a.synthetic else fp.load_data(a.data)
    source = "SYNTHETIC (smoke test - not real fraud data)" if a.synthetic else a.data
    info = fp.inspect_data(df)
    print("Data inspection:", json.dumps(info, indent=2, default=str))
    df = fp.engineer_features(fp.clean_data(df))
    clean_info = fp.inspect_data(df.drop(columns=["log_amount", "hour_sin", "hour_cos",
                                                   "is_night", "hour_of_day"]))

    # ---- EDA
    eda = fp.run_eda(df, out)

    # ---- split
    s = fp.split_data(df)
    print(f"Split sizes train/val/test = {len(s.X_train)}/{len(s.X_val)}/{len(s.X_test)}; "
          f"frauds = {int(s.y_train.sum())}/{int(s.y_val.sum())}/{int(s.y_test.sum())}")

    # ---- imbalance strategies, models
    imb = fp.compare_imbalance_strategies(s, out, a.fast)
    print(imb.to_string(index=False))
    comp, fitted, vsc, tsc = fp.train_and_evaluate(s, out, a.fast)
    fp.plot_model_curves(s, tsc, comp, out)

    final = fp.select_final_model(comp)
    thr = float(comp[(comp.model == final) & (comp.split == "validation")].threshold.iloc[0])
    print(f"Final model (by validation PR-AUC): {final}, threshold={thr:.4f}")

    # ---- error analysis, importance, risk scoring
    raw_test = s.X_test.copy()
    raw_test["Amount"] = (pd.Series(raw_test["log_amount"]).apply(lambda v: 2.718281828 ** v - 1))
    err = fp.error_analysis(s, final, tsc[final], thr, raw_test, out)
    imp = fp.feature_importance(fitted[final], s, out, n_repeats=3 if a.fast else 5)
    cfg = fp.RiskConfig()
    dist = fp.risk_distribution(s, tsc[final], cfg, out)

    joblib.dump({"model": fitted[final], "name": final, "threshold": thr,
                 "features": s.feature_names}, out / "final_model.joblib")
    # artifacts used by the Streamlit dashboard
    comp.to_csv(out / "model_comparison_all.csv", index=False)
    pd.DataFrame({"y_true": s.y_test.values, "score": tsc[final]}).to_csv(
        out / "test_scores_final_model.csv", index=False)

    # ---- auto-generated report (all numbers come from the run above)
    test = comp[comp.split == "test"].drop(columns="split").round(4)
    val = comp[comp.split == "validation"].drop(columns="split").round(4)
    t05 = comp[comp.split == "test@0.5"].drop(columns="split").round(4)
    row = comp[(comp.model == final) & (comp.split == "test")].iloc[0]
    report = f"""# Fraud Detection & Risk Analysis - Auto-generated Report

**Data source:** {source}
{"> **WARNING: synthetic data. None of the figures below describe real fraud.**" if a.synthetic else ""}

All values below were measured by `run_pipeline.py`. Re-run to reproduce (seed={fp.SEED}).

## 1. Dataset overview & quality
- Shape (raw): {info['shape']}, dtypes: {info['dtypes']}
- Missing values: {info['missing_total']}; exact duplicates: {info['duplicates']} (removed before splitting)
- Zero-amount rows: {info['zero_amounts']}; negative amounts: {info['negative_amounts']}
- Legitimate: {info['n_legit']} ({info['pct_legit']:.4f}%) | Fraud: {info['n_fraud']} ({info['pct_fraud']:.4f}%)
- After cleaning: {clean_info['shape'][0]} rows, fraud {clean_info['pct_fraud']:.4f}%

## 2. EDA highlights (see PNGs in outputs/)
- Features most correlated with Class: {eda['top_corr_with_target']}
- Share of fraud above the legitimate 99th-percentile amount ({eda['legit_p99_amount']:.2f}): {eda['fraud_share_above_legit_p99_amount']:.3f}
- Top Mann-Whitney effects: see `eda_stat_tests.csv` (Bonferroni-adjusted p-values included)
- Hourly fraud rates: `eda_time.png` (Time is an offset from the first transaction; the "hour" is relative, not clock time)

## 3. Preprocessing
Log amount, cyclic hour features, night flag; stratified 60/20/20 split; median imputation and scaling are inside
pipelines (fit on training data only). Resampling is applied only to training data.

## 4. Imbalance strategies (HistGradientBoosting, validation set)
{md_table(imb)}

## 5. Model comparison
### Validation (threshold chosen here to maximise F1)
{md_table(val)}
### Held-out TEST (same validation threshold)
{md_table(test)}
### Held-out TEST at the default 0.5 threshold (for contrast)
{md_table(t05)}

Accuracy is misleading here: a model predicting "legitimate" for everything achieves
{info['pct_legit']:.3f}% accuracy while catching zero fraud. Use PR-AUC, recall and precision.

## 6. Final model
**{final}** (selected by validation PR-AUC; threshold={thr:.4f}).
Test: precision={row.precision:.4f}, recall={row.recall:.4f}, F1={row.f1:.4f}, ROC-AUC={row.roc_auc:.4f}, PR-AUC={row.pr_auc:.4f}
Confusion matrix (test): TN={row.TN}, FP={row.FP}, FN={row.FN}, TP={row.TP}

## 7. False positives / false negatives (test)
{json.dumps(err, indent=2)}

Threshold trade-off: `threshold_tradeoff_test.csv`. Lowering the threshold raises recall and alert volume;
raising it reduces customer friction but misses more fraud.

## 8. Feature importance
{json.dumps(imp, indent=2, default=str)}

Importances show statistical association with the model's predictions, **not causes**. V1..V28 are anonymised
PCA components and cannot be interpreted as business concepts.

## 9. Risk categories on the test set (thresholds: medium={cfg.medium}, high={cfg.high})
{md_table(dist.reset_index())}

Scores are ranking scores, not calibrated probabilities. A flagged transaction triggers review or step-up
authentication; it is never treated as confirmed fraud.

## 10. Limitations, privacy, bias
- Only ~2 days of European card data (2013); concept drift is likely and fraud tactics evolve.
- Very few fraud cases -> metrics have wide uncertainty; consider bootstrap CIs and repeated splits.
- Features are anonymised; no merchant/geography/customer data so bias across customer groups cannot be assessed.
- A random split ignores time order; a temporal split would be a stricter test.
- No real cardholder data is exposed; dashboard demos use synthetic inputs.

## 11. Recommendations
1. Use tiered actions (approve / step-up auth / manual review) instead of hard declines.
2. Choose thresholds from an explicit cost model (cost of missed fraud vs. cost of friction/review).
3. Monitor weekly: alert rate, precision of reviewed alerts, recall on confirmed fraud, PR-AUC, score/feature drift (PSI).
4. Retrain regularly on fresh labelled data; calibrate scores (isotonic/Platt) if probabilities are needed.
5. Add behavioural features (velocity, merchant, device, geography) in production.
6. Audit performance across customer segments for bias before deployment.
"""
    (out / "final_report.md").write_text(report, encoding="utf-8")
    print(f"Done. Report and artifacts written to {out.resolve()}")


if __name__ == "__main__":
    main()
