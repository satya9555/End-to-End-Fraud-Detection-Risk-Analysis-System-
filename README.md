# Credit Card Fraud Detection & Risk Analysis

End-to-end pipeline: data inspection, EDA, imbalance strategies, 6 models (LogReg, Decision Tree, Random Forest,
HistGradientBoosting, XGBoost, Isolation Forest), tuning, validation-based thresholds, held-out test evaluation,
FP/FN analysis, feature importance, risk scoring, Streamlit dashboard, and an auto-generated report.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
XGBoost, SHAP and imbalanced-learn are optional; the code skips the parts that need them if missing.

## Dataset
1. Download `creditcard.csv` from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (Kaggle login required).
2. Put it at `data/creditcard.csv`.

Columns: `Time` (seconds since first transaction), `V1..V28` (anonymised PCA components), `Amount`, `Class` (1 = fraud).

## Run
```bash
python run_pipeline.py --data data/creditcard.csv     # full run -> outputs/ (plots, CSVs, final_report.md, model)
python run_pipeline.py --synthetic --fast             # smoke test on fake data (results are NOT real)
streamlit run streamlit_app.py                        # dashboard (after the pipeline has run)
python build_notebook.py && jupyter notebook fraud_detection.ipynb
```

## Files
| File | Purpose |
|---|---|
| `fraud_pipeline.py` | All reusable logic (data, EDA, models, evaluation, risk scoring) |
| `run_pipeline.py` | Runs everything, writes `outputs/` and `final_report.md` |
| `streamlit_app.py` | Interactive dashboard |
| `build_notebook.py` | Generates `fraud_detection.ipynb` |

## Methodology notes
- Duplicates removed before splitting; stratified 60/20/20 split; seed 42.
- Imputation/scaling inside pipelines (train-fitted only). Resampling only on training data.
- Tuning: RandomizedSearchCV on train with `average_precision`. Threshold (max F1) chosen on validation. Final model
  chosen by validation PR-AUC. Test set used once for reporting.
- Reported scores are measured, never hard-coded. Validation / test / test@0.5 are labelled separately.
- Risk scores are ranking scores, not calibrated probabilities. Flags mean "review", not "confirmed fraud".
- Importances are associations, not causes; V1..V28 are anonymised.

## Reproducibility
Fixed seeds; library versions affect exact numbers. Run time on the full dataset (284k rows) can be tens of minutes;
use `--fast` to shorten tuning.
