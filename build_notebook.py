"""Generate fraud_detection.ipynb (step-by-step walkthrough using fraud_pipeline.py)."""
import json

try:
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    M = lambda t: nbf.v4.new_markdown_cell(t)
    C = lambda t: nbf.v4.new_code_cell(t)
    _write = lambda nb, path: nbf.write(nb, path)
except ImportError:  # fallback: write the .ipynb JSON directly (nbformat v4)
    class _NB:
        cells = []
    nb = _NB()
    _src = lambda t: t.splitlines(keepends=True)
    M = lambda t: {"cell_type": "markdown", "metadata": {}, "source": _src(t)}
    C = lambda t: {"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": _src(t)}

    def _write(nb, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"cells": nb.cells, "nbformat": 4, "nbformat_minor": 5,
                       "metadata": {"kernelspec": {"display_name": "Python 3",
                                                   "language": "python", "name": "python3"}}},
                      f, indent=1)

nb.cells = [
M("# Credit Card Fraud Detection & Risk Analysis\nEvery number is computed when you run the cell. "
  "Set `USE_SYNTHETIC=True` only to test the code (results are then NOT real)."),
C("""from pathlib import Path
import pandas as pd, numpy as np, joblib
import fraud_pipeline as fp
USE_SYNTHETIC = False      # True = smoke test on fake data
DATA = 'data/creditcard.csv'
FAST = False
OUT = Path('outputs'); OUT.mkdir(exist_ok=True)"""),
M("## 1-2. Load, inspect, clean\nTarget `Class`: 1 = fraud. `V1..V28` are anonymised PCA components; `Time` = seconds since first transaction; `Amount` = transaction amount."),
C("""df = fp.make_synthetic_data() if USE_SYNTHETIC else fp.load_data(DATA)
info = fp.inspect_data(df); info"""),
C("""df = fp.engineer_features(fp.clean_data(df))
print(df.shape); df[['Amount','log_amount','hour_of_day','is_night']].describe()"""),
M("## 3. EDA\nEach saved plot is shown below; interpret it using the measured insights printed."),
C("""eda = fp.run_eda(df, OUT)
from IPython.display import Image, display
for f in ['eda_class_balance','eda_amount','eda_time','eda_corr_heatmap','eda_feature_dists']:
    display(Image(OUT/f'{f}.png'))
eda['top_corr_with_target'], eda['fraud_share_above_legit_p99_amount']"""),
C("pd.read_csv(OUT/'eda_stat_tests.csv').head(10)"),
M("## 4. Split and class-imbalance strategies\nResampling touches the training data only."),
C("""s = fp.split_data(df)
print(len(s.X_train), len(s.X_val), len(s.X_test), s.y_train.sum(), s.y_val.sum(), s.y_test.sum())
imb = fp.compare_imbalance_strategies(s, OUT, FAST); imb"""),
M("## 5-6. Models, tuning, evaluation\nTuning on train (CV, PR-AUC); threshold chosen on validation; test set is held out."),
C("""comp, fitted, vsc, tsc = fp.train_and_evaluate(s, OUT, FAST)
fp.plot_model_curves(s, tsc, comp, OUT)
comp[comp.split=='validation'].round(4)"""),
C("comp[comp.split=='test'].round(4)"),
C("""for f in ['model_roc_pr_curves','model_metric_bars','model_confusion_matrices']:
    display(Image(OUT/f'{f}.png'))"""),
C("""final = fp.select_final_model(comp)
thr = float(comp[(comp.model==final)&(comp.split=='validation')].threshold.iloc[0])
final, thr"""),
M("## 7. False positives / false negatives"),
C("""raw_test = s.X_test.copy(); raw_test['Amount'] = np.expm1(raw_test['log_amount'])
err = fp.error_analysis(s, final, tsc[final], thr, raw_test, OUT)
display(Image(OUT/'error_analysis.png')); err"""),
C("pd.read_csv(OUT/'threshold_tradeoff_test.csv')"),
M("## 8. Feature importance (association, not causation)"),
C("""imp = fp.feature_importance(fitted[final], s, OUT)
display(Image(OUT/'feature_importance_permutation.png')); imp['permutation_top10']"""),
M("## 9. Risk scoring\nScores are ranking scores, not calibrated probabilities. 'Flagged' means review/step-up, not confirmed fraud."),
C("""cfg = fp.RiskConfig(medium=0.30, high=0.70)
fp.risk_distribution(s, tsc[final], cfg, OUT)"""),
C("""sample = s.X_test.head(5)
fp.score_transactions(fitted[final], sample, cfg)"""),
M("## 10-11. Save artifacts, dashboard and report\nRun `python run_pipeline.py` to generate `outputs/final_report.md` and then `streamlit run streamlit_app.py`."),
C("""joblib.dump({'model': fitted[final], 'name': final, 'threshold': thr, 'features': s.feature_names}, OUT/'final_model.joblib')"""),
]
_write(nb, "fraud_detection.ipynb")
print("Wrote fraud_detection.ipynb")
