"""
Credit Card Fraud Detection & Risk Analysis - core pipeline module.

Every number reported by this project is computed from the data at run time.
Nothing is hard-coded or invented.

Sections (match the project prompt):
  2. load / inspect / clean / feature engineering / splitting
  3. EDA
  4. class imbalance strategies
  5-6. models, tuning, threshold selection, evaluation
  7. FP / FN analysis
  8. feature importance (permutation, model-based, SHAP if available)
  9. risk scoring
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:  # optional dependencies
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except Exception:  # pragma: no cover
    HAS_IMBLEARN = False
try:
    import shap
    HAS_SHAP = True
except Exception:  # pragma: no cover
    HAS_SHAP = False

SEED = 42
TARGET = "Class"
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------- 2. DATA ---
def load_data(path: str | os.PathLike) -> pd.DataFrame:
    """Load the Kaggle creditcard.csv with clear error messages."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place "
            "it there (see README), or run with --synthetic for a smoke test.")
    df = pd.read_csv(path)
    missing_cols = {TARGET, "Time", "Amount"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    return df


def make_synthetic_data(n: int = 30000, fraud_rate: float = 0.004,
                        seed: int = SEED) -> pd.DataFrame:
    """Synthetic stand-in with the same schema as the Kaggle dataset
    (Time, V1..V28, Amount, Class). FOR CODE TESTING ONLY - results computed
    on it say nothing about real fraud."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    X = rng.normal(size=(n, 28))
    shift = np.zeros(28)
    shift[[0, 2, 3, 9, 13, 16]] = [-2.5, -3, 2, -2.5, -3, -2]
    X[y == 1] += shift + rng.normal(scale=0.5, size=(y.sum(), 28))
    amount = np.where(y == 1, rng.lognormal(3.5, 1.6, n), rng.lognormal(3.3, 1.2, n))
    time = np.sort(rng.uniform(0, 172800, n))
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = amount.round(2)
    df[TARGET] = y
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """Shape, dtypes, missing, duplicates, inconsistent values, class balance."""
    counts = df[TARGET].value_counts().sort_index()
    info = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).value_counts().to_dict(),
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "negative_amounts": int((df["Amount"] < 0).sum()),
        "zero_amounts": int((df["Amount"] == 0).sum()),
        "non_binary_target": int((~df[TARGET].isin([0, 1])).sum()),
        "n_legit": int(counts.get(0, 0)),
        "n_fraud": int(counts.get(1, 0)),
        "pct_fraud": float(100 * counts.get(1, 0) / len(df)),
        "pct_legit": float(100 * counts.get(0, 0) / len(df)),
    }
    return info


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates, invalid targets and negative amounts.
    (Duplicates are removed BEFORE splitting so copies cannot leak across sets.)"""
    out = df.drop_duplicates().copy()
    out = out[out[TARGET].isin([0, 1])]
    out = out[out["Amount"] >= 0]
    out[TARGET] = out[TARGET].astype(int)
    return out.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise (leak-free) features: log amount and time-of-day features.
    Time is seconds since the first transaction (dataset spans ~2 days)."""
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"])
    out["hour_of_day"] = ((out["Time"] // 3600) % 24).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24)
    out["is_night"] = out["hour_of_day"].isin(range(0, 6)).astype(int)
    return out


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: list = field(default_factory=list)


def split_data(df: pd.DataFrame) -> Splits:
    """Stratified 60/20/20 train/val/test split. Scaling/imputation live inside
    sklearn Pipelines (fit on train only) -> no leakage.
    'Time' and 'Amount' raw columns are dropped in favour of engineered ones."""
    X = df.drop(columns=[TARGET, "Time", "Amount", "hour_of_day"])
    y = df[TARGET]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)  # 0.25*0.8=0.2
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test, list(X.columns))


# ------------------------------------------------------------------ 3. EDA ---
def run_eda(df: pd.DataFrame, outdir: Path) -> dict:
    """Produce EDA plots + statistical tests. Returns measured insights."""
    outdir.mkdir(parents=True, exist_ok=True)
    insights = {}
    labels = df[TARGET].map({0: "Legitimate", 1: "Fraud"})

    # 1) class imbalance
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(x=labels, ax=ax[0], order=["Legitimate", "Fraud"])
    ax[0].set_title("Class counts (linear scale)")
    sns.countplot(x=labels, ax=ax[1], order=["Legitimate", "Fraud"])
    ax[1].set_yscale("log")
    ax[1].set_title("Class counts (log scale)")
    fig.tight_layout(); fig.savefig(outdir / "eda_class_balance.png", dpi=120); plt.close(fig)

    # 2) amount distributions by class
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=df, x="log_amount", hue=labels, stat="density",
                 common_norm=False, bins=50, ax=ax[0])
    ax[0].set_title("log(1+Amount) by class (density)")
    sns.boxplot(data=df, x=labels, y="Amount", ax=ax[1], showfliers=False,
                order=["Legitimate", "Fraud"])
    ax[1].set_title("Amount by class (outliers hidden)")
    fig.tight_layout(); fig.savefig(outdir / "eda_amount.png", dpi=120); plt.close(fig)

    # 3) time patterns
    hourly = df.groupby("hour_of_day")[TARGET].agg(["count", "sum", "mean"])
    hourly.columns = ["transactions", "frauds", "fraud_rate"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(hourly.index, hourly["transactions"]); ax[0].set_title("Transactions per hour of day")
    ax[0].set_xlabel("Hour (derived from Time offset)")
    ax[1].bar(hourly.index, 100 * hourly["fraud_rate"], color="firebrick")
    ax[1].set_title("Fraud rate (%) per hour of day"); ax[1].set_xlabel("Hour")
    fig.tight_layout(); fig.savefig(outdir / "eda_time.png", dpi=120); plt.close(fig)
    insights["hourly_fraud_table"] = hourly.round(5).reset_index().to_dict("records")

    # 4) correlation with target + heatmap of the most correlated features
    feats = [c for c in df.columns if c not in (TARGET, "hour_of_day")]
    corr_t = df[feats + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
    top = list(corr_t.head(12).index)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[top + [TARGET]].corr(), cmap="coolwarm", center=0, ax=ax,
                annot=True, fmt=".2f", annot_kws={"size": 7})
    ax.set_title("Correlation heatmap: 12 features most correlated with Class")
    fig.tight_layout(); fig.savefig(outdir / "eda_corr_heatmap.png", dpi=120); plt.close(fig)
    insights["top_corr_with_target"] = corr_t.head(12).round(4).to_dict()

    # 5) feature distributions by class for the top-6 features
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, f in zip(axes.ravel(), top[:6]):
        sns.kdeplot(data=df, x=f, hue=labels, common_norm=False, ax=ax, warn_singular=False)
        ax.set_title(f)
    fig.tight_layout(); fig.savefig(outdir / "eda_feature_dists.png", dpi=120); plt.close(fig)

    # 6) statistical comparison (Mann-Whitney U + effect size AUC-like)
    fr, lg = df[df[TARGET] == 1], df[df[TARGET] == 0]
    rows = []
    for f in feats:
        u, p = stats.mannwhitneyu(fr[f], lg[f], alternative="two-sided")
        rows.append({"feature": f, "fraud_median": fr[f].median(),
                     "legit_median": lg[f].median(),
                     "rank_biserial": 1 - 2 * u / (len(fr) * len(lg)),
                     "p_value": p})
    st = pd.DataFrame(rows).sort_values("rank_biserial", key=np.abs, ascending=False)
    st["p_bonferroni"] = np.minimum(st["p_value"] * len(st), 1.0)
    st.to_csv(outdir / "eda_stat_tests.csv", index=False)
    insights["stat_tests_top10"] = st.head(10).round(5).to_dict("records")
    insights["amount_summary"] = df.groupby(labels)["Amount"].describe().round(2).to_dict("index")

    # 7) unusual-pattern indicator: share of frauds above the legit 99th pct amount
    p99 = lg["Amount"].quantile(0.99)
    insights["fraud_share_above_legit_p99_amount"] = float((fr["Amount"] > p99).mean())
    insights["legit_p99_amount"] = float(p99)
    return insights


# -------------------------------------------------- 4/5. MODELS & IMBALANCE ---
def _preproc(scale: bool = True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return steps


def build_models(scale_pos_weight: float) -> dict:
    """Model zoo with class-weighted learning built in (strategy: 'class_weight')."""
    models = {
        "Logistic Regression": Pipeline(_preproc(True) + [("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED))]),
        "Decision Tree": Pipeline(_preproc(False) + [("clf", DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=SEED))]),
        "Random Forest": Pipeline(_preproc(False) + [("clf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", n_jobs=-1,
            random_state=SEED))]),
        "HistGradientBoosting": Pipeline(_preproc(False) + [("clf", HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED))]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline(_preproc(False) + [("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            n_jobs=-1, random_state=SEED, verbosity=0))])
    return models


PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
    "Decision Tree": {"clf__max_depth": [3, 5, 8, 12], "clf__min_samples_leaf": [1, 5, 20]},
    "Random Forest": {"clf__max_depth": [8, 12, None], "clf__min_samples_leaf": [1, 3, 5],
                      "clf__max_features": ["sqrt", 0.3]},
    "HistGradientBoosting": {"clf__learning_rate": [0.03, 0.1, 0.2],
                             "clf__max_depth": [None, 4, 8],
                             "clf__l2_regularization": [0.0, 1.0]},
    "XGBoost": {"clf__max_depth": [3, 4, 6], "clf__learning_rate": [0.03, 0.1, 0.2],
                "clf__n_estimators": [200, 400]},
}


def compare_imbalance_strategies(s: Splits, outdir: Path, fast: bool = False) -> pd.DataFrame:
    """Same base model (HistGradientBoosting) under different imbalance strategies.
    Resampling is applied to the TRAINING data only; validation set is untouched."""
    base = HistGradientBoostingClassifier(random_state=SEED)
    strategies = {
        "None (baseline)": (None, None),
        "Class-weighted": (None, "balanced"),
    }
    if HAS_IMBLEARN:
        strategies["Random oversampling"] = (RandomOverSampler(random_state=SEED), None)
        strategies["Random undersampling"] = (RandomUnderSampler(random_state=SEED), None)
        if s.y_train.sum() > 6:
            strategies["SMOTE"] = (SMOTE(random_state=SEED), None)
    imp = SimpleImputer(strategy="median").fit(s.X_train)
    Xtr, Xva = imp.transform(s.X_train), imp.transform(s.X_val)
    rows = []
    for name, (sampler, cw) in strategies.items():
        Xr, yr = (sampler.fit_resample(Xtr, s.y_train) if sampler is not None
                  else (Xtr, s.y_train))
        m = clone(base).set_params(class_weight=cw).fit(Xr, yr)
        p = m.predict_proba(Xva)[:, 1]
        thr, _ = best_f1_threshold(s.y_val, p)
        pred = (p >= thr).astype(int)
        rows.append({"strategy": name, "train_size": len(yr),
                     "train_fraud_pct": 100 * float(np.mean(yr)),
                     "val_PR_AUC": average_precision_score(s.y_val, p),
                     "val_ROC_AUC": roc_auc_score(s.y_val, p),
                     "val_precision@bestF1thr": precision_score(s.y_val, pred, zero_division=0),
                     "val_recall@bestF1thr": recall_score(s.y_val, pred, zero_division=0),
                     "val_F1@bestF1thr": f1_score(s.y_val, pred, zero_division=0)})
    res = pd.DataFrame(rows).round(4)
    res.to_csv(outdir / "imbalance_strategy_comparison.csv", index=False)
    return res


def best_f1_threshold(y_true, scores) -> tuple[float, float]:
    """Pick threshold maximising F1 on the given (validation) data."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])


def tune_model(name: str, pipe: Pipeline, s: Splits, fast: bool = False) -> Pipeline:
    """RandomizedSearchCV on TRAIN only, scored with average precision (PR-AUC)."""
    grid = PARAM_GRIDS.get(name)
    if not grid:
        return pipe.fit(s.X_train, s.y_train)
    n_splits = int(min(3 if fast else 5, max(2, s.y_train.sum())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    n_iter = min(4 if fast else 10, int(np.prod([len(v) for v in grid.values()])))
    search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring="average_precision",
                                cv=cv, random_state=SEED, n_jobs=1, refit=True)
    search.fit(s.X_train, s.y_train)
    print(f"  [{name}] best CV PR-AUC={search.best_score_:.4f} params={search.best_params_}")
    return search.best_estimator_


def metrics_at(y_true, scores, thr) -> dict:
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"threshold": thr, "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, scores),
            "pr_auc": average_precision_score(y_true, scores),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def train_and_evaluate(s: Splits, outdir: Path, fast: bool = False):
    """Tune each model on train; choose threshold on VALIDATION; report both
    validation and held-out TEST metrics. Returns (comparison_df, fitted, scores)."""
    n_neg, n_pos = int((s.y_train == 0).sum()), int((s.y_train == 1).sum())
    models = build_models(scale_pos_weight=n_neg / max(n_pos, 1))
    fitted, val_scores, test_scores, rows = {}, {}, {}, []
    for name, pipe in models.items():
        print(f"Training {name} ...")
        best = tune_model(name, pipe, s, fast)
        pv, pt = best.predict_proba(s.X_val)[:, 1], best.predict_proba(s.X_test)[:, 1]
        thr, _ = best_f1_threshold(s.y_val, pv)
        fitted[name], val_scores[name], test_scores[name] = best, pv, pt
        for split, y, sc in (("validation", s.y_val, pv), ("test", s.y_test, pt)):
            rows.append({"model": name, "split": split, **metrics_at(y, sc, thr)})
        # default 0.5 threshold on test, to show why thresholds matter
        rows.append({"model": name, "split": "test@0.5", **metrics_at(s.y_test, pt, 0.5)})

    # optional anomaly detector (unsupervised; fit on legit-dominant train data)
    print("Training Isolation Forest ...")
    iso = Pipeline(_preproc(False) + [("clf", IsolationForest(
        n_estimators=200, contamination=max(float(s.y_train.mean()), 1e-4),
        random_state=SEED, n_jobs=-1))]).fit(s.X_train)
    sv, st_ = -iso.decision_function(s.X_val), -iso.decision_function(s.X_test)
    thr, _ = best_f1_threshold(s.y_val, sv)
    fitted["Isolation Forest"], val_scores["Isolation Forest"], test_scores["Isolation Forest"] = iso, sv, st_
    for split, y, sc in (("validation", s.y_val, sv), ("test", s.y_test, st_)):
        rows.append({"model": "Isolation Forest", "split": split, **metrics_at(y, sc, thr)})

    comp = pd.DataFrame(rows)
    comp.to_csv(outdir / "model_comparison_all.csv", index=False)
    return comp, fitted, val_scores, test_scores


def plot_model_curves(s: Splits, test_scores: dict, comp: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, sc in test_scores.items():
        fpr, tpr, _ = roc_curve(s.y_test, sc)
        ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(s.y_test, sc):.3f})")
        p, r, _ = precision_recall_curve(s.y_test, sc)
        ax[1].plot(r, p, label=f"{name} ({average_precision_score(s.y_test, sc):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=.7); ax[0].set_title("ROC curves (TEST set)")
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend(fontsize=8)
    ax[1].axhline(s.y_test.mean(), color="k", ls="--", lw=.7, label="No-skill")
    ax[1].set_title("Precision-Recall curves (TEST set)")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "model_roc_pr_curves.png", dpi=120); plt.close(fig)

    t = comp[comp.split == "test"].set_index("model")
    t[["precision", "recall", "f1", "pr_auc"]].plot.bar(figsize=(11, 4.5))
    plt.title("Test-set metric comparison"); plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(outdir / "model_metric_bars.png", dpi=120); plt.close()

    n = len(test_scores)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    for ax, (name, _) in zip(np.atleast_1d(axes), test_scores.items()):
        r = t.loc[name]
        sns.heatmap([[r.TN, r.FP], [r.FN, r.TP]], annot=True, fmt="d", cbar=False, ax=ax,
                    cmap="Blues", xticklabels=["Pred legit", "Pred fraud"],
                    yticklabels=["Legit", "Fraud"])
        ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(outdir / "model_confusion_matrices.png", dpi=120); plt.close(fig)


def select_final_model(comp: pd.DataFrame) -> str:
    """Select by VALIDATION PR-AUC (never by test performance)."""
    v = comp[(comp.split == "validation")].sort_values("pr_auc", ascending=False)
    return str(v.iloc[0]["model"])


# ------------------------------------------------------- 7. FP / FN ANALYSIS ---
def error_analysis(s: Splits, model_name: str, scores: np.ndarray, thr: float,
                   raw_test: pd.DataFrame, outdir: Path) -> dict:
    y = s.y_test.values
    pred = (scores >= thr).astype(int)
    df = raw_test.copy()
    df["score"], df["y"], df["pred"] = scores, y, pred
    fp, fn = df[(pred == 1) & (y == 0)], df[(pred == 0) & (y == 1)]
    tp, tn = df[(pred == 1) & (y == 1)], df[(pred == 0) & (y == 0)]
    fp.to_csv(outdir / "false_positives_test.csv", index=False)
    fn.to_csv(outdir / "false_negatives_test.csv", index=False)

    def summ(d):
        return {"count": int(len(d)),
                "amount_mean": float(d["Amount"].mean()) if len(d) else None,
                "amount_median": float(d["Amount"].median()) if len(d) else None,
                "amount_total": float(d["Amount"].sum()),
                "score_mean": float(d["score"].mean()) if len(d) else None}
    out = {"model": model_name, "threshold": thr, "FP": summ(fp), "FN": summ(fn),
           "TP": summ(tp), "TN": summ(tn),
           "missed_fraud_amount_pct": float(100 * fn["Amount"].sum() /
                                            max(df[df.y == 1]["Amount"].sum(), 1e-9)),
           "alerts_per_1000_txn": float(1000 * pred.mean())}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for d, lab in ((tn, "TN"), (fp, "FP"), (fn, "FN"), (tp, "TP")):
        if len(d):
            sns.kdeplot(d["score"], ax=ax[0], label=f"{lab} (n={len(d)})", warn_singular=False)
    ax[0].axvline(thr, color="k", ls="--"); ax[0].set_title("Risk-score distribution by outcome")
    ax[0].legend()
    for d, lab in ((tn, "TN"), (fp, "FP"), (fn, "FN"), (tp, "TP")):
        if len(d) > 1:
            sns.kdeplot(d["log_amount"], ax=ax[1], label=lab, warn_singular=False)
    ax[1].set_title("log(1+Amount) by outcome"); ax[1].legend()
    fig.tight_layout(); fig.savefig(outdir / "error_analysis.png", dpi=120); plt.close(fig)

    # threshold trade-off table (validation-independent illustration on test)
    rows = []
    for t in np.unique(np.quantile(scores, [.5, .9, .95, .99, .995, .999])):
        m = metrics_at(y, scores, float(t))
        rows.append({"threshold": t, "precision": m["precision"], "recall": m["recall"],
                     "FP": m["FP"], "FN": m["FN"]})
    pd.DataFrame(rows).round(4).to_csv(outdir / "threshold_tradeoff_test.csv", index=False)
    return out


# ------------------------------------------------------ 8. FEATURE IMPORTANCE ---
def feature_importance(model: Pipeline, s: Splits, outdir: Path, n_repeats: int = 5) -> dict:
    res = {}
    # permutation importance on the VALIDATION set (scoring = PR-AUC)
    pi = permutation_importance(model, s.X_val, s.y_val, scoring="average_precision",
                                n_repeats=n_repeats, random_state=SEED, n_jobs=1)
    pidf = pd.DataFrame({"feature": s.feature_names, "mean": pi.importances_mean,
                         "std": pi.importances_std}).sort_values("mean", ascending=False)
    pidf.to_csv(outdir / "permutation_importance.csv", index=False)
    top = pidf.head(15).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean"], xerr=top["std"])
    plt.title("Permutation importance (validation, drop in PR-AUC)")
    plt.tight_layout(); plt.savefig(outdir / "feature_importance_permutation.png", dpi=120); plt.close()
    res["permutation_top10"] = pidf.head(10).round(5).to_dict("records")

    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        mi = pd.Series(clf.feature_importances_, index=s.feature_names).sort_values(ascending=False)
        mi.to_csv(outdir / "model_feature_importance.csv")
        plt.figure(figsize=(8, 6)); mi.head(15).iloc[::-1].plot.barh()
        plt.title("Model-based feature importance"); plt.tight_layout()
        plt.savefig(outdir / "feature_importance_model.png", dpi=120); plt.close()
        res["model_based_top10"] = mi.head(10).round(5).to_dict()

    if HAS_SHAP and hasattr(clf, "feature_importances_"):
        try:
            Xs = s.X_val.sample(min(1000, len(s.X_val)), random_state=SEED)
            Xt = Pipeline(model.steps[:-1]).transform(Xs)
            sv = shap.TreeExplainer(clf).shap_values(Xt)
            sv = sv[1] if isinstance(sv, list) else sv
            shap.summary_plot(sv, pd.DataFrame(Xt, columns=s.feature_names), show=False)
            plt.tight_layout(); plt.savefig(outdir / "feature_importance_shap.png", dpi=120)
            plt.close()
            res["shap"] = "saved feature_importance_shap.png"
        except Exception as e:  # SHAP is optional
            res["shap"] = f"skipped: {e}"
    return res


# --------------------------------------------------------- 9. RISK SCORING ---
@dataclass
class RiskConfig:
    """Configurable thresholds on the model score. Scores are RANKING scores,
    not calibrated fraud probabilities (class weighting/resampling distorts them)."""
    medium: float = 0.30
    high: float = 0.70

    def categorize(self, score: float) -> str:
        return "High" if score >= self.high else "Medium" if score >= self.medium else "Low"

    @staticmethod
    def action(category: str) -> str:
        return {"Low": "Approve",
                "Medium": "Step-up authentication (e.g., OTP) / queue for analyst review",
                "High": "Hold transaction & request verification; manual review"}[category]


def score_transactions(model: Pipeline, X: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    sc = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({"risk_score": sc})
    out["risk_category"] = [cfg.categorize(v) for v in sc]
    out["recommended_action"] = out["risk_category"].map(RiskConfig.action)
    return out


def risk_distribution(s: Splits, scores: np.ndarray, cfg: RiskConfig, outdir: Path) -> pd.DataFrame:
    cat = pd.Series([cfg.categorize(v) for v in scores], name="category")
    tab = pd.crosstab(cat, s.y_test.values, margins=True)
    tab.columns = [{0: "legit", 1: "fraud"}.get(c, c) for c in tab.columns]
    tab["fraud_rate_%"] = 100 * tab.get("fraud", 0) / tab["All"]
    tab.round(3).to_csv(outdir / "risk_category_distribution_test.csv")
    return tab


def synthetic_demo_transaction(seed: int = 7) -> pd.DataFrame:
    """A made-up transaction for demos - never real cardholder data."""
    rng = np.random.default_rng(seed)
    row = {f"V{i}": rng.normal() for i in range(1, 29)}
    return pd.DataFrame([row])
