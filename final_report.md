# Fraud Detection & Risk Analysis - Auto-generated Report

**Data source:** .\data\creditcard.csv


All values below were measured by `run_pipeline.py`. Re-run to reproduce (seed=42).

## 1. Dataset overview & quality
- Shape (raw): (284807, 31), dtypes: {'float64': 30, 'int64': 1}
- Missing values: 0; exact duplicates: 1081 (removed before splitting)
- Zero-amount rows: 1825; negative amounts: 0
- Legitimate: 284315 (99.8273%) | Fraud: 492 (0.1727%)
- After cleaning: 283726 rows, fraud 0.1667%

## 2. EDA highlights (see PNGs in outputs/)
- Features most correlated with Class: {'V17': -0.3135, 'V14': -0.2934, 'V12': -0.2507, 'V10': -0.207, 'V16': -0.1872, 'V3': -0.1823, 'V7': -0.1723, 'V11': 0.1491, 'V4': 0.1293, 'V18': -0.1053, 'V1': -0.0945, 'V9': -0.094}
- Share of fraud above the legitimate 99th-percentile amount (1018.06): 0.019
- Top Mann-Whitney effects: see `eda_stat_tests.csv` (Bonferroni-adjusted p-values included)
- Hourly fraud rates: `eda_time.png` (Time is an offset from the first transaction; the "hour" is relative, not clock time)

## 3. Preprocessing
Log amount, cyclic hour features, night flag; stratified 60/20/20 split; median imputation and scaling are inside
pipelines (fit on training data only). Resampling is applied only to training data.

## 4. Imbalance strategies (HistGradientBoosting, validation set)
| strategy             |   train_size |   train_fraud_pct |   val_PR_AUC |   val_ROC_AUC |   val_precision@bestF1thr |   val_recall@bestF1thr |   val_F1@bestF1thr |
|:---------------------|-------------:|------------------:|-------------:|--------------:|--------------------------:|-----------------------:|-------------------:|
| None (baseline)      |       170235 |            0.1668 |       0.6617 |        0.8746 |                    0.8256 |                 0.7553 |             0.7889 |
| Class-weighted       |       170235 |            0.1668 |       0.7743 |        0.9562 |                    0.9048 |                 0.8085 |             0.8539 |
| Random oversampling  |       339902 |           50      |       0.8814 |        0.9694 |                    0.9318 |                 0.8723 |             0.9011 |
| Random undersampling |          568 |           50      |       0.7424 |        0.9805 |                    0.6792 |                 0.766  |             0.72   |
| SMOTE                |       339902 |           50      |       0.8131 |        0.9696 |                    0.9412 |                 0.8511 |             0.8939 |

## 5. Model comparison
### Validation (threshold chosen here to maximise F1)
| model                |   threshold |   accuracy |   precision |   recall |     f1 |   roc_auc |   pr_auc |    TN |   FP |   FN |   TP |
|:---------------------|------------:|-----------:|------------:|---------:|-------:|----------:|---------:|------:|-----:|-----:|-----:|
| Logistic Regression  |      1      |     0.9995 |      0.8764 |   0.8298 | 0.8525 |    0.9821 |   0.7899 | 56640 |   11 |   16 |   78 |
| Decision Tree        |      0.9986 |     0.9994 |      0.8831 |   0.7234 | 0.7953 |    0.9089 |   0.7357 | 56642 |    9 |   26 |   68 |
| Random Forest        |      0.5295 |     0.9996 |      0.9186 |   0.8404 | 0.8778 |    0.9542 |   0.8683 | 56644 |    7 |   15 |   79 |
| HistGradientBoosting |      0.9161 |     0.9996 |      0.8804 |   0.8617 | 0.871  |    0.9541 |   0.7753 | 56640 |   11 |   13 |   81 |
| XGBoost              |      0.429  |     0.9997 |      0.9419 |   0.8617 | 0.9    |    0.9697 |   0.88   | 56646 |    5 |   13 |   81 |
| Isolation Forest     |      0.0348 |     0.9983 |      0.4524 |   0.2021 | 0.2794 |    0.9452 |   0.2361 | 56628 |   23 |   75 |   19 |
### Held-out TEST (same validation threshold)
| model                |   threshold |   accuracy |   precision |   recall |     f1 |   roc_auc |   pr_auc |    TN |   FP |   FN |   TP |
|:---------------------|------------:|-----------:|------------:|---------:|-------:|----------:|---------:|------:|-----:|-----:|-----:|
| Logistic Regression  |      1      |     0.9993 |      0.8372 |   0.7579 | 0.7956 |    0.9624 |   0.6762 | 56637 |   14 |   23 |   72 |
| Decision Tree        |      0.9986 |     0.9993 |      0.8649 |   0.6737 | 0.7574 |    0.8938 |   0.6887 | 56641 |   10 |   31 |   64 |
| Random Forest        |      0.5295 |     0.9995 |      0.9444 |   0.7158 | 0.8144 |    0.9534 |   0.809  | 56647 |    4 |   27 |   68 |
| HistGradientBoosting |      0.9161 |     0.9994 |      0.8554 |   0.7474 | 0.7978 |    0.9652 |   0.7237 | 56639 |   12 |   24 |   71 |
| XGBoost              |      0.429  |     0.9995 |      0.96   |   0.7579 | 0.8471 |    0.9715 |   0.8242 | 56648 |    3 |   23 |   72 |
| Isolation Forest     |      0.0348 |     0.998  |      0.2766 |   0.1368 | 0.1831 |    0.9375 |   0.1407 | 56617 |   34 |   82 |   13 |
### Held-out TEST at the default 0.5 threshold (for contrast)
| model                |   threshold |   accuracy |   precision |   recall |     f1 |   roc_auc |   pr_auc |    TN |   FP |   FN |   TP |
|:---------------------|------------:|-----------:|------------:|---------:|-------:|----------:|---------:|------:|-----:|-----:|-----:|
| Logistic Regression  |         0.5 |     0.9715 |      0.0491 |   0.8737 | 0.093  |    0.9624 |   0.6762 | 55044 | 1607 |   12 |   83 |
| Decision Tree        |         0.5 |     0.9937 |      0.1829 |   0.7895 | 0.297  |    0.8938 |   0.6887 | 56316 |  335 |   20 |   75 |
| Random Forest        |         0.5 |     0.9994 |      0.9315 |   0.7158 | 0.8095 |    0.9534 |   0.809  | 56646 |    5 |   27 |   68 |
| HistGradientBoosting |         0.5 |     0.9986 |      0.5597 |   0.7895 | 0.655  |    0.9652 |   0.7237 | 56592 |   59 |   20 |   75 |
| XGBoost              |         0.5 |     0.9996 |      0.973  |   0.7579 | 0.8521 |    0.9715 |   0.8242 | 56649 |    2 |   23 |   72 |

Accuracy is misleading here: a model predicting "legitimate" for everything achieves
99.827% accuracy while catching zero fraud. Use PR-AUC, recall and precision.

## 6. Final model
**XGBoost** (selected by validation PR-AUC; threshold=0.4290).
Test: precision=0.9600, recall=0.7579, F1=0.8471, ROC-AUC=0.9715, PR-AUC=0.8242
Confusion matrix (test): TN=56648, FP=3, FN=23, TP=72

## 7. False positives / false negatives (test)
{
  "model": "XGBoost",
  "threshold": 0.42898452281951904,
  "FP": {
    "count": 3,
    "amount_mean": 8600.686651952155,
    "amount_median": 109.89999991181655,
    "amount_total": 25802.059955856465,
    "score_mean": 0.7886233925819397
  },
  "FN": {
    "count": 23,
    "amount_mean": 165.7513041685651,
    "amount_median": 2.219999999364123,
    "amount_total": 3812.279995876997,
    "score_mean": 0.03717007488012314
  },
  "TP": {
    "count": 72,
    "amount_mean": 152.13930539381965,
    "amount_median": 34.294999978757545,
    "amount_total": 10954.029988355016,
    "score_mean": 0.9985262751579285
  },
  "TN": {
    "count": 56648,
    "amount_mean": 88.25243300945336,
    "amount_median": 21.59999998810023,
    "amount_total": 4999323.825119514,
    "score_mean": 6.209353887243196e-05
  },
  "missed_fraud_amount_pct": 25.81741816302031,
  "alerts_per_1000_txn": 1.3216790610791949
}

Threshold trade-off: `threshold_tradeoff_test.csv`. Lowering the threshold raises recall and alert volume;
raising it reduces customer friction but misses more fraud.

## 8. Feature importance
{
  "permutation_top10": [
    {
      "feature": "V14",
      "mean": 0.20285,
      "std": 0.00647
    },
    {
      "feature": "V4",
      "mean": 0.06652,
      "std": 0.01261
    },
    {
      "feature": "V11",
      "mean": 0.0254,
      "std": 0.0079
    },
    {
      "feature": "V3",
      "mean": 0.01391,
      "std": 0.00561
    },
    {
      "feature": "V10",
      "mean": 0.01326,
      "std": 0.00368
    },
    {
      "feature": "V12",
      "mean": 0.01148,
      "std": 0.00358
    },
    {
      "feature": "log_amount",
      "mean": 0.01145,
      "std": 0.00566
    },
    {
      "feature": "V1",
      "mean": 0.00749,
      "std": 0.00232
    },
    {
      "feature": "V8",
      "mean": 0.00323,
      "std": 0.00072
    },
    {
      "feature": "V25",
      "mean": 0.00311,
      "std": 0.00072
    }
  ],
  "model_based_top10": {
    "V14": 0.5607100129127502,
    "V4": 0.0884300023317337,
    "V8": 0.03610000014305115,
    "V12": 0.02937999926507473,
    "V13": 0.018719999119639397,
    "V19": 0.01844000071287155,
    "V10": 0.01768999919295311,
    "log_amount": 0.016589999198913574,
    "V21": 0.016289999708533287,
    "V20": 0.016090000048279762
  },
  "shap": "saved feature_importance_shap.png"
}

Importances show statistical association with the model's predictions, **not causes**. V1..V28 are anonymised
PCA components and cannot be interpreted as business concepts.

## 9. Risk categories on the test set (thresholds: medium=0.3, high=0.7)
| category   |   legit |   fraud |   All |   fraud_rate_% |
|:-----------|--------:|--------:|------:|---------------:|
| High       |       2 |      72 |    74 |     97.2973    |
| Low        |   56647 |      23 | 56670 |      0.0405858 |
| Medium     |       2 |       0 |     2 |      0         |
| All        |   56651 |      95 | 56746 |      0.167413  |

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
