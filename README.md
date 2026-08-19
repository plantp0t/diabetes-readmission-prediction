# Predicting Early Hospital Readmission in Diabetic Patients

Data cleaning, exploratory analysis, and machine learning pipeline built to predict
whether a diabetic patient will be readmitted to hospital within 30 days.

Originally developed as my BSc thesis project. A supervisor and I later extended this work further for publication -
see [Related Publication](#related-publication) below. This repository contains my
original thesis version of the code.

## Dataset

[Diabetes 130-US hospitals for years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- a public dataset from the UCI Machine Learning Repository containing ~100,000 hospital
encounters for diabetic patients across 130 US hospitals.

The raw dataset is not included in this repository (see [Reproducing](#reproducing) below)
- download it directly from UCI using the link above.

## What this project covers

**Data cleaning & preprocessing**

- Deduplication to first-encounter-per-patient
- Missing value handling (explicit `?` values, sparse columns dropped, informed imputation
  e.g. using `admission_type_id` to fill missing `admission_source_id`)
- Categorical grouping and re-encoding (admission type, discharge disposition, admission
  source, diagnosis codes)
- Outlier and skewness analysis, PowerTransformer + StandardScaler for numeric features
- Class imbalance handling with **SMOTE**
- Feature engineering (e.g. `total_visits`, `num_changes` in medication)
- Multicollinearity check via correlation matrix + VIF

**Modeling & evaluation**

- Models trained: Logistic Regression, SGD Classifier, Decision Tree, Random Forest,
  Gradient Boosting, XGBoost
- Hyperparameter tuning via `RandomizedSearchCV` / `StratifiedKFold`
- Evaluated on held-out train / validation / test splits using AUC, accuracy, precision,
  recall, specificity, and F1
- Feature importance analysis across models
- Best model selected and serialized (`pickle` / `joblib`)

**Final test-set result (best model - tuned SGD Classifier, threshold 0.7):**

| Metric    | Train | Validation | Test  |
| --------- | ----- | ---------- | ----- |
| AUC       | 0.685 | 0.609      | 0.600 |
| Recall    | 0.056 | 0.693      | 0.678 |
| Precision | 0.784 | 0.138      | 0.142 |
| F1        | 0.105 | 0.230      | 0.235 |

## Repository structure

```
├── notebooks/
│   └── Data_preprocessing_and_model_evaluation_code.ipynb   # full pipeline, cleaning → modeling
├── requirements.txt
└── README.md
```

## Reproducing

1. Download `diabetic_data.csv` and `IDS_mapping.csv` from the
   [UCI dataset page](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
2. Install dependencies: `pip install -r requirements.txt`
3. Open `notebooks/Data_preprocessing_and_model_evaluation_code.ipynb` and update the
   file paths in the loading cells to point to your local copies of the CSVs (the
   notebook was originally run in Google Colab - see the note at the top of the notebook).

## Tech stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · imbalanced-learn (SMOTE) ·
Matplotlib · Seaborn · Statsmodels

## Related publication

This thesis project was later extended, with additional analysis and testing, and
published together with my supervisor.
https://learning-gate.com/index.php/2576-8484/article/view/4929/1842

This repository reflects my own original thesis implementation, prior to that extension.
