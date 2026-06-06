# Coderush 2026 — Economic Class Classification

![Top 5](https://img.shields.io/badge/placement-top%205-brightgreen)
![Macro F1](https://img.shields.io/badge/OOF%20Macro%20F1-0.7410-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-lightgrey)
![License](https://img.shields.io/badge/license-MIT-orange)

> **Coderush 2026 · ML Module** — Bag-level multiclass economic class classification.  
> Primary metric: Macro F1 · Tie-breaker: Middle Class F1

---

## Problem Overview

Each sample in the dataset is a **bag** — a variable-sized collection of individuals described by demographic and economic attributes (education, occupation, capital gains, hours worked, etc.). The task is to predict a single economic class label for the entire bag:

| Label | Class  | Description                          |
|-------|--------|--------------------------------------|
| 0     | lower  | Individuals in the lower economic tier  |
| 1     | middle | Individuals in the middle economic tier |
| 2     | upper  | Individuals in the upper economic tier  |

This is a **Multiple-Instance Learning (MIL)** problem — supervision is at the bag level only; no per-individual labels are provided.

---

## Dataset

| Split | Rows   | Bags  |
|-------|--------|-------|
| Train | 16,776 | 3,360 |
| Test  |  1,981 |   400 |

**Class distribution (train bags):**
- middle: 1,260
- upper:  1,120
- lower:    980

---

## Approach

### 1. Bag Aggregation

Raw rows are grouped by `bag_id` and collapsed into a single feature vector per bag using statistical aggregations (mean, std, min, max, count, sum, nunique) over all numeric and categorical columns. This converts the MIL setup into a standard tabular classification task.

- Train shape after aggregation: `(3360, 176)`
- Test shape after aggregation: `(400, 175)`
- **174 engineered features** used for training

### 2. Multi-Seed × 5-Fold Cross-Validation

To reduce variance and produce reliable out-of-fold (OOF) predictions for blending and threshold tuning, we train across **5 random seeds × 5-fold stratified CV** for each model.

| Seed  | OOF Macro F1 | Fold scores                                    |
|-------|-------------|------------------------------------------------|
| 42    | 0.7089      | 0.7042 · 0.7383 · 0.6983 · 0.6876 · 0.7162   |
| 123   | 0.7205      | 0.7069 · 0.7200 · 0.7362 · 0.7203 · 0.7192   |
| 777   | 0.7239      | 0.7202 · 0.7103 · 0.7209 · 0.7192 · 0.7489   |
| 2024  | 0.7210      | 0.7409 · 0.7300 · 0.7029 · 0.7317 · 0.6995   |
| 31415 | 0.7228      | 0.7409 · 0.6986 · 0.7104 · 0.7421 · 0.7219   |

### 3. Model Ensemble

Three gradient boosting frameworks were trained and blended:

| Model    | Role                        |
|----------|-----------------------------|
| LightGBM | Primary learner (70% weight) |
| XGBoost  | Secondary learner (30% weight) |
| CatBoost | Explored; 0% in final blend |

Blend weights were searched over OOF predictions:
```
Best OOF F1: 0.7127  →  LGBM=0.7, XGB=0.3, CB=0.0
```

### 4. Decision Threshold Tuning

Per-class decision thresholds were tuned on the combined OOF probability predictions to maximise Macro F1. This gave the biggest single improvement in the pipeline:

| Class  | Threshold |
|--------|-----------|
| lower  | 0.25      |
| middle | 0.25      |
| upper  | 0.55      |

```
Pre-threshold  OOF Macro F1: 0.7127
Post-threshold OOF Macro F1: 0.7410  (+0.028)
```

### 5. Test Predictions

```
lower: 136  |  middle: 212  |  upper: 52
Submission: submission_v3.csv
```

---

## Results

| Metric                  | Value  |
|-------------------------|--------|
| OOF Macro F1 (pre-threshold) | 0.7127 |
| **OOF Macro F1 (final)**     | **0.7410** |
| Competition placement   | **Top 5** |

---

## Repository Structure

```
.
├── coderush26_v2.ipynb               # Main solution notebook (run end-to-end)
├── Coderush-26-ML-Train.csv          # Training data (16,776 rows)
├── Coderush-26-ML-test.csv           # Test data (1,981 rows, no labels)
├── submission_v3.csv                 # Final Kaggle submission
├── Coderush26_ML_ProblemStatement.pdf # Official problem statement
└── README.md
```

---

## Quick Start

**Install dependencies:**
```bash
pip install lightgbm xgboost catboost scikit-learn pandas numpy
```

**Run the notebook:**

Open `coderush26_v2.ipynb` and run all cells. The notebook expects the CSV files in the same directory (update paths in the data-loading cell if needed). It will output:
- A saved model checkpoint (`.pkl` / `.joblib`)
- `submission_v3.csv` with bag-level predictions

---

## Key Takeaways

- **Bag aggregation** is the critical preprocessing step — collapsing variable-length bags into fixed-size feature vectors is what makes standard tabular ML applicable here.
- **Threshold tuning** on OOF probabilities gave +2.8 F1 points with zero additional training cost.
- **Multi-seed ensembling** provides more stable OOF estimates for blend-weight and threshold search than a single seed.
- **CatBoost added noise** in this setting despite being a strong baseline on many tabular tasks — the blend search correctly assigned it 0% weight.
- Judges reward simplicity: a well-tuned gradient boosting ensemble outperformed more complex approaches while remaining fully interpretable.

---

## Competition Details

| Item             | Detail                                      |
|------------------|---------------------------------------------|
| Competition      | Coderush 2026 — ML Module                   |
| Task             | Multiclass bag-level classification         |
| Primary metric   | Macro F1                                    |
| Tie-breaker      | Middle Class F1                             |
| Platform         | Kaggle                                      |
| Deadline         | 9 May 2026 · 08:00 AM PKT                   |
| Presentation day | 10 May 2026                                 |
