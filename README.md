# ⚽ Football Match Outcome Prediction with XAI 
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Challenge](https://img.shields.io/badge/ChallengeData-QRT-red)

## 🧾 Overview

This project was developed as part of the **QRT Data Challenge** on [ChallengeData](https://challengedata.ens.fr/challenges/143). The goal was to predict the outcomes of football matches (Home Win, Draw, Away Win) using real-world historical data from multiple global leagues provided by Sportmonks.

We explored multiple machine learning models and explainability techniques to build a robust classification system capable of handling noisy, imbalanced, and high-dimensional tabular data.

---

## 🧠 Problem Statement

Predict the outcome of a football match as one of the following classes:

- 🏠 Home Win
- ⚖️ Draw
- 🛫 Away Win

Given:
- Team-level statistics (e.g., passes, shots, fouls) over full season and recent 5 games.
- Player-level aggregated metrics.
- Preprocessed and normalized dataset.

---

## 🧰 Tools & Libraries

- Python 3.8+
- Scikit-learn
- XGBoost
- SHAP
- LIME
- NumPy, Pandas, Matplotlib, Seaborn
- Jupyter Notebook

---

## 🗃️ Dataset Overview

- `X_train.zip` and `X_test.zip` – Contains 4 CSVs each for HOME and AWAY (team and player stats).
- `Y_train.csv` – One-hot encoded labels for match outcome.
- `Y_train_supp.csv` – Contains goal difference labels.
- ~1000 matches across multiple leagues.

> ⚠️ Team and player names are anonymized in the test set. No external data allowed.

---

## 📊 Key Challenges

- 🧩 Missing values due to partial match records.
- ⚖️ Strong class imbalance (draws underrepresented).
- 🔍 Redundant features due to aggregation (_sum, _mean, _std).
- ⚙️ High feature dimensionality.

---

## 🧪 Models Tested

| Category           | Models                         | Accuracy (%) |
|--------------------|--------------------------------|--------------|
| Linear Models      | Logistic Regression            | 49.7         |
| Tree-Based Models  | Decision Tree, Random Forest, Gradient Boosting | 37–49.3  |
| Neural Nets / SVM  | SVM (RBF), MLP                 | **SVM: 49.7** |

Best-performing model: **SVM (RBF)** with hyperparameter tuning.

---

## ⚙️ Methodology

1. **EDA & Feature Selection**  
   Removed redundant stats (_sum, _std) in favor of normalized metrics (_average).

2. **Preprocessing**  
   - Missing value imputation (mean strategy)
   - Feature scaling (StandardScaler)

3. **Modeling**  
   - Multiple classifiers tested using cross-validation
   - Final model tuned with GridSearchCV

4. **Evaluation Metrics**  
   - Accuracy
   - Weighted F1-Score
   - Confusion Matrix
   - Classification Report

---

## 🔎 Explainability (XAI)

We applied both **SHAP** and **LIME** to interpret model predictions:

- SHAP: Provided global feature importance and per-instance explanations.
- LIME: Offered local interpretability for edge cases and misclassifications.

---

## 📈 Results Summary

- Best accuracy: **49.7%** (SVM with RBF kernel)
- Strong prediction for `Home Wins`, weaker performance on `Draw` due to imbalance
- SHAP showed top features: `GOALS`, `SHOTS_ON_TARGET`, `POSSESSION`

---

## 📁 File Structure

```text
├── data/
│   ├── X_train.zip
│   ├── X_test.zip
│   ├── Y_train.csv
│   └── Y_train_supp.csv
├── notebooks/
│   ├── eda.ipynb
│   ├── model_training.ipynb
│   └── explainability.ipynb
├── README.md
└── requirements.txt
