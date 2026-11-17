# Customer Churn Prediction & Explainable AI (SHAP) Retention Simulator

> **XGBoost/CatBoost churn scoring with SHAP-driven retention ROI simulation**

## Overview

A supervised churn prediction system that identifies high-risk subscribers, explains drivers with SHAP, and simulates the ROI of targeted retention campaigns.

## Approach

1. **Preprocessing** with class-imbalance handling (SMOTE / class weights).
2. **Modeling** with gradient boosting (XGBoost / CatBoost) and cross-validation.
3. **Explainability** using SHAP values at global and individual level.
4. **Retention simulator** estimating CLV saved under a campaign scenario.

## Repository Structure

```
customer-churn-shap/
├── src/train.py           # training pipeline
├── src/explain.py         # SHAP explanations
├── src/retention.py       # campaign ROI simulator
├── requirements.txt
└── README.md
```

## Tech Stack

`Python` · `XGBoost` · `CatBoost` · `scikit-learn` · `imbalanced-learn` · `SHAP`

## Getting Started

```bash
git clone https://github.com/henryferdianto/customer-churn-shap.git
cd customer-churn-shap
pip install -r requirements.txt
```

```bash
python src/train.py --input data/raw/churn.csv
python src/explain.py --model models/churn.joblib --input data/raw/churn.csv
```

## Project Status

This repository contains the project documentation and a runnable implementation
skeleton. Scripts are structured so that each pipeline stage can be executed and
extended independently. Update the configuration and data paths to run against
your own datasets.

## License

MIT License — see [LICENSE](LICENSE).


---

**Author:** Rifky Henry Ferdianto — Data Science undergraduate at ITERA  
GitHub: [@henryferdianto](https://github.com/henryferdianto) · Portfolio: [rifky-henry-ferdianto](https://github.com/henryferdianto/rifky-henry-ferdianto)
