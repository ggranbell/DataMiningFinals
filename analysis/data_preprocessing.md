# Data Preprocessing Documentation

## Overview
The `data_preprocessing.py` module is the foundation of the analytics pipeline. It transforms raw, noisy HR data into a high-quality, standardized dataset suitable for machine learning models.

## Key Features

### 1. Data Sanitization
- **Department Standardization**: Resolves high-cardinality issues caused by typos (e.g., mapping "Saless", "SALES", and "Sales" to a single category).
- **Duplicate Removal**: Identifies and removes records with duplicate `Employee_ID`s, keeping the most recent entry.
- **Outlier Management**: 
    - Salary: Values < ₱10,000 or > ₱120,000 are treated as errors and nullified.
    - Age/Absences: Impossible values (e.g., Age < 18) are capped or imputed.

### 2. Missing Value Imputation
- **Numerical**: Uses group-wise median imputation (e.g., salary is imputed based on the median of the employee's department and job level).
- **Categorical**: Uses the mode for missing categorical values.

### 3. Feature Engineering
This module creates several advanced composite features used by the predictive models:

| Feature | Description |
| :--- | :--- |
| **Stability_Score** | A positive composite score based on tenure, promotion count, and regular status. |
| **Attrition_Risk_Score** | A penalty-based score tracking high absences, overtime intensity, and performance drops. |
| **Engagement_Score** | Combines Job Satisfaction and Work-Life Balance scores. |
| **Interaction Terms** | Non-linear combinations like `Tenure_x_Education` and `Salary_Per_Tenure`. |
| **Temporal Features** | Derived metrics like `Career_Start_Age`. |

## Input/Output
- **Input**: `Data Mining_Workforce_Dataset.csv`
- **Output**: 
    - `analysis/output/cleaned_dataset.csv`: The encoded and cleaned data.
    - `analysis/output/data_summary.json`: Statistical summary of the dataset for the dashboard.
