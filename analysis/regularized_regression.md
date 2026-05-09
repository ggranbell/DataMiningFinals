# Regularized Regression Documentation

## Overview
The `regularized_regression.py` module predicts employee monthly salary. It focuses on high-precision forecasting by accounting for the complex, non-linear relationships between department, education, and tenure.

## Modeling Strategy

### 1. Target Transformation
The model predicts the **Log of Monthly Salary**. Salaries are naturally skewed; log-transformation normalizes the variance and allows the linear models to capture percentage-based growth rather than just absolute currency increases.

### 2. Interaction Features
The core of the model's accuracy (**>91% R²**) comes from specialized interaction terms:
- **Department × Tenure**: Recognizes that salary growth rates differ by department (e.g., IT vs. HR).
- **Department × Education**: Accounts for varying premiums on advanced degrees across different functions.
- **Education Squared**: Captures the non-linear "boost" provided by PhDs and Master's degrees.

### 3. Model Ensemble
The engine evaluates three types of regularized linear models using Cross-Validation:
- **Ridge (L2)**: Prevents coefficients from becoming too large; handles multicollinearity well.
- **Lasso (L1)**: Performs automatic feature selection by zeroing out irrelevant predictors.
- **Elastic Net**: A hybrid of L1 and L2, useful when many features are correlated.

## Key Insights
- **Top Predictors**: Education level, Department (IT/Finance), and Tenure.
- **Accuracy**: The **Lasso** model currently yields the best results with an **R² of 0.9110**.

## Output
- `analysis/output/regression_results.json`: Detailed comparison of all three models and top coefficient lists.
