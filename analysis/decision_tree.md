# Decision Tree Classification Documentation

## Overview
The `decision_tree.py` module handles the predictive modeling for employee attrition. It uses an entropy-based Decision Tree classifier to identify the factors most likely to lead to an employee leaving the company.

## Methodology

### 1. Handling Class Imbalance
The dataset contains a significant imbalance between employees who stay and those who leave. To address this, the model implements **SMOTE (Synthetic Minority Oversampling Technique)**, which generates synthetic examples of the minority class (attrition) to ensure the model doesn't become biased toward predicting "Stay".

### 2. Hyperparameter Optimization
The model uses `GridSearchCV` to find the optimal balance between depth and complexity, searching across:
- `max_depth`: [5, 7, 10, 15]
- `min_samples_leaf`: [10, 20, 50]
- `criterion`: ['entropy', 'gini']
- `ccp_alpha`: Cost-complexity pruning to prevent overfitting.

### 3. Feature Selection
The model prioritizes high-signal engineered features:
- **Stability_Score**: The strongest predictor of retention.
- **Attrition_Risk_Score**: Captures immediate red flags like high absences and low satisfaction.
- **Tenure_x_Promotions**: Identifies employees who have "stagnated" in their roles.

## Performance Metrics
The model is evaluated using 5-fold cross-validation focusing on:
- **ROC AUC**: Current performance ~**84.8%**.
- **F1-Score**: Current performance ~**80.2%**.
- **Confusion Matrix**: Visualized in the dashboard to track False Positives (over-predicting attrition) vs. False Negatives (missing actual departures).

## Output
- `analysis/output/decision_tree_results.json`: Contains metrics, feature importance, confusion matrix, and simplified tree rules.
