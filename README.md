# People Analytics - Data Mining Implementation Documentation

## Project Overview

Comprehensive people analytics model using employee HR, payroll, and performance data from a mid-to-large Philippine corporation (5,025 records over 15 years).

---

## 1. Data Preprocessing Pipeline

**File**: [data_preprocessing.py](file:///c:/Users/Granbell/DataMiningFinals/analysis/data_preprocessing.py)

### Issues Found & Fixed

| Issue | Examples | Fix Applied |
|-------|---------|-------------|
| **Typos in Department** | `SALES`, `Saless`, `I.T.`, `hr`, `Ops`, `Operationsn` | Mapped to standardized names |
| **Missing Values** | ~500 across Salary, Performance, Satisfaction, etc. | Median (numerical), Mode (categorical) |
| **Impossible Values** | Age=1, Absences=-5, Distance=500km | Capped/nullified + imputed |
| **Extreme Outliers** | Absences=150, Salary=1500, Overtime>60 | Capped at reasonable bounds |
| **Duplicate IDs** | 25 duplicate Employee_IDs | Kept first occurrence |

### Engineered Features

- `Salary_Bracket`: Low/Medium/High/Very High
- `Age_Group`: Young/Mid-Career/Senior
- `High_Performer`: Binary (score >= 4)
- `Engagement_Score`: Composite of satisfaction + WLB
- `Promotion_Rate`: Promotions per tenure year
- `Overtime_Intensity`: Low/Moderate/High/Very High

---

## 2. Decision Tree Classification

**File**: [decision_tree.py](file:///c:/Users/Granbell/DataMiningFinals/analysis/decision_tree.py)

| Metric | Value |
|--------|-------|
| **Accuracy** | 75.7% |
| **F1 Score** | 78.4% |
| **ROC AUC** | 82.3% |
| **CV Mean F1** | 5-fold cross-validated |

### Key Findings

- **Top attrition drivers**: Num_Promotions, Tenure_Years, Job_Satisfaction_Score
- Employees with fewer promotions and lower tenure are more likely to leave
- GridSearchCV used for hyperparameter tuning (max_depth, min_samples, criterion)

---

## 3. Hierarchical Clustering

**File**: [hierarchical_clustering.py](file:///c:/Users/Granbell/DataMiningFinals/analysis/hierarchical_clustering.py)

| Metric | Value |
|--------|-------|
| **Optimal Clusters** | 2 |
| **Silhouette Score** | 0.8998 |
| **Method** | Ward's (Agglomerative) |
| **Distance** | Euclidean |

### Segments Identified

1. **Core Workforce** (3,246 employees, 71.4% attrition) - Lower salary, shorter tenure
2. **Loyal Veterans** (1,754 employees, 27.0% attrition) - Higher salary, long tenure

---

## 4. Regularized Regression

**File**: [regularized_regression.py](file:///c:/Users/Granbell/DataMiningFinals/analysis/regularized_regression.py)

| Model | R² | MAE (₱) | Alpha |
|-------|-----|---------|-------|
| **Ridge** | 0.0040 | 20,193 | 6,136 |
| **Lasso** | 0.0057 | 19,202 | 5,094 |
| **Elastic Net** | 0.0041 | 20,213 | 27.8 |

### Top Salary Predictors

1. Education_Level (strongest positive)
2. Tenure_Years (positive)
3. Num_Promotions (positive)
4. Job_Satisfaction_Score (negative — higher satisfaction in lower-paid roles)

> [!NOTE]
> Low R² values are expected — salary is driven by many unmeasured factors (role level, market rates, negotiation). The models successfully identify *which measured features* correlate with pay.

---

## 5. Dashboard

**Location**: [dashboard/index.html](file:///c:/Users/Granbell/DataMiningFinals/dashboard/index.html)

### Tabs

- **Overview**: KPIs, attrition distribution, salary histogram, data quality summary
- **Demographics**: Department, education, gender, region, employment type, age charts
- **Performance**: Ratings distribution, attrition by dept, salary by dept, tenure-salary scatter
- **Decision Tree**: Feature importance, confusion matrix, decision rules
- **Clustering**: Silhouette analysis, cluster profiles with radar chart
- **Regression**: Model comparison (R², MAE), coefficient analysis, Lasso feature selection

### How to Run

```bash
# Run analysis (generates JSON outputs)
set PYTHONIOENCODING=utf-8
python analysis/run_analysis.py

# Serve dashboard
python -m http.server 8765
# Open http://localhost:8765/dashboard/index.html
```

---

## Project Structure

```
DataMiningFinals/
├── Data Mining_Final Exam_Workforce Dataset.csv  (raw data)
├── analysis/
│   ├── data_preprocessing.py      (cleaning pipeline)
│   ├── decision_tree.py           (attrition classification)
│   ├── hierarchical_clustering.py (employee segmentation)
│   ├── regularized_regression.py  (salary prediction)
│   ├── run_analysis.py            (orchestrator)
│   └── output/                    (JSON results + cleaned CSVs)
└── dashboard/
    ├── index.html                 (main page)
    ├── styles.css                 (dark theme)
    └── app.js                     (Chart.js visualizations)
```
