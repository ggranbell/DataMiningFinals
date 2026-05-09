<div align="center">
  <h1>📊 Data Mining Finals</h1>
  <p>
    <b>A Premium Predictive Modeling Suite for Strategic Human Capital Management and Attrition Forecasting.</b>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
    <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript" />
  </p>
</div>

---

## 📖 Overview

**Data Mining Finals** is an end-to-end data mining ecosystem designed to transform raw HR data into actionable organizational intelligence. By leveraging advanced machine learning architectures, the platform provides high-precision forecasting for employee attrition, salary benchmarks, and behavioral segmentation. It features a robust automated preprocessing engine and a sleek, glassmorphic analytics dashboard for real-time visualization of workforce health.

## ✨ Features

- **🛡️ Attrition Classification**: Advanced Decision Tree model using SMOTE (Synthetic Minority Oversampling) and cost-complexity pruning to predict employee turnover with **84.8% AUC**.
- **💰 Salary Regression Engine**: High-fidelity regularized regression (Lasso/Ridge) achieving **>91.1% R²** using non-linear interaction terms between departments, education, and tenure.
- **👥 Employee Segmentation**: Hierarchical clustering system using **QuantileTransformer** and multi-method linkage to achieve a **0.528 Silhouette Score** across 4 optimal personas.
- **🧹 Automated Data Sanitizer**: Intelligent pipeline that standardizes departmental typos, imputes missing metrics using group-wise medians, and handles extreme financial outliers.
- **📈 Interactive Analytics Dashboard**: A premium, dark-mode visualization suite built with Chart.js, featuring glassmorphism, fluid micro-animations, and real-time model metric displays.
- **📄 CSV & JSON Reporting**: Integrated orchestrator that exports clean datasets and comprehensive model performance logs for secondary analysis.

## 🛠️ Tech Stack

- **Core Analytics**: Python 3.x, NumPy, SciPy.
- **Machine Learning**: Scikit-Learn (Decision Trees, Regularized Linear Models).
- **Data Engineering**: Pandas (Complex feature engineering & interaction mapping).
- **Frontend Dashboard**: Vanilla HTML5, CSS3 (Custom Design System), ES6 JavaScript.
- **Visualization**: Chart.js for dynamic, interactive workforce distributions.

## 🚀 Quick Setup

### Prerequisites

1. **Python**: v3.8 or higher.
2. **Browser**: Any modern browser for the dashboard.

### Installation & Execution

1. **Initialize the Pipeline**

   ```bash
   # Install required analytics packages
   pip install pandas numpy scikit-learn scipy imbalanced-learn
   ```

2. **Run the Analysis Orchestrator**

   ```bash
   # Run full modeling pipeline (Preprocessing -> Models -> Export)
   set PYTHONIOENCODING=utf-8
   python analysis/run_analysis.py
   ```

3. **Launch the Visualization Suite**

   ```bash
   # Open the dashboard locally
   python -m http.server 8765
   # Navigate to http://localhost:8765/dashboard/index.html
   ```

## 📂 Project Architecture

```text
DataMiningFinals/
├── Data Mining_Workforce_Dataset.csv   # Raw source data
├── analysis/                          # Backend Analytics Engine
├── output/                            # Model Result Artifacts
└── dashboard/                         # Premium Visualization UI
```

### 📖 Technical Documentation

Detailed documentation for each analytical module:

- 🛠️ [**Data Preprocessing**](analysis/data_preprocessing.md): Cleaning, imputation, and feature engineering logic.
- 🌳 [**Decision Tree**](analysis/decision_tree.md): Attrition classification using SMOTE and cost-complexity pruning.
- 🧬 [**Hierarchical Clustering**](analysis/hierarchical_clustering.md): Workforce segmentation using QuantileTransformers and multi-linkage optimization.
- 📉 [**Regularized Regression**](analysis/regularized_regression.md): Salary prediction using Lasso/Ridge with interaction features.
- ⚙️ [**Analysis Orchestrator**](analysis/run_analysis.md): Main execution pipeline and system architecture.

## 🎨 Model Specifics

### Salary Prediction Engine (`regularized_regression.py`)

| Model | R² | Adj R² | MAE (₱) |
|-------|-----|--------|---------|
| Lasso | **0.9110** | **0.9033** | **3,281** |

- **Department Interactions**: Modeling how salary growth curves differ across IT, Finance, and Sales.
- **Log-Scaling**: Normalizing target distributions to handle financial skew effectively.

### Employee Segmentation (`hierarchical_clustering.py`)

| Metric | Value |
|--------|-------|
| **Silhouette Score** | **0.5287** |
| **Optimal Clusters** | 4 |

- **QuantileTransformer**: Normalizing features to maximize cluster cohesion.
- **Persona Identification**: Automatically segmenting the workforce into 4 distinct behavioral cohorts.

### Attrition Risk Engine (`decision_tree.py`)

Features a custom-weighted **Stability Score** and **Risk Score** that achieves **84.8% AUC** and **80.2% F1**, significantly outperforming baseline models.
