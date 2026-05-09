"""
================================================================================
MAIN RUNNER - PEOPLE ANALYTICS PIPELINE
================================================================================

This script orchestrates the complete data mining workflow:
1. Data Preprocessing (cleaning, imputation, feature engineering)
2. Decision Tree Classification (attrition prediction)
3. Hierarchical Clustering (employee segmentation)
4. Regularized Regression (Ridge, Lasso, Elastic Net for salary prediction)

All results are saved as JSON files in the analysis/output/ directory,
which are then consumed by the interactive visualization dashboard.

Usage:
    python analysis/run_analysis.py
================================================================================
"""

import sys
import os
import time

# Ensure the analysis package is in path
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import run_full_preprocessing
from decision_tree import run_decision_tree
from hierarchical_clustering import run_hierarchical_clustering
from regularized_regression import run_regularized_regression


def main():
    start_time = time.time()
    
    print("\n" + "█" * 70)
    print("█" + "  PEOPLE ANALYTICS - COMPREHENSIVE DATA MINING ANALYSIS".center(68) + "█")
    print("█" + "  Philippine Workforce Dataset | 5,025 Employee Records".center(68) + "█")
    print("█" * 70)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # PHASE 1: DATA PREPROCESSING
    # ========================================================================
    print("\n\n" + "▶ PHASE 1: DATA PREPROCESSING")
    print("─" * 70)
    
    df_clean, df_encoded, df_scaled, encoders = run_full_preprocessing()
    
    # ========================================================================
    # PHASE 2: DECISION TREE CLASSIFICATION
    # ========================================================================
    print("\n\n" + "▶ PHASE 2: DECISION TREE CLASSIFICATION")
    print("─" * 70)
    
    dt_results = run_decision_tree(df_encoded, output_dir)
    
    # ========================================================================
    # PHASE 3: HIERARCHICAL CLUSTERING
    # ========================================================================
    print("\n\n" + "▶ PHASE 3: HIERARCHICAL CLUSTERING")
    print("─" * 70)
    
    hc_results = run_hierarchical_clustering(df_clean, output_dir)
    
    # ========================================================================
    # PHASE 4: REGULARIZED REGRESSION
    # ========================================================================
    print("\n\n" + "▶ PHASE 4: REGULARIZED REGRESSION")
    print("─" * 70)
    
    reg_results = run_regularized_regression(df_encoded, output_dir)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    elapsed = time.time() - start_time
    
    print("\n\n" + "█" * 70)
    print("█" + "  ANALYSIS COMPLETE - SUMMARY".center(68) + "█")
    print("█" * 70)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │  Dataset: {df_clean.shape[0]} records × {df_clean.shape[1]} features{' ' * 27}│
    │  Time elapsed: {elapsed:.1f} seconds{' ' * (45 - len(f'{elapsed:.1f}'))}│
    ├─────────────────────────────────────────────────────────────────┤
    │  DECISION TREE (Attrition Prediction)                          │
    │    Accuracy: {dt_results['metrics']['accuracy']:.4f}   F1: {dt_results['metrics']['f1_score']:.4f}   AUC: {dt_results['metrics']['roc_auc']:.4f}{' ' * 15}│
    │    Top feature: {dt_results['feature_importance'][0]['feature']:<40}│
    ├─────────────────────────────────────────────────────────────────┤
    │  HIERARCHICAL CLUSTERING                                       │
    │    Optimal clusters: {hc_results['optimal_clusters']}{' ' * 42}│
    │    Silhouette score: {hc_results['best_silhouette']:.4f}{' ' * 38}│
    ├─────────────────────────────────────────────────────────────────┤
    │  REGULARIZED REGRESSION (Salary Prediction)                    │
    │    Best model: {reg_results['best_model']:<15} R²: {reg_results['models'][reg_results['best_model'].lower().replace(' ', '_')]['r2']:.4f}{' ' * 22}│
    │    Ridge R²: {reg_results['models']['ridge']['r2']:.4f}  Lasso R²: {reg_results['models']['lasso']['r2']:.4f}  ENet R²: {reg_results['models']['elastic_net']['r2']:.4f} │
    └─────────────────────────────────────────────────────────────────┘
    
    Output files saved in: {output_dir}
    → Launch the visualization dashboard: open dashboard/index.html
    """)
    
    return {
        'preprocessing': {'records': df_clean.shape[0], 'features': df_clean.shape[1]},
        'decision_tree': dt_results,
        'clustering': hc_results,
        'regression': reg_results,
    }


if __name__ == '__main__':
    results = main()
