"""
================================================================================
DECISION TREE CLASSIFICATION
People Analytics - Philippine Workforce Dataset
================================================================================

PURPOSE:
    Predict employee attrition (binary classification: 0 = Stay, 1 = Leave)
    using a Decision Tree Classifier. This model helps HR identify the key
    factors driving employee turnover.

WHY DECISION TREE?
    - Highly interpretable: produces visual rules that HR managers can understand
    - Handles both numerical and categorical features
    - No feature scaling required
    - Captures non-linear relationships and feature interactions
    - Provides feature importance rankings

KEY IMPROVEMENTS (v2):
    - Uses One-Hot Encoding for nominal categoricals (proper representation)
    - Adds class_weight='balanced' to handle imbalanced attrition classes
    - Includes Engagement_Score and Promotion_Rate as derived features
    - Expanded hyperparameter grid with class weighting options

MODEL DETAILS:
    Target Variable: Attrition (0/1)
    Algorithm: CART (Classification and Regression Trees)
    Split Criterion: Gini Impurity + Entropy comparison
    Pruning: Max depth and min samples to prevent overfitting
    Class Weighting: Balanced (compensates for class imbalance)
    Validation: 80/20 train-test split + 5-fold cross-validation
================================================================================
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)


def build_classification_features(df):
    """
    Build proper feature matrix for attrition classification.
    Uses One-Hot Encoding for nominal features (trees can handle this
    properly since each binary column becomes an independent split candidate).
    
    Returns:
        X: Feature DataFrame
        feature_names: List of feature column names
    """
    X_parts = []
    
    # --- Core numerical features ---
    numerical_cols = [
        'Age', 'Tenure_Years', 'Monthly_Salary_PHP', 'Performance_Score',
        'Training_Hours_YTD', 'Absences_YTD', 'Overtime_Hours_Monthly',
        'Distance_Office_KM', 'Job_Satisfaction_Score', 'Work_Life_Balance_Score',
        'Num_Promotions', 'Prev_Companies'
    ]
    num_available = [c for c in numerical_cols if c in df.columns]
    X_num = df[num_available].fillna(df[num_available].median())
    X_parts.append(X_num)
    
    # --- Derived features ---
    if 'Job_Satisfaction_Score' in df.columns and 'Work_Life_Balance_Score' in df.columns:
        engagement = (df['Job_Satisfaction_Score'].fillna(df['Job_Satisfaction_Score'].median()) +
                      df['Work_Life_Balance_Score'].fillna(df['Work_Life_Balance_Score'].median())) / 2
        X_parts.append(engagement.to_frame('Engagement_Score'))
    
    if 'Num_Promotions' in df.columns and 'Tenure_Years' in df.columns:
        tenure = df['Tenure_Years'].fillna(df['Tenure_Years'].median()).replace(0, 0.5)
        X_parts.append((df['Num_Promotions'].fillna(0) / tenure).to_frame('Promotion_Rate'))
    
    # --- Ordinal features ---
    if 'Education_Level' in df.columns:
        edu_map = {'High School': 1, 'Vocational': 2, 'Bachelor': 3, 'Master': 4, 'PhD': 5}
        X_parts.append(df['Education_Level'].map(edu_map).fillna(3).to_frame('Education_Ordinal'))
    
    if 'Performance_Rating' in df.columns:
        perf_map = {'Needs Improvement': 1, 'Meets Expectations': 2,
                    'Exceeds Expectations': 3, 'Outstanding': 4}
        X_parts.append(df['Performance_Rating'].map(perf_map).fillna(2).to_frame('PerfRating_Ordinal'))
    
    # --- One-Hot Encoding for nominal categoricals ---
    nominal_cols = ['Department', 'Gender', 'Employment_Type', 'Shift',
                    'Marital_Status', 'Region']
    
    for col in nominal_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            X_parts.append(dummies)
    
    X = pd.concat(X_parts, axis=1)
    X = X.fillna(0)
    
    return X, list(X.columns)


def run_decision_tree(df_encoded, output_dir):
    """
    Execute Decision Tree Classification for Attrition Prediction.
    
    Parameters:
        df_encoded: Preprocessed DataFrame
        output_dir: Directory to save results
    
    Returns:
        dict: Comprehensive results including metrics, feature importance, and tree rules
    """
    print("=" * 70)
    print("  DECISION TREE CLASSIFICATION")
    print("  Predicting Employee Attrition")
    print("=" * 70)
    
    # ========================================================================
    # 1. PREPARE DATA
    # ========================================================================
    print("\n[1] Building feature matrix (One-Hot + Ordinal encoding)...")
    
    target = 'Attrition'
    X, feature_cols = build_classification_features(df_encoded)
    y = df_encoded[target]
    
    print(f"    Features: {len(feature_cols)}")
    print(f"    Samples: {len(X)}")
    print(f"    Class distribution:")
    print(f"      Stay (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"      Leave (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    
    # ========================================================================
    # 2. TRAIN-TEST SPLIT
    # ========================================================================
    print("\n[2] Splitting data (80/20)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"    Training set: {len(X_train)} samples")
    print(f"    Test set: {len(X_test)} samples")
    
    # ========================================================================
    # 3. HYPERPARAMETER TUNING with GridSearch
    # ========================================================================
    print("\n[3] Hyperparameter tuning (GridSearchCV)...")
    
    param_grid = {
        'max_depth': [5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None]
    }
    
    dt_base = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        dt_base, param_grid, cv=5, scoring='f1',
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print(f"    Best parameters: {best_params}")
    print(f"    Best CV F1 score: {grid_search.best_score_:.4f}")
    
    # ========================================================================
    # 4. TRAIN BEST MODEL
    # ========================================================================
    print("\n[4] Training best model...")
    
    best_dt = grid_search.best_estimator_
    
    # ========================================================================
    # 5. EVALUATE MODEL
    # ========================================================================
    print("\n[5] Evaluating model...")
    
    y_pred = best_dt.predict(X_test)
    y_pred_proba = best_dt.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n    PERFORMANCE METRICS:")
    print(f"    {'-' * 40}")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    ROC AUC:   {roc_auc:.4f}")
    print(f"\n    Confusion Matrix:")
    print(f"    {cm}")
    
    # ========================================================================
    # 6. CROSS-VALIDATION
    # ========================================================================
    print("\n[6] 5-Fold Cross-Validation...")
    
    cv_scores = cross_val_score(best_dt, X, y, cv=5, scoring='f1')
    print(f"    CV F1 Scores: {cv_scores.round(4)}")
    print(f"    Mean CV F1: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")
    
    # ========================================================================
    # 7. FEATURE IMPORTANCE
    # ========================================================================
    print("\n[7] Feature Importance:")
    
    importances = best_dt.feature_importances_
    feat_importance = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1], reverse=True
    )
    
    print(f"    {'Feature':<35} {'Importance':<10}")
    print(f"    {'-' * 45}")
    for feat, imp in feat_importance[:15]:
        bar = '#' * int(imp * 50)
        print(f"    {feat:<35} {imp:.4f} {bar}")
    
    # ========================================================================
    # 8. DECISION RULES (Text representation)
    # ========================================================================
    print("\n[8] Top Decision Rules:")
    
    # Create a simpler tree for interpretable rules
    simple_params = {k: v for k, v in best_params.items() if k not in ['max_depth']}
    simple_dt = DecisionTreeClassifier(max_depth=4, random_state=42, **simple_params)
    simple_dt.fit(X_train, y_train)
    tree_rules = export_text(simple_dt, feature_names=feature_cols, max_depth=4)
    print(f"    (Showing simplified tree with max_depth=4)")
    for line in tree_rules.split('\n')[:20]:
        print(f"    {line}")
    
    # ========================================================================
    # 9. SAVE RESULTS
    # ========================================================================
    # Convert best_params to JSON-serializable format
    json_params = {}
    for k, v in best_params.items():
        if v is None:
            json_params[k] = None
        else:
            json_params[k] = v
    
    results = {
        'model': 'Decision Tree Classifier',
        'target': 'Attrition',
        'best_params': json_params,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
        },
        'confusion_matrix': cm.tolist(),
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'feature_importance': [
            {'feature': feat, 'importance': float(imp)}
            for feat, imp in feat_importance[:20]  # Top 20 for dashboard
        ],
        'class_distribution': {
            'stay': int((y == 0).sum()),
            'leave': int((y == 1).sum()),
        },
        'tree_rules': tree_rules,
    }
    
    output_path = os.path.join(output_dir, 'decision_tree_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n    Results saved to: {output_path}")
    print("\n[OK] Decision Tree Classification complete!")
    
    return results


if __name__ == '__main__':
    from data_preprocessing import run_full_preprocessing
    df_clean, df_encoded, df_scaled, encoders = run_full_preprocessing()
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    results = run_decision_tree(df_encoded, output_dir)
