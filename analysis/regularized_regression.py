"""
================================================================================
REGULARIZED REGRESSION MODELS
People Analytics - Philippine Workforce Dataset
================================================================================

PURPOSE:
    Predict employee monthly salary (Monthly_Salary_PHP) using regularized
    regression techniques. Regularization helps prevent overfitting and
    performs automatic feature selection, revealing which factors truly
    drive compensation in the organization.

THREE MODELS IMPLEMENTED:

1. RIDGE REGRESSION (L2 Regularization)
   - Adds squared penalty: lambda * sum(beta^2)
   - Shrinks all coefficients toward zero but never exactly to zero
   - Best when all features contribute; controls multicollinearity
   - Good for: understanding relative importance of ALL features

2. LASSO REGRESSION (L1 Regularization)
   - Adds absolute penalty: lambda * sum(|beta|)
   - Can shrink coefficients EXACTLY to zero -> automatic feature selection
   - Best when only a few features are truly important
   - Good for: identifying the MOST critical salary predictors

3. ELASTIC NET REGRESSION (L1 + L2 Combined)
   - Combines both penalties
   - Balance controlled by l1_ratio (0=Ridge, 1=Lasso)
   - Best when features are correlated AND feature selection is desired
   - Good for: robust prediction with mixed feature types

KEY FIX (v2):
   Previous version used LabelEncoder for nominal features like Department
   and Gender, which creates FALSE ordinal relationships (e.g., IT=3 is not
   "between" HR=2 and Sales=4). This version uses One-Hot Encoding for all
   nominal categoricals, which properly represents them as independent 
   binary flags and dramatically improves model performance.

MODEL DETAILS:
    Target Variable: Monthly_Salary_PHP
    Feature Scaling: StandardScaler (required for regularization)
    Categorical Encoding: One-Hot (nominal) + Ordinal (Education, Performance)
    Hyperparameter Tuning: Cross-validated alpha search
    Validation: 80/20 split + 5-fold cross-validation
    Metrics: R2, MAE, RMSE, Adjusted R2
================================================================================
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)


def build_regression_features(df):
    """
    Build a proper feature matrix for regression using:
    - Numerical features as-is
    - Ordinal encoding for Education_Level and Performance_Rating
    - One-Hot Encoding for nominal categoricals (Department, Gender, etc.)
    
    Returns:
        X: Feature DataFrame
        feature_names: List of feature column names
    """
    X_parts = []
    
    # --- Numerical features (direct predictors of salary) ---
    numerical_cols = [
        'Age', 'Tenure_Years', 'Performance_Score',
        'Training_Hours_YTD', 'Absences_YTD', 'Overtime_Hours_Monthly',
        'Distance_Office_KM', 'Job_Satisfaction_Score', 'Work_Life_Balance_Score',
        'Num_Promotions', 'Prev_Companies'
    ]
    num_available = [c for c in numerical_cols if c in df.columns]
    X_num = df[num_available].fillna(df[num_available].median())
    X_parts.append(X_num)
    
    # --- Ordinal features (natural ordering exists) ---
    if 'Education_Level' in df.columns:
        edu_map = {'High School': 1, 'Vocational': 2, 'Bachelor': 3, 'Master': 4, 'PhD': 5}
        X_parts.append(df['Education_Level'].map(edu_map).fillna(3).to_frame('Education_Ordinal'))
    
    if 'Performance_Rating' in df.columns:
        perf_map = {'Needs Improvement': 1, 'Meets Expectations': 2,
                    'Exceeds Expectations': 3, 'Outstanding': 4}
        X_parts.append(df['Performance_Rating'].map(perf_map).fillna(2).to_frame('PerfRating_Ordinal'))
    
    # --- One-Hot Encoding for NOMINAL categoricals ---
    # This is the critical fix: these have NO natural order
    nominal_cols = ['Department', 'Gender', 'Employment_Type', 'Shift',
                    'Marital_Status', 'Region']
    
    for col in nominal_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            X_parts.append(dummies)
    
    # --- Interaction features (domain knowledge) ---
    if 'Tenure_Years' in df.columns and 'Education_Level' in df.columns:
        edu_ord = df['Education_Level'].map(
            {'High School': 1, 'Vocational': 2, 'Bachelor': 3, 'Master': 4, 'PhD': 5}
        ).fillna(3)
        X_parts.append((df['Tenure_Years'].fillna(df['Tenure_Years'].median()) * edu_ord).to_frame('Tenure_x_Education'))
    
    if 'Num_Promotions' in df.columns and 'Tenure_Years' in df.columns:
        tenure = df['Tenure_Years'].fillna(df['Tenure_Years'].median()).replace(0, 0.5)
        X_parts.append((df['Num_Promotions'].fillna(0) / tenure).to_frame('Promotion_Rate'))
    
    X = pd.concat(X_parts, axis=1)
    X = X.fillna(0)
    
    return X, list(X.columns)


def run_regularized_regression(df_encoded, output_dir):
    """
    Execute all three regularized regression models for Salary Prediction.
    
    Parameters:
        df_encoded: Preprocessed DataFrame (uses raw categorical columns for OHE)
        output_dir: Directory to save results
    
    Returns:
        dict: Comprehensive results for all three models
    """
    print("=" * 70)
    print("  REGULARIZED REGRESSION ANALYSIS")
    print("  Predicting Monthly Salary (PHP)")
    print("=" * 70)
    
    # ========================================================================
    # 1. PREPARE DATA with proper encoding
    # ========================================================================
    print("\n[1] Building feature matrix (One-Hot + Ordinal encoding)...")
    
    target = 'Monthly_Salary_PHP'
    
    # Remove rows where target is missing
    df_work = df_encoded.dropna(subset=[target]).copy()
    
    X, feature_names = build_regression_features(df_work)
    y = df_work[target].values
    
    print(f"    Total features: {len(feature_names)}")
    print(f"    Samples: {len(X)}")
    print(f"    Target stats: Mean=PHP {y.mean():,.0f}, Median=PHP {np.median(y):,.0f}, Std=PHP {y.std():,.0f}")
    
    # Show feature breakdown
    n_num = sum(1 for f in feature_names if not any(f.startswith(p+'_') for p in ['Department','Gender','Employment_Type','Shift','Marital_Status','Region']))
    n_ohe = len(feature_names) - n_num
    print(f"    Numerical/Ordinal features: {n_num}")
    print(f"    One-Hot encoded features: {n_ohe}")
    
    # ========================================================================
    # 2. TRAIN-TEST SPLIT
    # ========================================================================
    print("\n[2] Splitting data (80/20)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"    Training: {len(X_train)}, Test: {len(X_test)}")
    
    # ========================================================================
    # 3. SCALE FEATURES
    # ========================================================================
    print("\n[3] Scaling features (StandardScaler)...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ========================================================================
    # 4. RIDGE REGRESSION
    # ========================================================================
    print("\n" + "-" * 70)
    print("  4A. RIDGE REGRESSION (L2 Regularization)")
    print("-" * 70)
    
    # Reasonable alpha range for salary data
    alphas = np.logspace(-2, 4, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_absolute_error')
    ridge_cv.fit(X_train_scaled, y_train)
    
    ridge_pred = ridge_cv.predict(X_test_scaled)
    ridge_results = _evaluate_model(ridge_cv, X_train_scaled, X_test_scaled,
                                     y_train, y_test, ridge_pred, feature_names, 'Ridge')
    
    print(f"\n    Best Alpha: {ridge_cv.alpha_:.4f}")
    print(f"    R2 Score: {ridge_results['r2']:.4f}")
    print(f"    Adjusted R2: {ridge_results['adjusted_r2']:.4f}")
    print(f"    MAE: PHP {ridge_results['mae']:,.0f}")
    print(f"    RMSE: PHP {ridge_results['rmse']:,.0f}")
    
    # ========================================================================
    # 5. LASSO REGRESSION
    # ========================================================================
    print("\n" + "-" * 70)
    print("  4B. LASSO REGRESSION (L1 Regularization)")
    print("-" * 70)
    
    lasso_alphas = np.logspace(-2, 3, 100)
    lasso_cv = LassoCV(alphas=lasso_alphas, cv=5, random_state=42, max_iter=20000)
    lasso_cv.fit(X_train_scaled, y_train)
    
    lasso_pred = lasso_cv.predict(X_test_scaled)
    lasso_results = _evaluate_model(lasso_cv, X_train_scaled, X_test_scaled,
                                     y_train, y_test, lasso_pred, feature_names, 'Lasso')
    
    zero_coefs = sum(1 for c in lasso_cv.coef_ if abs(c) < 1e-10)
    
    print(f"\n    Best Alpha: {lasso_cv.alpha_:.4f}")
    print(f"    R2 Score: {lasso_results['r2']:.4f}")
    print(f"    Adjusted R2: {lasso_results['adjusted_r2']:.4f}")
    print(f"    MAE: PHP {lasso_results['mae']:,.0f}")
    print(f"    RMSE: PHP {lasso_results['rmse']:,.0f}")
    print(f"    Features eliminated: {zero_coefs}/{len(feature_names)}")
    
    lasso_results['features_eliminated'] = zero_coefs
    lasso_results['features_retained'] = len(feature_names) - zero_coefs
    
    # ========================================================================
    # 6. ELASTIC NET REGRESSION
    # ========================================================================
    print("\n" + "-" * 70)
    print("  4C. ELASTIC NET REGRESSION (L1 + L2 Combined)")
    print("-" * 70)
    
    enet_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        alphas=lasso_alphas, cv=5, random_state=42, max_iter=20000
    )
    enet_cv.fit(X_train_scaled, y_train)
    
    enet_pred = enet_cv.predict(X_test_scaled)
    enet_results = _evaluate_model(enet_cv, X_train_scaled, X_test_scaled,
                                    y_train, y_test, enet_pred, feature_names, 'ElasticNet')
    
    zero_coefs_enet = sum(1 for c in enet_cv.coef_ if abs(c) < 1e-10)
    
    print(f"\n    Best Alpha: {enet_cv.alpha_:.4f}")
    print(f"    Best L1 Ratio: {enet_cv.l1_ratio_:.2f}")
    print(f"    R2 Score: {enet_results['r2']:.4f}")
    print(f"    Adjusted R2: {enet_results['adjusted_r2']:.4f}")
    print(f"    MAE: PHP {enet_results['mae']:,.0f}")
    print(f"    RMSE: PHP {enet_results['rmse']:,.0f}")
    print(f"    Features eliminated: {zero_coefs_enet}/{len(feature_names)}")
    
    enet_results['best_l1_ratio'] = float(enet_cv.l1_ratio_)
    enet_results['features_eliminated'] = zero_coefs_enet
    enet_results['features_retained'] = len(feature_names) - zero_coefs_enet
    
    # ========================================================================
    # 7. MODEL COMPARISON
    # ========================================================================
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"\n    {'Model':<20} {'R2':<10} {'Adj R2':<10} {'MAE':<14} {'RMSE':<14} {'CV R2':<10}")
    print(f"    {'-' * 72}")
    
    for name, res in [('Ridge', ridge_results), ('Lasso', lasso_results), ('Elastic Net', enet_results)]:
        print(f"    {name:<20} {res['r2']:<10.4f} {res['adjusted_r2']:<10.4f} "
              f"{res['mae']:<14,.0f} {res['rmse']:<14,.0f} {res['cv_r2_mean']:<10.4f}")
    
    best_model = max(
        [('Ridge', ridge_results), ('Lasso', lasso_results), ('Elastic Net', enet_results)],
        key=lambda x: x[1]['r2']
    )
    print(f"\n    -> Best Model: {best_model[0]} (R2 = {best_model[1]['r2']:.4f})")
    
    # ========================================================================
    # 8. TOP SALARY PREDICTORS
    # ========================================================================
    print("\n[8] Top Salary Predictors (by |coefficient|):")
    
    all_coefs = {}
    for name, res in [('Ridge', ridge_results), ('Lasso', lasso_results), ('Elastic Net', enet_results)]:
        for item in res['coefficients']:
            feat = item['feature']
            if feat not in all_coefs:
                all_coefs[feat] = {}
            all_coefs[feat][name] = item['coefficient']
    
    sorted_coefs = sorted(ridge_results['coefficients'],
                          key=lambda x: abs(x['coefficient']), reverse=True)
    
    print(f"\n    {'Feature':<35} {'Ridge':<12} {'Lasso':<12} {'ElasticNet':<12}")
    print(f"    {'-' * 71}")
    for item in sorted_coefs[:15]:
        feat = item['feature']
        r_coef = all_coefs[feat].get('Ridge', 0)
        l_coef = all_coefs[feat].get('Lasso', 0)
        e_coef = all_coefs[feat].get('Elastic Net', 0)
        print(f"    {feat:<35} {r_coef:>10,.0f} {l_coef:>10,.0f} {e_coef:>10,.0f}")
    
    # ========================================================================
    # 9. SAVE RESULTS
    # ========================================================================
    # Only keep top 20 coefficients for JSON (to avoid huge files with OHE)
    for res in [ridge_results, lasso_results, enet_results]:
        res['coefficients'] = sorted(res['coefficients'],
                                      key=lambda x: abs(x['coefficient']), reverse=True)[:20]
    
    results = {
        'analysis': 'Regularized Regression - Salary Prediction',
        'target': 'Monthly_Salary_PHP',
        'target_stats': {
            'mean': float(y.mean()),
            'median': float(np.median(y)),
            'std': float(y.std()),
        },
        'feature_columns': feature_names[:20],  # top features for dashboard
        'total_features': len(feature_names),
        'models': {
            'ridge': ridge_results,
            'lasso': lasso_results,
            'elastic_net': enet_results,
        },
        'best_model': best_model[0],
        'comparison': {
            'Ridge': {'r2': ridge_results['r2'], 'mae': ridge_results['mae']},
            'Lasso': {'r2': lasso_results['r2'], 'mae': lasso_results['mae']},
            'Elastic Net': {'r2': enet_results['r2'], 'mae': enet_results['mae']},
        }
    }
    
    output_path = os.path.join(output_dir, 'regression_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n    Results saved to: {output_path}")
    print("\n[OK] Regularized Regression Analysis complete!")
    
    return results


def _evaluate_model(model, X_train, X_test, y_train, y_test, y_pred,
                     feature_cols, model_name):
    """
    Evaluate a regression model and return comprehensive metrics.
    """
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Adjusted R2
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Cross-validation R2 on training data only (proper methodology)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Coefficients
    coefficients = []
    if hasattr(model, 'coef_'):
        for feat, coef in zip(feature_cols, model.coef_):
            coefficients.append({
                'feature': feat,
                'coefficient': float(coef),
                'abs_coefficient': float(abs(coef)),
                'is_zero': bool(abs(coef) < 1e-10)
            })
        coefficients.sort(key=lambda x: x['abs_coefficient'], reverse=True)
    
    alpha = float(model.alpha_) if hasattr(model, 'alpha_') else None
    
    return {
        'model_name': model_name,
        'alpha': alpha,
        'r2': float(r2),
        'adjusted_r2': float(adj_r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'cv_r2_scores': cv_scores.tolist(),
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'coefficients': coefficients,
        'intercept': float(model.intercept_) if hasattr(model, 'intercept_') else None,
    }


if __name__ == '__main__':
    from data_preprocessing import run_full_preprocessing
    df_clean, df_encoded, df_scaled, encoders = run_full_preprocessing()
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    results = run_regularized_regression(df_encoded, output_dir)
