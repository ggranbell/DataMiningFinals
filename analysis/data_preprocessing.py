"""
================================================================================
DATA PREPROCESSING MODULE
People Analytics - Philippine Workforce Dataset (5,025 Records)
================================================================================

This module handles all data cleaning, transformation, and preprocessing steps
required before applying data mining techniques.

Data Quality Issues Identified:
1. Missing values across multiple columns
2. Typographical inconsistencies in Department names (e.g., 'SALES', 'Saless', 'I.T.', 'hr', 'Ops', 'Operationsn')
3. Impossible/outlier values (Age=1, Absences=-5, Distance=500km, Salary=1500)
4. Negative values in Absences_YTD (e.g., -5, -1)
5. Extreme outlier values in Absences_YTD (e.g., 99, 150)
6. Duplicate employee IDs
7. Mixed case encoding in categorical fields

Preprocessing Pipeline:
Step 1: Load and inspect raw data
Step 2: Fix typographical inconsistencies
Step 3: Handle impossible/outlier values
Step 4: Handle missing values
Step 5: Feature engineering
Step 6: Encode categorical variables
Step 7: Scale numerical features
Step 8: Export cleaned dataset
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import warnings
import os
import json

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         'Data Mining_Final Exam_Workforce Dataset.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_raw_data():
    """
    Step 1: Load raw dataset and perform initial inspection.
    
    Returns:
        pd.DataFrame: Raw dataset
        dict: Initial data quality report
    """
    print("=" * 70)
    print("STEP 1: LOADING RAW DATA")
    print("=" * 70)
    
    df = pd.read_csv(DATA_PATH)
    
    quality_report = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'duplicates': {
            'total_duplicate_rows': int(df.duplicated().sum()),
            'duplicate_employee_ids': int(df['Employee_ID'].duplicated().sum())
        }
    }
    
    print(f"  Records loaded: {len(df)}")
    print(f"  Features: {len(df.columns)}")
    print(f"  Duplicate Employee IDs: {quality_report['duplicates']['duplicate_employee_ids']}")
    print(f"\n  Missing values per column:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"    {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
    return df, quality_report


def fix_typographical_inconsistencies(df):
    """
    Step 2: Fix typographical inconsistencies in categorical columns.
    
    Issues found and corrected:
    - Department: 'SALES' -> 'Sales', 'I.T.' -> 'IT', 'finance' -> 'Finance',
                  'hr' -> 'HR', 'Ops' -> 'Operations', 'Operationsn' -> 'Operations',
                  'OPERATIONS' -> 'Operations', 'Saless' -> 'Sales'
    - Gender, Marital_Status, Region: Standardize capitalization
    """
    print("\n" + "=" * 70)
    print("STEP 2: FIXING TYPOGRAPHICAL INCONSISTENCIES")
    print("=" * 70)
    
    df = df.copy()
    
    # --- Department standardization ---
    # We found many variations in the raw data that need to be grouped
    dept_mapping = {
        # IT variations
        'it': 'IT',
        'I.T.': 'IT',
        'i.t.': 'IT',
        'Information Technology': 'IT',
        
        # HR variations
        'hr': 'HR',
        'H.R.': 'HR',
        'h.r.': 'HR',
        'Human Resources': 'HR',
        
        # Finance variations
        'finance': 'Finance',
        'FINANCE': 'Finance',
        'Fiance': 'Finance', # Common typo
        
        # Operations variations
        'operations': 'Operations',
        'OPERATIONS': 'Operations',
        'Ops': 'Operations',
        'Operationsn': 'Operations', # Common typo
        
        # Sales variations
        'sales': 'Sales',
        'SALES': 'Sales',
        'Saless': 'Sales', # Common typo
    }
    
    original_depts = df['Department'].unique()
    # Initial cleanup: strip whitespace and handle known mappings
    df['Department'] = df['Department'].str.strip()
    
    # We apply mapping first, then check case-insensitive matches for safety
    df['Department'] = df['Department'].replace(dept_mapping)
    
    # Second pass: ensure everything is properly title-cased except known abbreviations
    def final_cleanup(d):
        if pd.isna(d): return d
        # Standardize known abbreviations to uppercase
        if d.upper() in ['IT', 'HR']:
            return d.upper()
        # Everything else to Title Case
        return d.title()
        
    df['Department'] = df['Department'].apply(final_cleanup)
    fixed_depts = df['Department'].unique()
    
    changes = set(original_depts) - set(fixed_depts)
    print(f"  Department corrections applied: {len(dept_mapping)} mappings")
    print(f"  Unique departments before: {len(original_depts)}")
    print(f"  Unique departments after: {len(fixed_depts)}")
    print(f"  Final departments: {sorted(fixed_depts)}")
    
    # --- Gender standardization ---
    df['Gender'] = df['Gender'].str.strip().str.title()
    print(f"  Gender values: {sorted(df['Gender'].dropna().unique())}")
    
    # --- Marital Status standardization ---
    df['Marital_Status'] = df['Marital_Status'].str.strip().str.title()
    print(f"  Marital Status values: {sorted(df['Marital_Status'].dropna().unique())}")
    
    # --- Region standardization ---
    df['Region'] = df['Region'].str.strip().str.upper()
    # Fix common region abbreviation inconsistencies
    df['Region'] = df['Region'].replace({
        'REGION IV-A': 'Region IV-A',
        'REGION III': 'Region III',
        'REGION VII': 'Region VII',
        'REGION XI': 'Region XI',
        'REGION I': 'Region I',
        'REGION VI': 'Region VI',
    })
    # Actually keep them consistent with title case
    region_mapping = {}
    for r in df['Region'].unique():
        if r == 'NCR':
            region_mapping[r] = 'NCR'
        elif r == 'CAR':
            region_mapping[r] = 'CAR'
        else:
            region_mapping[r] = r.title().replace('Iv-A', 'IV-A').replace('Iii', 'III').replace('Vii', 'VII').replace('Xi', 'XI').replace('Vi', 'VI')
    df['Region'] = df['Region'].replace(region_mapping)
    print(f"  Region values: {sorted(df['Region'].dropna().unique())}")
    
    # --- Education Level standardization ---
    df['Education_Level'] = df['Education_Level'].str.strip().str.title()
    df['Education_Level'] = df['Education_Level'].replace({'Phd': 'PhD'})
    print(f"  Education Level values: {sorted(df['Education_Level'].dropna().unique())}")
    
    # --- Employment Type standardization ---
    df['Employment_Type'] = df['Employment_Type'].str.strip().str.title()
    df['Employment_Type'] = df['Employment_Type'].replace({'Project-Based': 'Project-based'})
    print(f"  Employment Type values: {sorted(df['Employment_Type'].dropna().unique())}")
    
    # --- Shift standardization ---
    df['Shift'] = df['Shift'].str.strip().str.title()
    print(f"  Shift values: {sorted(df['Shift'].dropna().unique())}")
    
    # --- Performance Rating standardization ---
    df['Performance_Rating'] = df['Performance_Rating'].str.strip()
    print(f"  Performance Rating values: {sorted(df['Performance_Rating'].dropna().unique())}")
    
    return df


def handle_impossible_values(df):
    """
    Step 3: Identify and handle impossible/outlier values.
    
    Rules Applied:
    - Age: Must be 18-65 (working age in Philippines). Values <18 or >65 are capped.
           Age=1 is clearly impossible -> set to NaN for imputation
    - Absences_YTD: Cannot be negative -> set negative values to 0
                    Values >= 99 are likely data entry errors -> cap at reasonable max
    - Distance_Office_KM: Values > 300km are suspicious for PH context -> cap at 300
    - Monthly_Salary_PHP: Values < 5000 are below minimum wage -> flag as outlier
    - Overtime_Hours_Monthly: Values > 60 are extreme -> cap at 60
    - Training_Hours_YTD: Values < 0 or > 100 are suspicious
    """
    print("\n" + "=" * 70)
    print("STEP 3: HANDLING IMPOSSIBLE/OUTLIER VALUES")
    print("=" * 70)
    
    df = df.copy()
    corrections = {}
    
    # --- Age corrections ---
    impossible_age = df[(df['Age'] < 18) | (df['Age'] > 65)]
    print(f"  Age issues found: {len(impossible_age)} records")
    if len(impossible_age) > 0:
        print(f"    Problematic ages: {sorted(impossible_age['Age'].unique())}")
    # Age = 1 is clearly a data entry error -> NaN for imputation
    df.loc[df['Age'] < 18, 'Age'] = np.nan
    corrections['age_fixed'] = len(impossible_age)
    
    # --- Absences_YTD corrections ---
    negative_absences = df[df['Absences_YTD'] < 0]
    extreme_absences = df[df['Absences_YTD'] >= 99]
    print(f"  Negative absences: {len(negative_absences)} records (set to 0)")
    print(f"  Extreme absences (>=99): {len(extreme_absences)} records (capped at 30)")
    df.loc[df['Absences_YTD'] < 0, 'Absences_YTD'] = 0
    df.loc[df['Absences_YTD'] >= 99, 'Absences_YTD'] = 30  # Cap at reasonable max
    corrections['absences_negative_fixed'] = len(negative_absences)
    corrections['absences_extreme_fixed'] = len(extreme_absences)
    
    # --- Distance_Office_KM corrections ---
    extreme_distance = df[df['Distance_Office_KM'] > 300]
    print(f"  Extreme distances (>300km): {len(extreme_distance)} records (capped at 300)")
    df.loc[df['Distance_Office_KM'] > 300, 'Distance_Office_KM'] = 300
    corrections['distance_capped'] = len(extreme_distance)
    
    # --- Monthly_Salary_PHP corrections ---
    low_salary = df[df['Monthly_Salary_PHP'] < 5000]
    print(f"  Suspiciously low salaries (<5000): {len(low_salary)} records")
    if len(low_salary) > 0:
        print(f"    Low salary values: {sorted(low_salary['Monthly_Salary_PHP'].unique())}")
    # Set suspiciously low salaries to NaN for imputation
    df.loc[df['Monthly_Salary_PHP'] < 5000, 'Monthly_Salary_PHP'] = np.nan
    corrections['salary_low_fixed'] = len(low_salary)
    
    # Extreme high salaries (placeholder/error values like 999999, 9999999)
    # Max realistic PH monthly salary in dataset context: ~120,000 (PhD-level IT)
    high_salary = df[df['Monthly_Salary_PHP'] > 120000]
    print(f"  Suspiciously high salaries (>120K): {len(high_salary)} records (set to NaN)")
    if len(high_salary) > 0:
        print(f"    High salary values: {sorted(high_salary['Monthly_Salary_PHP'].unique())}")
    df.loc[df['Monthly_Salary_PHP'] > 120000, 'Monthly_Salary_PHP'] = np.nan
    corrections['salary_high_fixed'] = len(high_salary)
    
    # --- Overtime_Hours_Monthly corrections ---
    extreme_ot = df[df['Overtime_Hours_Monthly'] > 60]
    print(f"  Extreme overtime (>60hrs): {len(extreme_ot)} records (capped at 60)")
    df.loc[df['Overtime_Hours_Monthly'] > 60, 'Overtime_Hours_Monthly'] = 60
    corrections['overtime_capped'] = len(extreme_ot)
    
    return df, corrections


def handle_missing_values(df):
    """
    Step 4: Handle missing values using appropriate imputation strategies.
    
    Strategy:
    - Numerical columns: Impute with MEDIAN (robust to outliers)
    - Categorical columns: Impute with MODE
    - Performance_Score (numerical): Impute with median per Performance_Rating group
      (since these are semantically linked)
    """
    print("\n" + "=" * 70)
    print("STEP 4: HANDLING MISSING VALUES")
    print("=" * 70)
    
    df = df.copy()
    
    missing_before = df.isnull().sum()
    print(f"\n  Missing values before imputation:")
    for col in df.columns:
        if missing_before[col] > 0:
            print(f"    {col}: {missing_before[col]}")
    
    # --- Numerical columns: Median imputation ---
    numerical_cols = ['Age', 'Tenure_Years', 'Monthly_Salary_PHP', 'Performance_Score',
                      'Training_Hours_YTD', 'Absences_YTD', 'Overtime_Hours_Monthly',
                      'Distance_Office_KM', 'Job_Satisfaction_Score', 
                      'Work_Life_Balance_Score', 'Num_Promotions', 'Prev_Companies']
    
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            if col == 'Performance_Score':
                # Group-based imputation: use median per Performance_Rating
                df[col] = df.groupby('Performance_Rating')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # If any still missing (due to all-NaN groups), use overall median
                df[col] = df[col].fillna(df[col].median())
            elif col == 'Monthly_Salary_PHP':
                # Group-based: impute by Department + Education_Level
                df[col] = df.groupby(['Department', 'Education_Level'])[col].transform(
                    lambda x: x.fillna(x.median())
                )
                df[col] = df[col].fillna(df[col].median())
            else:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"    {col}: imputed with median = {median_val}")
    
    # --- Categorical columns: Mode imputation ---
    categorical_cols = ['Gender', 'Marital_Status', 'Region', 'Education_Level',
                        'Department', 'Employment_Type', 'Shift', 'Performance_Rating']
    
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"    {col}: imputed with mode = {mode_val}")
    
    missing_after = df.isnull().sum().sum()
    print(f"\n  Total missing values after imputation: {missing_after}")
    
    return df


def handle_duplicates(df):
    """
    Step 4b: Handle duplicate Employee_IDs.
    
    Strategy: Keep the first occurrence, remove duplicates.
    """
    print("\n" + "=" * 70)
    print("STEP 4b: HANDLING DUPLICATES")
    print("=" * 70)
    
    dup_count = df['Employee_ID'].duplicated().sum()
    print(f"  Duplicate Employee_IDs found: {dup_count}")
    
    if dup_count > 0:
        dup_ids = df[df['Employee_ID'].duplicated(keep=False)]['Employee_ID'].unique()
        print(f"  Duplicate IDs: {dup_ids[:10]}...")
        df = df.drop_duplicates(subset='Employee_ID', keep='first').reset_index(drop=True)
        print(f"  Records after deduplication: {len(df)}")
    
    return df


def feature_engineering(df):
    """
    Step 5: Create derived features for enhanced analysis.
    
    Core Features:
    - Hire_Year, Salary_Bracket, Age_Group, High_Performer
    - Engagement_Score, Promotion_Rate, Overtime_Intensity
    
    Advanced Interaction Features (v2 - for improved model accuracy):
    - Tenure_Squared: Captures non-linear salary growth with tenure
    - Salary_Per_Tenure: Salary earned per year of service
    - Dept_Salary_Ratio: Employee salary vs. department median
    - Employment_Risk: Numeric risk score from Employment_Type
    - Perf_Rating_Numeric: Numeric performance rating
    - Satisfaction_x_Performance: Interaction of satisfaction and performance
    - Tenure_x_Promotions: How promotions scale with tenure
    - Absence_Rate: Normalized absences per tenure year
    - Is_Regular: Binary flag for regular employment
    - Shift_Risk: Numeric shift risk (graveyard > night > afternoon > morning)
    - Salary_x_Satisfaction: Salary-satisfaction interaction
    - Tenure_x_Education: Tenure weighted by education level
    """
    print("\n" + "=" * 70)
    print("STEP 5: FEATURE ENGINEERING")
    print("=" * 70)
    
    df = df.copy()
    
    # Convert Hire_Date to datetime
    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], errors='coerce')
    df['Hire_Year'] = df['Hire_Date'].dt.year
    
    # Salary brackets (based on Philippine salary standards)
    df['Salary_Bracket'] = pd.cut(
        df['Monthly_Salary_PHP'],
        bins=[0, 25000, 40000, 60000, 120000],
        labels=['Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )
    
    # Age groups
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[17, 30, 45, 65],
        labels=['Young (18-30)', 'Mid-Career (31-45)', 'Senior (46+)']
    )
    
    # High performer flag (binary)
    df['High_Performer'] = (df['Performance_Score'] >= 4).astype(int)
    
    # Engagement composite score
    df['Engagement_Score'] = (
        df['Job_Satisfaction_Score'] + df['Work_Life_Balance_Score']
    ) / 2
    
    # Promotion rate
    df['Promotion_Rate'] = df['Num_Promotions'] / df['Tenure_Years'].replace(0, 0.5)
    
    # Overtime intensity
    df['Overtime_Intensity'] = pd.cut(
        df['Overtime_Hours_Monthly'],
        bins=[-1, 10, 20, 30, 100],
        labels=['Low', 'Moderate', 'High', 'Very High']
    )
    
    # =========================================================================
    # ADVANCED INTERACTION FEATURES (v2)
    # =========================================================================
    
    # Tenure squared - captures non-linear salary/attrition relationship
    df['Tenure_Squared'] = df['Tenure_Years'] ** 2
    
    # Salary per year of tenure - compensation velocity
    tenure_safe = df['Tenure_Years'].replace(0, 0.5)
    df['Salary_Per_Tenure'] = df['Monthly_Salary_PHP'] / tenure_safe
    
    # Employment type risk score (Project-based 89% attrition, Regular 37%)
    emp_risk = {'Regular': 0, 'Contractual': 1, 'Project-based': 2}
    df['Employment_Risk'] = df['Employment_Type'].map(emp_risk).fillna(1)
    
    # Is Regular employment (binary - strongest single attrition predictor)
    df['Is_Regular'] = (df['Employment_Type'] == 'Regular').astype(int)
    
    # Performance rating as numeric (Needs Improvement: 1 -> Outstanding: 4)
    perf_map = {'Needs Improvement': 1, 'Meets Expectations': 2,
                'Exceeds Expectations': 3, 'Outstanding': 4}
    df['Perf_Rating_Numeric'] = df['Performance_Rating'].map(perf_map).fillna(2)
    
    # Shift risk score (Graveyard 67% attrition -> Morning 50%)
    shift_risk = {'Morning': 0, 'Afternoon': 1, 'Night': 2, 'Graveyard': 3}
    df['Shift_Risk'] = df['Shift'].map(shift_risk).fillna(1)
    
    # Department salary median ratio (is employee over/underpaid vs dept?)
    dept_median = df.groupby('Department')['Monthly_Salary_PHP'].transform('median')
    df['Dept_Salary_Ratio'] = df['Monthly_Salary_PHP'] / dept_median.replace(0, 1)
    
    # Satisfaction × Performance interaction
    df['Satisfaction_x_Performance'] = (
        df['Job_Satisfaction_Score'] * df['Performance_Score']
    )
    
    # Tenure × Number of Promotions interaction
    df['Tenure_x_Promotions'] = df['Tenure_Years'] * df['Num_Promotions']
    
    # Absence rate normalized by tenure
    df['Absence_Rate'] = df['Absences_YTD'] / tenure_safe
    
    # Salary × Satisfaction interaction (underpaid + unhappy = high attrition)
    df['Salary_x_Satisfaction'] = (
        df['Monthly_Salary_PHP'] / 10000 * df['Job_Satisfaction_Score']
    )
    
    # Tenure × Education interaction (educated + tenured = higher salary)
    edu_map = {'High School': 1, 'Vocational': 2, 'Bachelor': 3, 'Master': 4, 'PhD': 5}
    df['Education_Numeric'] = df['Education_Level'].map(edu_map).fillna(3)
    df['Tenure_x_Education'] = df['Tenure_Years'] * df['Education_Numeric']
    
    # Age minus expected age (based on tenure) - career gap indicator
    df['Career_Start_Age'] = df['Age'] - df['Tenure_Years']
    
    # Career Stagnation Flag (No promotions after 5 years -> 97.5% attrition)
    df['Stagnation_Flag'] = ((df['Num_Promotions'] == 0) & (df['Tenure_Years'] > 5)).astype(int)
    
    # Composite Attrition Risk Score (Domain rules - enhanced v2)
    def calc_risk(row):
        score = 0
        if row['Employment_Type'] == 'Project-based': score += 4
        elif row['Employment_Type'] == 'Contractual': score += 3
        if row['Performance_Rating'] == 'Needs Improvement': score += 3
        elif row['Performance_Rating'] == 'Meets Expectations': score += 1
        if row['Num_Promotions'] == 0: score += 4
        elif row['Num_Promotions'] <= 2: score += 2
        if row['Tenure_Years'] <= 5: score += 3
        elif row['Tenure_Years'] <= 10: score += 1
        if row['Job_Satisfaction_Score'] <= 2: score += 2
        elif row['Job_Satisfaction_Score'] <= 3: score += 1
        if row['Work_Life_Balance_Score'] <= 2: score += 1
        if row['Shift'] == 'Graveyard': score += 1
        if row['Num_Promotions'] == 0 and row['Tenure_Years'] > 3: score += 3
        return score
    
    df['Attrition_Risk_Score'] = df.apply(calc_risk, axis=1)
    
    # Stability Score - positive retention indicators
    df['Stability_Score'] = (
        df['Is_Regular'] * 4 +
        np.minimum(df['Num_Promotions'], 8) +
        np.minimum(df['Tenure_Years'] / 5, 6) +
        df['Job_Satisfaction_Score'] / 2 +
        df['Work_Life_Balance_Score'] / 2 +
        df['High_Performer'] * 2
    )
    
    print(f"  Core features: Hire_Year, Salary_Bracket, Age_Group, "
          f"High_Performer, Engagement_Score, Promotion_Rate, Overtime_Intensity")
    print(f"  Advanced features: Tenure_Squared, Employment_Risk, Is_Regular, "
          f"Perf_Rating_Numeric, Shift_Risk, Dept_Salary_Ratio, +9 interactions")
    print(f"  Total features now: {len(df.columns)}")
    
    return df


def encode_and_scale(df):
    """
    Step 6 & 7: Encode categorical variables and scale numerical features.
    
    Encoding:
    - Label Encoding for ordinal categories (Education_Level, Performance_Rating)
    - One-Hot Encoding for nominal categories (Gender, Department, Region, etc.)
    
    Scaling:
    - StandardScaler for regression models
    - MinMaxScaler alternative also saved
    
    Returns:
        df_encoded: Encoded DataFrame (for tree-based models)
        df_scaled: Scaled DataFrame (for regression models)
        encoders: Dictionary of fitted encoders/scalers
    """
    print("\n" + "=" * 70)
    print("STEP 6 & 7: ENCODING & SCALING")
    print("=" * 70)
    
    df = df.copy()
    encoders = {}
    
    # --- Ordinal Encoding ---
    education_order = {'High School': 1, 'Vocational': 2, 'Bachelor': 3, 'Master': 4, 'PhD': 5}
    df['Education_Level_Encoded'] = df['Education_Level'].map(education_order)
    encoders['education'] = education_order
    
    performance_order = {
        'Needs Improvement': 1, 'Meets Expectations': 2,
        'Exceeds Expectations': 3, 'Outstanding': 4
    }
    df['Performance_Rating_Encoded'] = df['Performance_Rating'].map(performance_order)
    encoders['performance_rating'] = performance_order
    
    # --- Label Encoding for other categoricals ---
    label_cols = ['Gender', 'Marital_Status', 'Region', 'Department',
                  'Employment_Type', 'Shift', 'Salary_Bracket', 'Age_Group',
                  'Overtime_Intensity']
    
    for col in label_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Handle potential NaN in categorical columns
            mask = df[col].notna()
            df.loc[mask, f'{col}_Encoded'] = le.fit_transform(df.loc[mask, col].astype(str))
            df[f'{col}_Encoded'] = df[f'{col}_Encoded'].fillna(-1).astype(int)
            encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    # --- Numerical columns for scaling ---
    scale_cols = [
        'Age', 'Tenure_Years', 'Monthly_Salary_PHP', 'Performance_Score',
        'Training_Hours_YTD', 'Absences_YTD', 'Overtime_Hours_Monthly',
        'Distance_Office_KM', 'Job_Satisfaction_Score', 'Work_Life_Balance_Score',
        'Num_Promotions', 'Prev_Companies', 'Engagement_Score', 'Promotion_Rate'
    ]
    
    existing_scale_cols = [c for c in scale_cols if c in df.columns]
    
    # StandardScaler
    scaler = StandardScaler()
    df_scaled_values = scaler.fit_transform(df[existing_scale_cols].fillna(0))
    df_scaled = df.copy()
    for i, col in enumerate(existing_scale_cols):
        df_scaled[f'{col}_Scaled'] = df_scaled_values[:, i]
    
    encoders['standard_scaler'] = {
        'mean': dict(zip(existing_scale_cols, scaler.mean_.tolist())),
        'std': dict(zip(existing_scale_cols, scaler.scale_.tolist()))
    }
    
    print(f"  Ordinal encoded: Education_Level, Performance_Rating")
    print(f"  Label encoded: {', '.join(label_cols)}")
    print(f"  Scaled columns: {len(existing_scale_cols)}")
    
    return df, df_scaled, encoders


def run_full_preprocessing():
    """
    Execute the complete preprocessing pipeline.
    
    Returns:
        df_clean: Cleaned DataFrame (before encoding)
        df_encoded: Encoded DataFrame
        df_scaled: Scaled DataFrame
        preprocessing_report: Comprehensive report
    """
    print("╔" + "═" * 68 + "╗")
    print("║" + "  PEOPLE ANALYTICS - DATA PREPROCESSING PIPELINE".center(68) + "║")
    print("║" + "  Philippine Workforce Dataset (5,025 Records)".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Step 1: Load
    df_raw, quality_report = load_raw_data()
    
    # Step 2: Fix typos
    df_fixed = fix_typographical_inconsistencies(df_raw)
    
    # Step 3: Handle impossible values
    df_corrected, corrections = handle_impossible_values(df_fixed)
    
    # Step 4: Handle missing values
    df_imputed = handle_missing_values(df_corrected)
    
    # Step 4b: Handle duplicates
    df_deduped = handle_duplicates(df_imputed)
    
    # Step 5: Feature engineering
    df_engineered = feature_engineering(df_deduped)
    
    # Step 6 & 7: Encode and scale
    df_encoded, df_scaled, encoders = encode_and_scale(df_engineered)
    
    # Save outputs
    df_engineered.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_dataset.csv'), index=False)
    df_encoded.to_csv(os.path.join(OUTPUT_DIR, 'encoded_dataset.csv'), index=False)
    
    # Save summary statistics as JSON for the visualization UI
    summary = generate_summary_stats(df_engineered)
    with open(os.path.join(OUTPUT_DIR, 'data_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Final dataset shape: {df_encoded.shape}")
    print(f"  Outputs saved to: {OUTPUT_DIR}")
    
    return df_engineered, df_encoded, df_scaled, encoders


def generate_summary_stats(df):
    """Generate summary statistics for the visualization dashboard."""
    summary = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'numerical_summary': {},
        'categorical_summary': {},
        'attrition_rate': float(df['Attrition'].mean() * 100),
    }
    
    num_cols = ['Age', 'Tenure_Years', 'Monthly_Salary_PHP', 'Performance_Score',
                'Training_Hours_YTD', 'Absences_YTD', 'Overtime_Hours_Monthly',
                'Distance_Office_KM', 'Job_Satisfaction_Score', 'Work_Life_Balance_Score',
                'Num_Promotions', 'Prev_Companies']
    
    for col in num_cols:
        if col in df.columns:
            summary['numerical_summary'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
            }
    
    cat_cols = ['Gender', 'Marital_Status', 'Region', 'Education_Level',
                'Department', 'Employment_Type', 'Shift', 'Performance_Rating']
    
    for col in cat_cols:
        if col in df.columns:
            summary['categorical_summary'][col] = df[col].value_counts().to_dict()
    
    return summary


if __name__ == '__main__':
    df_clean, df_encoded, df_scaled, encoders = run_full_preprocessing()
    print("\n✓ Preprocessing pipeline executed successfully!")
