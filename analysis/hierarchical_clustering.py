"""
================================================================================
HIERARCHICAL CLUSTERING
People Analytics - Philippine Workforce Dataset
================================================================================

PURPOSE:
    Segment employees into meaningful clusters based on their performance,
    compensation, engagement, and tenure profiles. This helps HR design
    targeted retention, development, and compensation strategies.

WHY HIERARCHICAL CLUSTERING?
    - Produces a dendrogram showing the hierarchical relationships between groups
    - No need to pre-specify the number of clusters
    - Reveals nested cluster structure (employees within sub-groups within groups)
    - Agglomerative approach builds clusters bottom-up

MODEL DETAILS:
    Algorithm: Agglomerative Hierarchical Clustering
    Linkage Methods: Ward (minimizes within-cluster variance)
    Distance Metric: Euclidean
    Optimal Clusters: Determined via dendrogram + silhouette analysis
    Feature Scaling: StandardScaler (required for distance-based methods)

CLUSTERING FEATURES:
    Monthly_Salary_PHP, Performance_Score, Tenure_Years, Age,
    Training_Hours_YTD, Absences_YTD, Overtime_Hours_Monthly,
    Job_Satisfaction_Score, Work_Life_Balance_Score, Num_Promotions

CLUSTER INTERPRETATION:
    Each cluster is profiled by its centroid values to derive HR-actionable
    segment names (e.g., "High Performers at Risk", "Loyal Mid-Tier", etc.)
================================================================================
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist


def get_clustering_features():
    """
    Define features used for hierarchical clustering.
    
    Uses a focused subset of high-separation features to maximize
    cluster quality (silhouette score). Performance and Satisfaction
    provide the strongest natural segmentation, supplemented by
    Salary and Tenure for career-stage differentiation.
    """
    return [
        'Monthly_Salary_PHP', 'Performance_Score', 
        'Tenure_Years', 'Job_Satisfaction_Score'
    ]


def run_hierarchical_clustering(df_clean, output_dir):
    """
    Execute Hierarchical Clustering for Employee Segmentation.
    
    Parameters:
        df_clean: Preprocessed DataFrame (before encoding)
        output_dir: Directory to save results
    
    Returns:
        dict: Comprehensive results including cluster profiles, metrics, and assignments
    """
    print("╔" + "═" * 68 + "╗")
    print("║" + "  HIERARCHICAL CLUSTERING".center(68) + "║")
    print("║" + "  Employee Segmentation Analysis".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    # ========================================================================
    # 1. PREPARE DATA
    # ========================================================================
    print("\n[1] Preparing clustering features...")
    
    feature_cols = get_clustering_features()
    X = df_clean[feature_cols].fillna(df_clean[feature_cols].median()).copy()
    
    # Sampling for dendrogram computation (full dataset for final clustering)
    # Hierarchical clustering is O(n^3), so sample for linkage visualization
    n_sample = min(2000, len(X))
    np.random.seed(42)
    sample_idx = np.random.choice(len(X), n_sample, replace=False)
    X_sample = X.iloc[sample_idx]
    
    print(f"    Features: {len(feature_cols)}")
    print(f"    Full dataset: {len(X)} samples")
    print(f"    Dendrogram sample: {n_sample} samples")
    
    # ========================================================================
    # 2. SCALE FEATURES
    # ========================================================================
    print("\n[2] Transforming features (QuantileTransformer - Normal)...")
    
    # QuantileTransformer makes the data normal, which improves distance metrics
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)
    X_scaled = scaler.fit_transform(X)
    X_sample_scaled = scaler.transform(X_sample)
    
    print(f"    Features transformed: {len(feature_cols)}")
    
    # ========================================================================
    # 3. COMPUTE LINKAGE & FIND BEST METHOD
    # ========================================================================
    print("\n[3] Testing linkage methods (Ward, Complete, Average)...")
    
    linkage_methods = ['ward', 'complete', 'average']
    best_sil = -1
    best_method = 'ward'
    best_k = 2
    
    silhouette_scores = {}
    ch_scores = {}
    db_scores = {}
    
    for method in linkage_methods:
        linkage_matrix = linkage(X_sample_scaled, method=method, metric='euclidean')
        
        for k in range(2, 8):
            labels = fcluster(linkage_matrix, k, criterion='maxclust')
            
            # Reject degenerate clusters (any cluster < 5% of data)
            min_cluster_pct = min(np.bincount(labels)[1:]) / len(labels)
            if min_cluster_pct < 0.05:
                print(f"    {method:>8s} k={k}: SKIPPED (smallest cluster = {min_cluster_pct:.1%})")
                continue
            
            sil = silhouette_score(X_sample_scaled, labels)
            ch = calinski_harabasz_score(X_sample_scaled, labels)
            db = davies_bouldin_score(X_sample_scaled, labels)
            
            if sil > best_sil:
                best_sil = sil
                best_method = method
                best_k = k
            
            print(f"    {method:>8s} k={k}: Silhouette={sil:.4f}, CH={ch:.1f}, DB={db:.4f}")
        print()
    
    # Recompute scores for the best method
    linkage_matrix = linkage(X_sample_scaled, method=best_method, metric='euclidean')
    for k in range(2, 8):
        labels = fcluster(linkage_matrix, k, criterion='maxclust')
        sil = silhouette_score(X_sample_scaled, labels)
        ch = calinski_harabasz_score(X_sample_scaled, labels)
        db = davies_bouldin_score(X_sample_scaled, labels)
        silhouette_scores[k] = float(sil)
        ch_scores[k] = float(ch)
        db_scores[k] = float(db)
    
    optimal_k = best_k
    print(f"    → Best: {best_method} linkage with k={best_k} (silhouette={best_sil:.4f})")
    
    # ========================================================================
    # 4. ASSIGN CLUSTERS TO FULL DATASET
    # ========================================================================
    print(f"\n[4] Assigning {optimal_k} clusters to full dataset...")
    
    # Use full-dataset linkage with the best method for final assignment
    full_linkage = linkage(X_scaled, method=best_method, metric='euclidean')
    cluster_labels = fcluster(full_linkage, optimal_k, criterion='maxclust')
    df_clustered = df_clean.copy()
    df_clustered['Cluster'] = cluster_labels
    
    for c in sorted(df_clustered['Cluster'].unique()):
        count = (df_clustered['Cluster'] == c).sum()
        pct = count / len(df_clustered) * 100
        print(f"    Cluster {c}: {count} employees ({pct:.1f}%)")
    
    # ========================================================================
    # 5. CLUSTER PROFILING
    # ========================================================================
    print("\n[5] Cluster Profiling:")
    
    profiles = {}
    # Profile using all available HR metrics, not just clustering features
    profile_cols = ['Monthly_Salary_PHP', 'Performance_Score', 'Tenure_Years', 'Age',
                    'Training_Hours_YTD', 'Absences_YTD', 'Overtime_Hours_Monthly',
                    'Job_Satisfaction_Score', 'Work_Life_Balance_Score', 'Num_Promotions']
    available_profile_cols = [c for c in profile_cols if c in df_clustered.columns]
    profile_data = df_clustered.groupby('Cluster')[available_profile_cols].mean()
    
    # Also compute attrition rate per cluster
    attrition_by_cluster = df_clustered.groupby('Cluster')['Attrition'].mean()
    
    print(f"\n    {'Cluster':<10} {'Salary':<12} {'Perf':<8} {'Tenure':<8} {'Satis':<8} {'Attrition':<10}")
    print(f"    {'─' * 56}")
    
    for cluster_id in sorted(profile_data.index):
        row = profile_data.loc[cluster_id]
        attrition_rate = attrition_by_cluster.loc[cluster_id]
        
        profile = {
            'cluster_id': int(cluster_id),
            'size': int((df_clustered['Cluster'] == cluster_id).sum()),
            'avg_salary': float(row.get('Monthly_Salary_PHP', 0)),
            'avg_performance': float(row.get('Performance_Score', 0)),
            'avg_tenure': float(row.get('Tenure_Years', 0)),
            'avg_age': float(row.get('Age', 0)),
            'avg_training': float(row.get('Training_Hours_YTD', 0)),
            'avg_absences': float(row.get('Absences_YTD', 0)),
            'avg_overtime': float(row.get('Overtime_Hours_Monthly', 0)),
            'avg_satisfaction': float(row.get('Job_Satisfaction_Score', 0)),
            'avg_wlb': float(row.get('Work_Life_Balance_Score', 0)),
            'avg_promotions': float(row.get('Num_Promotions', 0)),
            'attrition_rate': float(attrition_rate),
        }
        
        # Auto-generate segment label based on profile
        profile['segment_label'] = _generate_segment_label(profile)
        profiles[int(cluster_id)] = profile
        
        print(f"    {cluster_id:<10} ₱{row.get('Monthly_Salary_PHP', 0):>9,.0f} "
              f"{row.get('Performance_Score', 0):>6.2f} "
              f"{row.get('Tenure_Years', 0):>6.1f} "
              f"{row.get('Job_Satisfaction_Score', 0):>6.2f} "
              f"{attrition_rate:>8.1%}")
    
    # ========================================================================
    # 7. CLUSTER LABELS AND INSIGHTS
    # ========================================================================
    print("\n[6] Segment Labels:")
    for cid, prof in profiles.items():
        print(f"    Cluster {cid}: {prof['segment_label']}")
        print(f"      → Size: {prof['size']}, Attrition: {prof['attrition_rate']:.1%}")
    
    # ========================================================================
    # 8. CATEGORICAL DISTRIBUTION PER CLUSTER
    # ========================================================================
    print("\n[7] Categorical distributions per cluster:")
    cat_distributions = {}
    for col in ['Department', 'Gender', 'Education_Level', 'Employment_Type']:
        if col in df_clustered.columns:
            dist = df_clustered.groupby('Cluster')[col].value_counts(normalize=True)
            cat_distributions[col] = {}
            for cluster_id in sorted(df_clustered['Cluster'].unique()):
                if cluster_id in dist.index.get_level_values(0):
                    cat_distributions[col][int(cluster_id)] = dist.loc[cluster_id].head(3).to_dict()
    
    # ========================================================================
    # 9. SAVE RESULTS
    # ========================================================================
    results = {
        'model': 'Hierarchical Clustering (Agglomerative)',
        'linkage_method': best_method.capitalize(),
        'distance_metric': 'Euclidean',
        'optimal_clusters': optimal_k,
        'silhouette_scores': silhouette_scores,
        'calinski_harabasz_scores': ch_scores,
        'davies_bouldin_scores': db_scores,
        'best_silhouette': float(silhouette_scores[optimal_k]),
        'cluster_profiles': profiles,
        'feature_columns': feature_cols,
        'categorical_distributions': cat_distributions,
        'cluster_assignments': cluster_labels.tolist(),
    }
    
    output_path = os.path.join(output_dir, 'hierarchical_clustering_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save clustered dataset
    df_clustered.to_csv(os.path.join(output_dir, 'clustered_dataset.csv'), index=False)
    
    print(f"\n    Results saved to: {output_path}")
    print("\n✓ Hierarchical Clustering complete!")
    
    return results


def _generate_segment_label(profile):
    """Generate a descriptive segment label based on cluster profile."""
    salary = profile['avg_salary']
    perf = profile['avg_performance']
    tenure = profile['avg_tenure']
    satisfaction = profile['avg_satisfaction']
    attrition = profile['attrition_rate']
    
    labels = []
    
    if perf >= 4.0 and salary >= 55000:
        labels.append("High-Value Star Performers")
    elif perf >= 3.5 and attrition > 0.5:
        labels.append("High Performers at Risk")
    elif tenure >= 20 and perf >= 3.0:
        labels.append("Loyal Veterans")
    elif tenure < 10 and perf >= 3.5:
        labels.append("Rising Talent")
    elif perf < 2.5 and attrition > 0.6:
        labels.append("Disengaged & Departing")
    elif satisfaction >= 5.0 and perf >= 3.0:
        labels.append("Engaged Mid-Tier")
    elif salary < 35000 and tenure < 10:
        labels.append("Entry-Level Workers")
    else:
        labels.append("Core Workforce")
    
    return labels[0]


if __name__ == '__main__':
    from data_preprocessing import run_full_preprocessing
    df_clean, df_encoded, df_scaled, encoders = run_full_preprocessing()
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    results = run_hierarchical_clustering(df_clean, output_dir)
