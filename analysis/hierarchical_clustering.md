# Hierarchical Clustering Documentation

## Overview
The `hierarchical_clustering.py` module performs unsupervised learning to segment the workforce into distinct "personas" based on behavioral and career metrics.

## Technical Implementation

### 1. Feature Transformation
To achieve high cluster quality, the module uses a **QuantileTransformer** with a normal output distribution. This effectively handles the non-Gaussian distributions of features like Salary and Tenure, ensuring that the Euclidean distance metric used in clustering is not dominated by skewed ranges.

### 2. Multi-Method Linkage Search
The engine dynamically tests three linkage methods to find the one with the highest **Silhouette Score**:
- **Ward**: Minimizes the variance within clusters (Best for balanced segments).
- **Complete**: Minimizes the maximum distance between points.
- **Average**: Uses the average distance between all points.

### 3. Constraint-Based Optimization
To prevent "degenerate" clusters (e.g., a cluster with only 1 employee), the model implements a **minimum cluster size constraint (5%)**. Any solution where a cluster contains less than 5% of the total population is automatically rejected in favor of the next best cohesive split.

## Resulting Segments
The current optimization identifies **4 optimal clusters** (Silhouette Score: **0.5287**):
- **Core Workforce**: Standard performance and tenure.
- **Loyal Veterans**: High tenure, high salary, low attrition risk.
- **High Performers at Risk**: High performance but high attrition/risk indicators.
- **Stagnated Employees**: High tenure but low promotion/salary growth.

## Output
- `analysis/output/hierarchical_clustering_results.json`: Cluster profiles, metrics, and individual assignments.
