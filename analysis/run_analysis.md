# Analysis Orchestrator Documentation

## Overview
The `run_analysis.py` file is the central nervous system of the project. It manages the execution sequence of all analytical modules and ensures that data flows correctly between them.

## Workflow Execution Sequence

1.  **Environment Setup**: Configures global parameters and initializes the output directory.
2.  **Data Preprocessing**: Executes `data_preprocessing.py` to generate the `cleaned_dataset.csv`.
3.  **Predictive Modeling**:
    - Runs **Decision Tree** for attrition classification.
    - Runs **Hierarchical Clustering** for segmentation.
    - Runs **Regularized Regression** for salary prediction.
4.  **Performance Logging**: Captures execution time and success/failure status for each phase.
5.  **Final Summary**: Generates a consolidated ASCII report of the key metrics achieved across all models.

## Usage
To execute the entire pipeline, run:
```bash
python analysis/run_analysis.py
```

## Error Handling
The orchestrator uses a phase-based execution model. If one model fails, the script will log the error but attempt to continue with the remaining models, ensuring partial results are still available for the dashboard.

## Output
This script triggers the creation of all files in the `analysis/output/` directory.
