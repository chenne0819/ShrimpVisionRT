# Shrimp Size Estimation - LOOCV Statistical Analysis

This repository contains scripts, datasets, and results for **shrimp length and width estimation** using image-based measurements and **Leave-One-Out Cross-Validation (LOOCV)**. It provides tools for training, testing, and statistically analyzing predicted shrimp sizes against manually measured ground truth.

---

## Overview

The workflow of this project consists of the following steps:

1. Prepare your dataset Excel files with required columns.  
   (This project provided **Example Dataset**: `data.xlsx` contains **17 example shrimp measurements** to allow immediate testing of the scripts and LOOCV workflow.)
2. Train LOOCV models for shrimp length or width.
3. Test the trained models to ensure proper operation.
4. Perform statistical analysis and generate plots to evaluate predictive performance.

The main goals are:

- Predict shrimp **length** and **width** from images.
- Compare predicted measurements with manual measurements.
- Perform statistical analyses and generate visualizations for evaluation.

---

## Folder Structure
<pre>
Shrimp_Size_Estimation/
    ├ LOOCV_Length_train.py
    ├ LOOCV_Length_test.py
    ├ LOOCV_Width_train.py
    ├ LOOCV_Width_test.py
    ├ Statistical Analysis for Shrimp Length Comparison.py
    ├ Statistical Analysis for Shrimp Width Comparison.py
    ├ data.xlsx
    ├ README.md
    └ output/          # Generated plots and analysis results
</pre>

---

## Requirements

- Python 3.8+
- Required packages:

    ```bash
    pip install pandas numpy scipy matplotlib seaborn openpyxl
    ```
- Recommended environment: create a dedicated Python virtual environment.

---

## Setup

1. Open a terminal or command prompt.

2. Navigate to the project root folder:
    ```bash
    cd Shrimp_Size_Estimation
    ```

## Data Preparation
1. Prepare Your Dataset: Create your dataset Excel file(**Shrimp_Size_Estimation/data.xlsx**) with the following columns:
    <table>
      <tr>
        <th>Column Name</th>
        <th>Description</th>
      </tr>
      <tr>
        <td><code>Image Estimated Length (mm)</code></td>
        <td>Length of shrimp estimated from images, calculated as the distance between annotated shrimp points in the video frames.</td>
      </tr>
      <tr>
        <td><code>Actual Length (mm)</code></td>
        <td>Manually measured shrimp length (ground truth).</td>
      </tr>
      <tr>
        <td><code>Image Estimated Width (mm)</code></td>
        <td>Width of shrimp estimated from images, calculated as the distance between annotated shrimp points in the video frames.</td>
      </tr>
      <tr>
        <td><code>Actual Width (mm)</code></td>
        <td>Manually measured shrimp width (ground truth).</td>
      </tr>
      <tr>
        <td><code>Actual Weight (g)</code></td>
        <td>Manually measured shrimp weight (ground truth).</td>
      </tr>
    </table>
---

## Usage
1. Train LOOCV Model (Length)
    ```bash
    python LOOCV_Length_train.py
    ```
   Outputs:
   <pre>
   LOOCV_Length_predict/
   │
   ├─ train_leave_one_out/          # Prediction results for each LOOCV run
   │   ├─ predicted_results_run_1.xlsx
   │   ├─ predicted_results_run_2.xlsx
   │   └─ ...
   │
   ├─ plots/                        # Plots for each LOOCV run and final model
   │   ├─ combined_plot_run_1.png
   │   ├─ combined_plot_run_2.png
   │   ├─ ...
   │   ├─ final_model_regression_plot.png
   │   └─ final_model_error_plot.png
   │
   ├─ summary/                      # Summary Excel files
   │   ├─ metrics_XX_runs.xlsx       # Metrics of all LOOCV runs
   │   └─ all_loocv_predictions.xlsx # Combined predictions of all runs
   │
   └─ final_model/                   # Final model trained on all data
       ├─ final_linear_model_length.pkl
       └─ final_model_metrics_length.xlsx
   </pre>  
  

2. Test LOOCV Model (Length)

   ```bash
   python LOOCV_Length_test.py
   ```

   - Checks if the LOOCV model runs correctly.
   
   - Generates output folders (LOOCV_Length_Prediction\test_results\) but contents are placeholder; adjust saving paths if needed.
  

3. Statistical Analysis for Shrimp Length
   ```bash
   python "Statistical Analysis for Shrimp Length Comparison.py"
   ```
   Outputs:
   <pre>
   LOOCV_Length_statistical_analysis_results/
   │
   ├─ 1_bland_altman_plot.png
   ├─ 2_actual_vs_predicted.png
   ├─ 3_residual_distribution.png
   ├─ 4_qq_plot.png
   ├─ 5_distribution_comparison.png
   └─ 6_absolute_error_distribution.png
   </pre>
   - Computes paired t-test, Wilcoxon test, Bland-Altman analysis, descriptive statistics, and normality test.

   - Generates all plots to visualize differences between actual and predicted lengths.
  

4. Width Analysis

   The workflow for shrimp width is identical to length:
   
   1. Train LOOCV model: python LOOCV_Width_train.py
   
   2. Test LOOCV model: python LOOCV_Width_test.py
   
   3. Statistical analysis: python "Statistical Analysis for Shrimp Width Comparison.py"  
   
   Outputs follow the same folder structure, but for width:
---

## Workflow Overview

<pre>
data.xlsx
   │
   ├─> LOOCV_Length_train.py / LOOCV_Width_train.py
   │       ├─ train_leave_one_out/      # per-run predictions
   │       ├─ plots/                    # per-run and final plots
   │       ├─ summary/                  # metrics and all predictions
   │       └─ final_model/              # trained linear model
   │
   ├─> LOOCV_Length_test.py / LOOCV_Width_test.py
   │       └─ LOOCV_Length_Prediction\test_results / LOOCV_Width_Prediction\test_results #checks model execution
   │
   └─> Statistical Analysis for Shrimp Length/Width Comparison.py
           └─ LOOCV_Length_statistical_analysis_results / LOOCV_Width_statistical_analysis_results
                ├─ Bland-Altman
                ├─ Actual vs Predicted
                ├─ Residuals
                ├─ Q-Q Plot
                ├─ Box Plot
                └─ Absolute Error Distribution
</pre>