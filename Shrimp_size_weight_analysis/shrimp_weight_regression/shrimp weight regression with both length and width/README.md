# Shrimp Weight Regression Analysis - Length & Width

This repository contains scripts for **estimating shrimp weight** based on length and width measurements. It utilizes **Polynomial Regression (Degree 3)** to model the relationship between size dimensions and weight. The project includes workflows for both **2D** and **3D** analysis, and verifies the model using previously predicted size data (LOOCV results).

---

## Overview

The workflow of this project consists of the following steps:

1.  **Prepare Datasets**: Ensure training data (`shrimp_len_wid_wei.xlsx`) and previous LOOCV size prediction results are available.
2.  **Train Regression Models**: Train polynomial regression models using actual length/width and weight.
3.  **Compare Predictions**: Validate the model by comparing weight predictions derived from **Image Estimated Length/Width** vs. **Actual Length/Width**.
4.  **3D Visualization**: Perform training and comparison with 3D visualization to better understand the multi-variable relationship.

The main goals are:
- Develop a formula to convert Shrimp Length (L) and Width (W) into Weight (g).
- Evaluate how error in size estimation (from image processing) affects weight calculation.
- Generate LaTeX formulas and 3D plots for reporting.

---

## Folder Structure
<pre>
shrimp_weight_regression/
    ├ shrimp_len_wid_wei.xlsx  (Training Data)
    └ shrimp weight regression with both length and width/
        ├ train_shrimp_weight_regression_LW.py
        ├ compare_predicted_actual_LW_weight.py
        ├ train_3D_shrimp_weight_regression_LW.py
        ├ compare_predicted_actual_LW_weight_3D.py
        ├ README.md
        └ (Generated Output)
            ├ shrimp_weight_regression_LW/
            ├ shrimp_weight_regression_LW_test/
            ├ shrimp_weight_regression_LW_3D/
            └ shrimp_weight_regression_LW_test_3D/
</pre>

---

## Requirements

- Python 3.8+
- Required packages:
    ```bash
    pip install pandas numpy scipy matplotlib scikit-learn openpyxl joblib
    ```

---

## Setup

1. Open a terminal or command prompt.

2. Navigate to the project root folder:
    ```bash
    cd shrimp_weight_regression
    ```

## Data Preparation
1. **Training Data**: Ensure `shrimp_len_wid_wei.xlsx` is in the root directory with the following columns:
    <table>
      <tr>
        <th>Column Name</th>
        <th>Description</th>
      </tr>
      <tr>
        <td><code>length (mm)</code></td>
        <td>Actual measured length of the shrimp.</td>
      </tr>
      <tr>
        <td><code>width (mm)</code></td>
        <td>Actual measured width of the shrimp.</td>
      </tr>
      <tr>
        <td><code>weight (g)</code></td>
        <td>Actual measured weight of the shrimp.</td>
      </tr>
    </table>

2. **LOOCV Prediction Data (Dependencies)**:
   The comparison scripts require output from the previous Size Estimation project. Ensure the directory structure contains:
   - `Shrimp_size_weight_analysis\Shrimp_Size_Estimation\LOOCV_Length_predict\summary\all_loocv_predictions.xlsx`
   - `Shrimp_size_weight_analysis\Shrimp_Size_Estimation\LOOCV_Width_predict\summary\all_loocv_predictions.xlsx`
   - `Shrimp_size_weight_analysis\Shrimp_Size_Estimation\data.xlsx`

---

## Usage

1. Train Standard Regression Model (2D)
    ```bash
    python train_shrimp_weight_regression_LW.py
    ```
   **Outputs:** `shrimp_weight_regression_LW/`
   - **models/**: `multi_feature_model.pkl` (L+W), `length_model.pkl`, `width_model.pkl`.
   - **formulas/**: `.tex` files containing the generated mathematical formulas.
   - **plots/**: Regression curves, residual plots, and boxplots.
   - **summary/**: `output_metrics.csv` containing MSE, RMSE, MAE, R², etc.

2. Compare Predictions (Standard)
    ```bash
    python compare_predicted_actual_LW_weight.py
    ```
   **Outputs:** `shrimp_weight_regression_LW_test/`
   - **plots/**: Comparison of fit curves and residuals between "Predicted LW" weight and "Actual LW" weight.
   - **summary/**: `evaluation_comparison.csv` comparing error metrics between the two approaches.

3. Train 3D Regression Model
    ```bash
    python train_3D_shrimp_weight_regression_LW.py
    ```
   **Outputs:** `shrimp_weight_regression_LW_3D/`
   - **plots/**: `3d_length_width_weight_curve.png` (3D plot of the regression plane).
   - **models/**: 3D-specific model files.

4. Compare Predictions in 3D
    ```bash
    python compare_predicted_actual_LW_weight_3D.py
    ```
   **Outputs:** `shrimp_weight_regression_LW_test_3D/`
   - **plots/**: `3d_regression_plot_comparison.png` showing data points relative to the weight surface.
   - **summary/**: Metrics and error data comparisons specific to the 3D context.

---

## Workflow Overview

<pre>
shrimp_len_wid_wei.xlsx
   │
   ├─> 1. train_shrimp_weight_regression_LW.py
   │       ├─ models/ (Saves .pkl models)
   │       └─ formulas/ (Generates LaTeX)
   │
   ├─> 2. compare_predicted_actual_LW_weight.py
   │       (Uses models from Step 1 + LOOCV Data)
   │       └─ summary/ (Compares Predicted-LW inputs vs Actual-LW inputs)
   │
   ├─> 3. train_3D_shrimp_weight_regression_LW.py
   │       └─ plots/ (Generates 3D visualization of the regression plane)
   │
   └─> 4. compare_predicted_actual_LW_weight_3D.py
           (Uses models from Step 3 + LOOCV Data)
           └─ plots/ (3D Scatter comparison of prediction accuracy)
</pre>