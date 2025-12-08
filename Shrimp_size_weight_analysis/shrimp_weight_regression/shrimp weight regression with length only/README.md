# Shrimp Weight Prediction Analysis

This repository contains tools for **modeling and predicting shrimp weight** based on **length**. It implements a complete workflow from training a **Polynomial Regression Model (Degree 3)** to validating the model using both manually measured lengths and LOOCV-predicted lengths from computer vision tasks.

The project aims to quantify the weight prediction accuracy and analyze the error differences between using "Perfect" (manual) input vs. "Estimated" (LOOCV) input.  

In this project, **Example dataset** with 42 set of data provided in **'shrimp_len_wid_wei.xlsx'**.

---

## Overview

The workflow consists of two main stages:

1.  **Model Training (`shrimp_length_weight_regression.py.py`)**
    * Trains a 3rd-degree polynomial regression model using length and weight data.
    * Performs 5-fold Cross-Validation (KFold).
    * Generates the regression formula (Text/LaTeX).
    * Saves the trained model (`.pkl`) and evaluation metrics.

2.  **Model Testing & Comparison (`compare_loocv_actual_length.py`)**
    * Loads the pre-trained model.
    * Predicts weight using **LOOCV Predicted Length** (from image analysis).
    * Predicts weight using **Actual Measured Length** (ground truth).
    * Compares the results (MSE, RMSE, R², etc.) and generates comparison plots.

---

## Folder Structure

Based on the scripts, the project generates and utilizes the following directory structure:

<pre>
Root/
│
├─ shrimp_weight_regression/
│   ├─ shrimp_len_wid_wei.xlsx                   # <b>[Input]</b> Dataset for Training
│   └─ shrimp weight regression with length only/
│        ├─ manual_length_weight_prediction/     # <b>[Output]</b> Training Results
│        │    ├─ models/
│        │    │    └─ polynomial_regression_model_degree3.pkl
│        │    ├─ formulas/
│        │    │    ├─ regression_formula.tex
│        │    │    └─ regression_formula.txt
│        │    ├─ logs/
│        │    │    ├─ metrics_output.csv
│        │    │    └─ metrics_output.txt
│        │    └─ plots/                          # Training visualizations
│        │
│        └─ manual_length_weight_prediction_test/# <b>[Output]</b> Comparison/Testing Results
│             ├─ logs/
│             │    ├─ evaluation_comparison.xlsx # Metrics comparison (LOOCV vs Actual)
│             │    └─ error_data_comparison.xlsx # Detailed residual data
│             └─ plots/                          # Comparison visualizations
│    
└─ Shrimp_Size_Estimation/                   # <b>[Input]</b> External validation data
     ├─ LOOCV_Length_predict/summary/
     │    └─ all_loocv_predictions.xlsx      # Contains 'Predicted Length (mm)'
     └─ data.xlsx                            # Contains 'Actual Length (mm)' & 'Actual Weight (g)'
</pre>

---
## Setup

1. Open a terminal or command prompt.

2. Navigate to the folder:
    ```bash
    cd shrimp_weight_regression
    ```

---

## Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
    pip install pandas numpy scipy matplotlib openpyxl joblib scikit-learn
```
---

## Usage

### 1. Train the Model

Run the training script to build the regression model from `shrimp_len_wid_wei.xlsx`.

```bash
   python .\shrimp weight regression with length only\shrimp_length_weight_regression.py
```
- Input: shrimp_len_wid_wei.xlsx
- Output: 
  - Trained Model: polynomial_regression_model_degree3.pkl
  - Formula: Regression formula generated in both LaTeX (.tex) and Text (.txt) formats. 
  - Metrics Logs: metrics_output.csv containing MSE, RMSE, R², etc.
  - Generated Plots (manual_length_weight_prediction/plots/)
    - length_vs_actual_weight.png: Visualizes the training data with the fitted regression curve.
    - predicted_vs_actual_weight.png: Compares predicted weight against ground truth.
    - residual_plot.png: Shows the distribution of errors (residuals). 
    - boxplot_length_weight.png: Displays data distribution for length and weight.

### 2.Compare Predictions (Test)
Run the comparison script to validate the model using both predicted (LOOCV) and actual lengths.
```bash
    python .\shrimp weight regression with length only\compare_loocv_actual_length.py
```  
- Inputs:
  - Model: Loaded from manual_length_weight_prediction/models/
  - LOOCV Data: Shrimp_Size_Estimation/.../all_loocv_predictions.xlsx
  - Actual Data: Shrimp_Size_Estimation/data.xlsx
- Output:
  - Comparison Report: evaluation_comparison.xlsx (Metrics) and error_data_comparison.xlsx (Detailed residuals).
  - Generated Plots (manual_length_weight_prediction_test/plots/):
    - fitted_curve_loocv.png: Regression curve using LOOCV estimated lengths. 
    - fitted_curve_actual.png: Regression curve using Actual measured lengths. 
    - residual_comparison.png: Side-by-side comparison of residuals (errors) for both methods to visualize estimation stability.
---
## Metrics Explained
The comparison script generates the following metrics in the logs:
<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MSE (g²)</strong></td>
      <td>Mean Squared Error - Average squared difference between estimated and actual weight.</td>
    </tr>
    <tr>
      <td><strong>RMSE (g)</strong></td>
      <td>Root Mean Squared Error - Standard deviation of the prediction errors.</td>
    </tr>
    <tr>
      <td><strong>MAE (g)</strong></td>
      <td>Mean Absolute Error - Average absolute difference.</td>
    </tr>
    <tr>
      <td><strong>MAPE (%)</strong></td>
      <td>Mean Absolute Percentage Error - Average percentage deviation.</td>
    </tr>
    <tr>
      <td><strong>R²</strong></td>
      <td>R-squared - Proportion of variance explained by the model.</td>
    </tr>
  </tbody>
</table>
---

## Further Statistical Analysis
If you wish to perform additional statistical hypothesis testing or comparison between the image-based estimation (weight) and manual measurement (weight) methods, please refer to the resources located in the **Shrimp_Size_Estimation/** directory:

- **README.md**: Provides an overview of the size estimation project and data.  

- **Statistical Analysis for Shrimp Length Comparison.py**: A dedicated script for performing advanced statistical tests (e.g., t-tests, Shapiro-Wilk test, etc.) to analyze the significance of differences between methods.