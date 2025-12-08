# Shrimp Size & Weight Analysis Framework

This repository provides a complete workflow for analyzing shrimp imagery, designed to estimate actual shrimp dimensions (length/width) and weight using computer vision and statistical regression methods.

> **Disclaimer**:
> This repository is intended to demonstrate our **research methodology** and **experimental code**.
> **Sample data** is provided to verify the functionality of the scripts, but this does not represent the full dataset used in our specific study.

---

## Project Structure & Workflow

The experimental process is divided into three sequential stages. Please follow this order to understand the logic and execute the code:

### 1. Linear Relationship Verification (`linear_relationship`)
**Goal: Feasibility Analysis**
Before attempting complex predictions, we first validate whether the "pixel distance estimated from images" and the "manual ground truth measurement" exhibit a significant **positive linear correlation**.
* This step confirms that the annotated features in the images accurately reflect the physical size of the shrimp.
* Establishing this linearity is the foundation for the subsequent estimation steps.

### 2. Shrimp Size Estimation (`Shrimp_Size_Estimation`)
**Goal: Precise Size Prediction (Pixels → mm)**
Once linearity is confirmed, we use regression models (specifically Leave-One-Out Cross-Validation, LOOCV) to convert image data into physical units (mm).
* The objective is to minimize prediction error, bringing the predicted Length and Width closer to the actual ground truth values.
* The output of this stage (Predicted Length/Width) serves as the input for the weight regression model.

### 3. Weight Regression Analysis (`shrimp_weight_regression`)
**Goal: Weight Prediction (mm → g)**
Using the "Predicted Length" and "Predicted Width" generated in the previous step, we estimate the shrimp's weight.
* **Single vs. Multi-variable Analysis**: We compare methods using only Length versus using both Length + Width.
* Our experiments demonstrate that incorporating **Width** alongside Length typically reduces error and provides a more accurate weight estimation closer to the actual value.

---

##  Usage Guide

To ensure the system works correctly, execute the modules in the following order:

### Step 1: Check Linear Relationship
Navigate to the `linear_relationship` directory to verify data quality and linearity.

### Step 2: Execute Size Estimation
Navigate to the `Shrimp_Size_Estimation` directory to train models and generate predicted length/width data.

### Step 3: Execute Weight Analysis
Navigate to the `shrimp_weight_regression` directory. This step utilizes the prediction results from Step 2 to perform weight regression.

---

##  Requirements & Important Notes

1.  **Environment**:
    * Python 3.8+ is recommended.
    * Install required packages:
        ```bash
        pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl
        ```
2.  **Path Configuration**:
    * This project involves reading files across different directories (e.g., Weight Regression scripts need to load results from Size Estimation).
    * **Please check the file paths in the code before running.** If you rename folders or change the project structure, you must update the paths in the Python scripts to match your local environment.

3.  **Data Format**:
    * If you wish to use your own dataset, please refer to the `README.md` in each subfolder for specific Excel column requirements.