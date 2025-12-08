import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import joblib
import pandas as pd
import numpy as np
import os

# Set English font
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Define output directories
output_dir = r".\shrimp weight regression with length only\manual_length_weight_prediction_test"
plots_dir = os.path.join(output_dir, "plots")
logs_dir = os.path.join(output_dir, "logs")

# Create directories if they do not exist
for d in [output_dir, plots_dir, logs_dir]:
    os.makedirs(d, exist_ok=True)

# Load the trained model
model_path = r".\shrimp weight regression with length only\manual_length_weight_prediction\models\polynomial_regression_model_degree3.pkl"
model = joblib.load(model_path)
print(f"Loaded model: {model_path}")

# Load LOOCV predicted length data
LOOCV_path = r"Shrimp_size_weight_analysis\Shrimp_Size_Estimation\LOOCV_Length_predict\summary\all_loocv_predictions.xlsx"
LOOCV = pd.read_excel(LOOCV_path)

# X_test_LOOCV: using LOOCV predicted length
X_test_loocv = LOOCV['Predicted Length (mm)'].to_numpy().reshape(-1, 1)
# X_test_actual: using actual measured length
X_test_actual = LOOCV['Actual Length (mm)'].to_numpy().reshape(-1, 1)

# Load actual weight
Weight_path = r"Shrimp_size_weight_analysis\Shrimp_Size_Estimation\data.xlsx"
Weight_df = pd.read_excel(Weight_path)
y_test = Weight_df['Actual Weight (g)'].to_numpy()

# Predict weight using the model
y_pred_loocv = model.predict(X_test_loocv)
y_pred_actual = model.predict(X_test_actual)

# Function to compute evaluation metrics
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    me = np.mean(y_true - y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    return mse, rmse, mae, mape, me, r2, corr

# Compute metrics for both predicted lengths and actual lengths
metrics_loocv = compute_metrics(y_test, y_pred_loocv)
metrics_actual = compute_metrics(y_test, y_pred_actual)

# Save evaluation metrics to Excel
metrics_df = pd.DataFrame({
    'MSE (g²)': [metrics_loocv[0], metrics_actual[0]],
    'RMSE (g)': [metrics_loocv[1], metrics_actual[1]],
    'MAE (g)': [metrics_loocv[2], metrics_actual[2]],
    'MAPE (%)': [metrics_loocv[3], metrics_actual[3]],
    'ME (g)': [metrics_loocv[4], metrics_actual[4]],
    'R²': [metrics_loocv[5], metrics_actual[5]],
    'PCC': [metrics_loocv[6], metrics_actual[6]]
}, index=['Predicted Length', 'Actual Length'])

excel_path = os.path.join(logs_dir, 'evaluation_comparison.xlsx')
metrics_df.to_excel(excel_path)
print(f"Comparison metrics saved to {excel_path}")

# Save error data
residuals_loocv = y_test - y_pred_loocv
residuals_actual = y_test - y_pred_actual
error_data = pd.DataFrame({
    'Actual Weight (g)': y_test,
    'Predicted Weight (LOOCV Length) (g)': y_pred_loocv,
    'Residual (LOOCV Length) (g)': residuals_loocv,
    'Predicted Weight (Actual Length) (g)': y_pred_actual,
    'Residual (Actual Length) (g)': residuals_actual
})
error_excel_path = os.path.join(logs_dir, 'error_data_comparison.xlsx')
error_data.to_excel(error_excel_path, index=False)
print(f"Error data saved to {error_excel_path}")

# Plot Figures

# Figure 1: LOOCV Length
plt.figure(figsize=(8, 6))
plt.scatter(X_test_loocv, y_test, alpha=0.6)
X_range = np.linspace(X_test_loocv.min(), X_test_loocv.max(), 100).reshape(-1, 1)
y_range_loocv = model.predict(X_range)
plt.plot(X_range, y_range_loocv, 'r-', label="Fitted Curve")
plt.xlabel("Predicted Length (mm)")
plt.ylabel("Actual Weight (g)")
plt.title("LOOCV Predicted Length vs Actual Weight")
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'fitted_curve_loocv.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Actual Length
plt.figure(figsize=(8, 6))
plt.scatter(X_test_actual, y_test, alpha=0.6)
X_range_actual = np.linspace(X_test_actual.min(), X_test_actual.max(), 100).reshape(-1, 1)
y_range_actual = model.predict(X_range_actual)
plt.plot(X_range_actual, y_range_actual, 'b-', label="Fitted Curve")
plt.xlabel("Actual Length (mm)")
plt.ylabel("Actual Weight (g)")
plt.title("Actual Length vs Actual Weight")
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'fitted_curve_actual.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Residual comparison with lines to zero
plt.figure(figsize=(8, 6))
plt.scatter(y_test, residuals_loocv, alpha=0.6, label='Residual (LOOCV Length)')
plt.scatter(y_test, residuals_actual, alpha=0.6, label='Residual (Actual Length)')
for i in range(len(y_test)):
    plt.plot([y_test[i], y_test[i]], [0, residuals_loocv[i]], color='gray', linestyle='--', alpha=0.5)
    plt.plot([y_test[i], y_test[i]], [0, residuals_actual[i]], color='gray', linestyle='--', alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Actual Weight (g)")
plt.ylabel("Residual (g)")
plt.title("Residual Comparison")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'residual_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Plots saved to:", plots_dir)
