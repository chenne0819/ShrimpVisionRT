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
output_dir = r".\shrimp weight regression with both length and width\shrimp_weight_regression_LW_test"
plots_dir = os.path.join(output_dir, "plots")
summary_dir = os.path.join(output_dir, "summary")

# Create directories if they do not exist
for d in [plots_dir, summary_dir]:
    os.makedirs(d, exist_ok=True)

# Load trained multi-feature LW model
model_path = r".\shrimp weight regression with both length and width\shrimp_weight_regression_LW\models\multi_feature_model.pkl"
model = joblib.load(model_path)
print(f"Loaded model: {model_path}")

# Load LOOCV predicted length and width
LOOCV_len_path = r"Shrimp_size_weight_analysis\Shrimp_Size_Estimation\LOOCV_Length_predict\summary\all_loocv_predictions.xlsx"
LOOCV_len = pd.read_excel(LOOCV_len_path)
pred_len = LOOCV_len['Predicted Length (mm)'].to_numpy().reshape(-1, 1)
actual_len = LOOCV_len['Actual Length (mm)'].to_numpy()

LOOCV_wid_path = r"Shrimp_size_weight_analysis\Shrimp_Size_Estimation\LOOCV_Width_predict\summary\all_loocv_predictions.xlsx"
LOOCV_wid = pd.read_excel(LOOCV_wid_path)
pred_wid = LOOCV_wid['Predicted Width (mm)'].to_numpy().reshape(-1, 1)
actual_wid = LOOCV_wid['Actual Width (mm)'].to_numpy()

# Combine predicted and actual LW features
X_pred = np.hstack([pred_len, pred_wid])
X_actual = np.hstack([actual_len.reshape(-1,1), actual_wid.reshape(-1,1)])

# Load actual weight
Weight_path = r"Shrimp_size_weight_analysis\Shrimp_Size_Estimation\data.xlsx"
Weight_df = pd.read_excel(Weight_path)
y_test = Weight_df['Actual Weight (g)'].to_numpy()

# Predict weight
y_pred_loocv = model.predict(X_pred)
y_pred_actual = model.predict(X_actual)

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

# Compute metrics
metrics_loocv = compute_metrics(y_test, y_pred_loocv)
metrics_actual = compute_metrics(y_test, y_pred_actual)

# Save evaluation metrics to CSV in summary folder
metrics_df = pd.DataFrame({
    'MSE (g^2)': [metrics_loocv[0], metrics_actual[0]],
    'RMSE (g)': [metrics_loocv[1], metrics_actual[1]],
    'MAE (g)': [metrics_loocv[2], metrics_actual[2]],
    'MAPE (%)': [metrics_loocv[3], metrics_actual[3]],
    'ME (g)': [metrics_loocv[4], metrics_actual[4]],
    'R^2': [metrics_loocv[5], metrics_actual[5]],
    'PCC': [metrics_loocv[6], metrics_actual[6]]
}, index=['Predicted LW', 'Actual LW'])

metrics_csv_path = os.path.join(summary_dir, 'evaluation_comparison.csv')
metrics_df.to_csv(metrics_csv_path)
print(f"Comparison metrics saved to {metrics_csv_path}")

# Save detailed error data
residuals_loocv = y_test - y_pred_loocv
residuals_actual = y_test - y_pred_actual
error_data = pd.DataFrame({
    'Actual Weight (g)': y_test,
    'Predicted Weight (Predicted LW) (g)': y_pred_loocv,
    'Residual (Predicted LW) (g)': residuals_loocv,
    'Predicted Weight (Actual LW) (g)': y_pred_actual,
    'Residual (Actual LW) (g)': residuals_actual
})
error_csv_path = os.path.join(summary_dir, 'error_data_comparison.csv')
error_data.to_csv(error_csv_path, index=False)
print(f"Error data saved to {error_csv_path}")

# Plot 1: Predicted LW vs Actual Weight
plt.figure(figsize=(8,6))
plt.scatter(X_pred[:,0], y_test, alpha=0.6, label="Length")
plt.scatter(X_pred[:,1], y_test, alpha=0.6, label="Width")
X_range_len = np.linspace(X_pred[:,0].min(), X_pred[:,0].max(), 100).reshape(-1,1)
X_range_wid = np.linspace(X_pred[:,1].min(), X_pred[:,1].max(), 100).reshape(-1,1)
y_range_len = model.predict(np.hstack([X_range_len, X_pred[:,1].mean()*np.ones_like(X_range_len)]))
y_range_wid = model.predict(np.hstack([X_pred[:,0].mean()*np.ones_like(X_range_wid), X_range_wid]))
plt.plot(X_range_len, y_range_len, 'r-', label='Length Fit Curve')
plt.plot(X_range_wid, y_range_wid, 'b--', label='Width Fit Curve')
plt.xlabel("Length / Width (mm)")
plt.ylabel("Actual Weight (g)")
plt.title("Predicted LW vs Actual Weight with Fit Curves")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'predicted_LW_fit_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Actual LW vs Actual Weight
plt.figure(figsize=(8,6))
plt.scatter(X_actual[:,0], y_test, alpha=0.6, label="Length")
plt.scatter(X_actual[:,1], y_test, alpha=0.6, label="Width")
X_range_len = np.linspace(X_actual[:,0].min(), X_actual[:,0].max(), 100).reshape(-1,1)
X_range_wid = np.linspace(X_actual[:,1].min(), X_actual[:,1].max(), 100).reshape(-1,1)
y_range_len = model.predict(np.hstack([X_range_len, X_actual[:,1].mean()*np.ones_like(X_range_len)]))
y_range_wid = model.predict(np.hstack([X_actual[:,0].mean()*np.ones_like(X_range_wid), X_range_wid]))
plt.plot(X_range_len, y_range_len, 'r-', label='Length Fit Curve')
plt.plot(X_range_wid, y_range_wid, 'b--', label='Width Fit Curve')
plt.xlabel("Length / Width (mm)")
plt.ylabel("Actual Weight (g)")
plt.title("Actual LW vs Actual Weight with Fit Curves")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'actual_LW_fit_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Residual comparison with vertical lines to zero
plt.figure(figsize=(8,6))
plt.scatter(y_test, residuals_loocv, alpha=0.6, label='Residual (Predicted LW)')
plt.scatter(y_test, residuals_actual, alpha=0.6, label='Residual (Actual LW)')
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
