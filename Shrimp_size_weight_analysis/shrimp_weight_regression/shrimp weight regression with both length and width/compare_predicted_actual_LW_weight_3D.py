import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import joblib
import os

# Set English font
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Output directories
output_dir = r"./shrimp weight regression with both length and width/shrimp_weight_regression_LW_test_3D"
plots_dir = os.path.join(output_dir, "plots")
summary_dir = os.path.join(output_dir, "summary")
for d in [plots_dir, summary_dir]:
    os.makedirs(d, exist_ok=True)

# Load LOOCV predicted Length/Width
LOOCV_len_path = r"Shrimp_size_weight_analysis\Shrimp_Size_Estimation\LOOCV_Length_predict\summary\all_loocv_predictions.xlsx"
LOOCV_len = pd.read_excel(LOOCV_len_path)
pred_len = LOOCV_len['Predicted Length (mm)'].to_numpy().reshape(-1, 1)
actual_len = LOOCV_len['Actual Length (mm)'].to_numpy().reshape(-1, 1)

LOOCV_wid_path = r"Shrimp_size_weight_analysis\Shrimp_Size_Estimation\LOOCV_Width_predict\summary\all_loocv_predictions.xlsx"
LOOCV_wid = pd.read_excel(LOOCV_wid_path)
pred_wid = LOOCV_wid['Predicted Width (mm)'].to_numpy().reshape(-1, 1)
actual_wid = LOOCV_wid['Actual Width (mm)'].to_numpy().reshape(-1, 1)

# Load actual weight
Weight_path = r"Shrimp_size_weight_analysis\Shrimp_Size_Estimation\data.xlsx"
Weight_df = pd.read_excel(Weight_path)
y_test = Weight_df['Actual Weight (g)'].to_numpy()

# Combine features
X_pred = np.hstack([pred_len, pred_wid])
X_actual = np.hstack([actual_len, actual_wid])

# Load trained multi-feature model
model_path = r".\shrimp weight regression with both length and width\shrimp_weight_regression_LW_3D\models\multi_feature_model.pkl"
model = joblib.load(model_path)
print(f"Loaded model: {model_path}")

# Predict weight
y_pred_loocv = model.predict(X_pred)
y_pred_actual = model.predict(X_actual)

# Compute metrics
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    me = np.mean(y_true - y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    return mse, rmse, mae, mape, me, r2, corr

metrics_loocv = compute_metrics(y_test, y_pred_loocv)
metrics_actual = compute_metrics(y_test, y_pred_actual)

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'MSE (g^2)': [metrics_loocv[0], metrics_actual[0]],
    'RMSE (g)': [metrics_loocv[1], metrics_actual[1]],
    'MAE (g)': [metrics_loocv[2], metrics_actual[2]],
    'MAPE (%)': [metrics_loocv[3], metrics_actual[3]],
    'ME (g)': [metrics_loocv[4], metrics_actual[4]],
    'R²': [metrics_loocv[5], metrics_actual[5]],
    'PCC': [metrics_loocv[6], metrics_actual[6]]
}, index=['Predicted LW', 'Actual LW'])

metrics_csv_path = os.path.join(summary_dir, 'evaluation_metrics_comparison.csv')
metrics_df.to_csv(metrics_csv_path)
print(f"Metrics saved to {metrics_csv_path}")

# Save residuals
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

# 3D Regression Plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Plot actual data points
ax.scatter(actual_len.ravel(), actual_wid.ravel(), y_test, alpha=0.6, label='Data')

# Fit curve along diagonal (predicted)
t = np.linspace(0, 1, 100)
x_curve = actual_len.min() + t*(actual_len.max() - actual_len.min())
m_curve = actual_wid.min() + t*(actual_wid.max() - actual_wid.min())
X_curve = np.column_stack([x_curve, m_curve])
y_curve = model.predict(X_curve)

ax.plot(x_curve, m_curve, y_curve, color='red', linewidth=2, label='Fitted Curve')

# Labels
ax.set_xlabel('Actual Length (mm)')
ax.set_ylabel('Actual Width (mm)')
ax.set_zlabel('Actual Weight (g)')
ax.set_title('3D Regression: Length & Width vs Actual Weight')
ax.legend()

plt.savefig(os.path.join(plots_dir, "3d_regression_plot_comparison.png"), dpi=300, bbox_inches='tight')
plt.show()
