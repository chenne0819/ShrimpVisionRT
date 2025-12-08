import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import joblib
import pandas as pd
import numpy as np
import os

# Basic matplotlib settings
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Output directory structure
output_dir = ".\\shrimp weight regression with length only\\manual_length_weight_prediction"
plot_dir = os.path.join(output_dir, "plots")
model_dir = os.path.join(output_dir, "models")
formula_dir = os.path.join(output_dir, "formulas")
log_dir = os.path.join(output_dir, "logs")

for d in [output_dir, plot_dir, model_dir, formula_dir, log_dir]:
    os.makedirs(d, exist_ok=True)

# Load dataset
data_path = r".\\shrimp_len_wid_wei.xlsx"
df = pd.read_excel(data_path)

X = df['length (mm)'].to_numpy().reshape(-1, 1)
y = df['weight (g)'].to_numpy()

length_min, length_max = X.min(), X.max()
weight_min, weight_max = y.min(), y.max()

# Polynomial regression model
degree = 3
model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())

cv = 5
kf = KFold(n_splits=cv, shuffle=True, random_state=42)

# Compute MSE and R² via CV
mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

mse_mean = -np.mean(mse_scores)
mse_std = np.std(mse_scores)
r2_mean = np.mean(r2_scores)
r2_std = np.std(r2_scores)

rmse_mean = np.sqrt(mse_mean)
rmse_std = mse_std / (2 * np.sqrt(mse_mean)) if mse_mean > 0 else 0

# Manually compute MAE / MAPE / ME
mae_scores = []
mape_scores = []
me_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae_scores.append(np.mean(np.abs(y_test - y_pred)))
    mape_scores.append(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
    me_scores.append(np.mean(y_test - y_pred))

mae_mean = np.mean(mae_scores)
mae_std = np.std(mae_scores)
mape_mean = np.mean(mape_scores)
mape_std = np.std(mape_scores)
me_mean = np.mean(me_scores)
me_std = np.std(me_scores)

# Cross-validated prediction
predicted = cross_val_predict(model, X, y, cv=cv)
corr, _ = pearsonr(y, predicted)

# Train final model & save
model.fit(X, y)
model_path = os.path.join(model_dir, 'polynomial_regression_model_degree3.pkl')
joblib.dump(model, model_path)

# Build metrics report
metrics_report = []
metrics_report.append("==== Polynomial Regression for Shrimp Weight (Length Only) ====\n")
metrics_report.append(f"Data file: {data_path}\n")
metrics_report.append(f"Number of samples: {len(y)}")
metrics_report.append(f"Feature: Length (mm)")
metrics_report.append(f"Target: Weight (g)\n")
metrics_report.append(f"Length range: {length_min:.2f} mm to {length_max:.2f} mm")
metrics_report.append(f"Weight range: {weight_min:.2f} g to {weight_max:.2f} g\n")
metrics_report.append(f"Polynomial degree: {degree}")
metrics_report.append(f"Cross-validation folds: {cv}\n")
metrics_report.append(f"Mean Squared Error (MSE): {mse_mean:.4f} (±{mse_std:.4f}) g²")
metrics_report.append(f"Root Mean Squared Error (RMSE): {rmse_mean:.4f} (±{rmse_std:.4f}) g")
metrics_report.append(f"Mean Absolute Error (MAE): {mae_mean:.4f} (±{mae_std:.4f}) g")
metrics_report.append(f"Mean Absolute Percentage Error (MAPE): {mape_mean:.4f} (±{mape_std:.4f}) %")
metrics_report.append(f"Mean Error (ME): {me_mean:.4f} (±{me_std:.4f}) g")
metrics_report.append(f"Mean R² score: {r2_mean:.4f} (±{r2_std:.4f})")
metrics_report.append(f"Pearson correlation coefficient: {corr:.4f}\n")
metrics_report.append(f"Saved model path: {model_path}\n")

print("\n".join(metrics_report))

log_path = os.path.join(log_dir, "metrics_output.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(metrics_report))

print(f"\nMetrics report saved to: {log_path}")

# Save metrics as CSV
csv_path = os.path.join(log_dir, "metrics_output.csv")

metrics_dict = {
    "num_samples": [len(y)],
    "length_min_mm": [length_min],
    "length_max_mm": [length_max],
    "weight_min_g": [weight_min],
    "weight_max_g": [weight_max],
    "degree": [degree],
    "cv_folds": [cv],
    "mse_mean": [mse_mean],
    "mse_std": [mse_std],
    "rmse_mean": [rmse_mean],
    "rmse_std": [rmse_std],
    "mae_mean": [mae_mean],
    "mae_std": [mae_std],
    "mape_mean": [mape_mean],
    "mape_std": [mape_std],
    "me_mean": [me_mean],
    "me_std": [me_std],
    "r2_mean": [r2_mean],
    "r2_std": [r2_std],
    "pearson_corr": [corr],
    "model_path": [model_path]
}

df_metrics = pd.DataFrame(metrics_dict)
df_metrics.to_csv(csv_path, index=False, encoding="utf-8")

print(f"CSV metrics saved to: {csv_path}")

# Plot Figures
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.6)
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range = model.predict(X_range)
plt.plot(X_range, y_range)
plt.xlabel("Actual Length (mm)")
plt.ylabel("Actual Weight (g)")
plt.title("Actual Length vs Actual Weight with Fitted Curve")
plt.savefig(os.path.join(plot_dir, "length_vs_actual_weight.png"), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(predicted, y, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='--')
plt.xlabel("Predicted Weight (g)")
plt.ylabel("Actual Weight (g)")
plt.title("Predicted Weight vs Actual Weight")
plt.savefig(os.path.join(plot_dir, "predicted_vs_actual_weight.png"), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
residuals = y - predicted
plt.scatter(y, residuals, alpha=0.6)
plt.axhline(0, linestyle='--')
plt.xlabel("Actual Weight (g)")
plt.ylabel("Residuals (g)")
plt.title("Residual Plot")
plt.savefig(os.path.join(plot_dir, "residual_plot.png"), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.boxplot([X.flatten(), y], labels=['Length (mm)', 'Weight (g)'])
plt.title("Boxplot of Length and Weight")
plt.savefig(os.path.join(plot_dir, "boxplot_length_weight.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Figures saved.")

# Save polynomial formula
poly_features = model.named_steps['polynomialfeatures']
linear_model = model.named_steps['linearregression']

feature_names = poly_features.get_feature_names_out(input_features=['L'])
coefficients = linear_model.coef_
intercept = linear_model.intercept_

text_formula = f"Weight (g) = {intercept:.6f}"
for coef, name in zip(coefficients, feature_names):
    sign = '+' if coef >= 0 else '-'
    text_formula += f" {sign} {abs(coef):.6f} * {name}"

latex_formula = r"\[ \text{Weight (g)} = " + f"{intercept:.6f}"
for coef, name in zip(coefficients, feature_names):
    sign = '+' if coef >= 0 else '-'
    latex_formula += f" {sign} {abs(coef):.6f} \\cdot {name}"
latex_formula += r" \]"

latex_path = os.path.join(formula_dir, 'regression_formula.tex')
text_formula_path = os.path.join(formula_dir, 'regression_formula.txt')

with open(latex_path, 'w', encoding='utf-8') as f:
    f.write(latex_formula)
with open(text_formula_path, 'w', encoding='utf-8') as f:
    f.write(text_formula)

print("Formulas saved.")
