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

# Configure English display for plots
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Ensure output directories exist
output_root = ".\shrimp weight regression with both length and width\shrimp_weight_regression_LW"
models_dir = os.path.join(output_root, "models")
plots_dir = os.path.join(output_root, "plots")
formulas_dir = os.path.join(output_root, "formulas")
summary_dir = os.path.join(output_root, "summary")

for d in [output_root, models_dir, plots_dir, formulas_dir, summary_dir]:
    os.makedirs(d, exist_ok=True)

# Load dataset
df = pd.read_excel(r"shrimp_len_wid_wei.xlsx")
print(df.columns)

# Extract features and target variable
X = df['length (mm)'].to_numpy()  # Length (mm)
M = df['width (mm)'].to_numpy()  # Width (mm)
y = df['weight (g)'].to_numpy()  # Weight (g)

# Combine features into multi-feature matrix
X_multi = np.column_stack((X, M))

# Initialize polynomial regression model (degree=3)
degree = 3
model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())

# Perform 5-fold cross-validation
cv = 5
kf = KFold(n_splits=cv, shuffle=True, random_state=42)

# Compute MSE and R²
mse_scores = cross_val_score(model, X_multi, y, cv=kf, scoring='neg_mean_squared_error')
r2_scores = cross_val_score(model, X_multi, y, cv=kf, scoring='r2')

mse_mean = -np.mean(mse_scores)  # g²
mse_std = np.std(mse_scores)
r2_mean = np.mean(r2_scores)
r2_std = np.std(r2_scores)

# Compute RMSE
rmse_mean = np.sqrt(mse_mean)
rmse_std = mse_std / (2 * np.sqrt(mse_mean))

# Manually compute MAE, MAPE, ME using CV
mae_scores, mape_scores, me_scores = [], [], []

for train_index, test_index in kf.split(X_multi):
    X_train, X_test = X_multi[train_index], X_multi[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae_scores.append(np.mean(np.abs(y_test - y_pred)))
    mape_scores.append(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
    me_scores.append(np.mean(y_test - y_pred))

mae_mean, mae_std = np.mean(mae_scores), np.std(mae_scores)
mape_mean, mape_std = np.mean(mape_scores), np.std(mape_scores)
me_mean, me_std = np.mean(me_scores), np.std(me_scores)

# Output metrics
print(f"X_multi shape: {X_multi.shape}")
print(f"y shape: {y.shape}")
print(f"Cross-validation folds: {cv}")
print(f"Mean Squared Error (MSE): {mse_mean:.4f} (±{mse_std:.4f}) g²")
print(f"Root Mean Squared Error (RMSE): {rmse_mean:.4f} (±{rmse_std:.4f}) g")
print(f"Mean Absolute Error (MAE): {mae_mean:.4f} (±{mae_std:.4f}) g")
print(f"Mean Absolute Percentage Error (MAPE): {mape_mean:.4f} (±{mape_std:.4f}) %")
print(f"Mean Error (ME): {me_mean:.4f} (±{me_std:.4f}) g")
print(f"Mean R² score: {r2_mean:.4f} (±{r2_std:.4f})")

# Get predicted values and Pearson correlation
predicted = cross_val_predict(model, X_multi, y, cv=cv)
corr, _ = pearsonr(y, predicted)
print(f"Pearson correlation coefficient: {corr:.4f}")

# Save multi-feature model
joblib.dump(model, os.path.join(models_dir, 'multi_feature_model.pkl'))
print(f"Multi-feature model saved as '{os.path.join(models_dir, 'multi_feature_model.pkl')}'")

# Prepare single-feature fit curves
X_reshaped = X.reshape(-1, 1)
model_X = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
model_X.fit(X_reshaped, y)
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_X = model_X.predict(X_range)
joblib.dump(model_X, os.path.join(models_dir, 'length_model.pkl'))

M_reshaped = M.reshape(-1, 1)
model_M = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
model_M.fit(M_reshaped, y)
M_range = np.linspace(M.min(), M.max(), 100).reshape(-1, 1)
y_range_M = model_M.predict(M_range)
joblib.dump(model_M, os.path.join(models_dir, 'width_model.pkl'))

# Plot 1: Length and Width vs Weight with Fit Curves
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.6, label="Length vs Weight", color='blue')
plt.scatter(M, y, alpha=0.6, label="Width vs Weight", color='green')
plt.plot(X_range, y_range_X, 'r-', label="Length Fit Curve", linewidth=2)
plt.plot(M_range, y_range_M, 'r--', label="Width Fit Curve", linewidth=2)
plt.xlabel("Actual Length/Width (mm)")
plt.ylabel("Weight (g)")
plt.title("Actual Length and Width vs Actual Weight with Fit Curves")
plt.legend()
plt.savefig(os.path.join(plots_dir, "length_width_vs_weight.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Predicted vs Actual Weight
plt.figure(figsize=(8, 6))
plt.scatter(predicted, y, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Predicted Weight (g)")
plt.ylabel("Actual Weight (g)")
plt.title("Predicted vs Actual Weight (Using Actual Length and Width)")
plt.savefig(os.path.join(plots_dir, "predicted_vs_actual_weight.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Residual plot
plt.figure(figsize=(8, 6))
residuals = y - predicted
plt.scatter(y, residuals, alpha=0.6)
for i in range(len(y)):
    plt.plot([y[i], y[i]], [0, residuals[i]], color='gray', linestyle='--', alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Actual Weight (g)")
plt.ylabel("Residuals (g)")
plt.title("Residual Plot (Using Actual Length and Width)")
plt.savefig(os.path.join(plots_dir, "residual_plot.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Boxplot of Length, Width, Weight
plt.figure(figsize=(8, 6))
plt.boxplot([X, M, y], labels=['Length (mm)', 'Width (mm)', 'Weight (g)'])
plt.title("Boxplot of Length, Width, and Weight")
plt.savefig(os.path.join(plots_dir, "boxplot_length_width_weight.png"), dpi=300, bbox_inches='tight')
plt.close()

# Show all plots (optional)
plt.show()

# Export polynomial regression formula (multi-feature)
model.fit(X_multi, y)
poly_features = model.named_steps['polynomialfeatures']
linear_model = model.named_steps['linearregression']
feature_names = poly_features.get_feature_names_out(input_features=['L', 'W'])
coefficients = linear_model.coef_
intercept = linear_model.intercept_

text_formula = f"Weight (g) = {intercept:.4f}"
latex_formula = r"\[ \text{Weight (g)} = " + f"{intercept:.4f}"
for coef, name in zip(coefficients, feature_names):
    if coef != 0:
        sign = '+' if coef > 0 else '-'
        text_formula += f" {sign} {abs(coef):.4f} * {name}"
        name_latex = name.replace('L', 'L').replace('W', 'W').replace(' ', ' \cdot ').replace('^', '^')
        latex_formula += f" {sign} {abs(coef):.4f} \cdot {name_latex}"
latex_formula += r" \]"

with open(os.path.join(formulas_dir, 'multi_feature_formula.tex'), 'w', encoding='utf-8') as f:
    f.write(latex_formula)

# Length-only formula
model_X.fit(X_reshaped, y)
poly_features_X = model_X.named_steps['polynomialfeatures']
linear_model_X = model_X.named_steps['linearregression']
feature_names_X = poly_features_X.get_feature_names_out(input_features=['L'])
coefficients_X = linear_model_X.coef_
intercept_X = linear_model_X.intercept_

text_formula_X = f"Weight (g) = {intercept_X:.4f}"
latex_formula_X = r"\[ \text{Weight (g)} = " + f"{intercept_X:.4f}"
for coef, name in zip(coefficients_X, feature_names_X):
    if coef != 0:
        sign = '+' if coef > 0 else '-'
        text_formula_X += f" {sign} {abs(coef):.4f} * {name}"
        name_latex = name.replace('L', 'L').replace('^', '^')
        latex_formula_X += f" {sign} {abs(coef):.4f} \cdot {name_latex}"
latex_formula_X += r" \]"

with open(os.path.join(formulas_dir, 'length_formula.tex'), 'w', encoding='utf-8') as f:
    f.write(latex_formula_X)

# Width-only formula
model_M.fit(M_reshaped, y)
poly_features_M = model_M.named_steps['polynomialfeatures']
linear_model_M = model_M.named_steps['linearregression']
feature_names_M = poly_features_M.get_feature_names_out(input_features=['W'])
coefficients_M = linear_model_M.coef_
intercept_M = linear_model_M.intercept_

text_formula_M = f"Weight (g) = {intercept_M:.4f}"
latex_formula_M = r"\[ \text{Weight (g)} = " + f"{intercept_M:.4f}"
for coef, name in zip(coefficients_M, feature_names_M):
    if coef != 0:
        sign = '+' if coef > 0 else '-'
        text_formula_M += f" {sign} {abs(coef):.4f} * {name}"
        name_latex = name.replace('W', 'W').replace('^', '^')
        latex_formula_M += f" {sign} {abs(coef):.4f} \cdot {name_latex}"
latex_formula_M += r" \]"

with open(os.path.join(formulas_dir, 'width_formula.tex'), 'w', encoding='utf-8') as f:
    f.write(latex_formula_M)

# Save cross-validation metrics as CSV
metrics_dict = {
    "MSE (g^2)": [mse_mean],
    "MSE std": [mse_std],
    "RMSE (g)": [rmse_mean],
    "RMSE std": [rmse_std],
    "MAE (g)": [mae_mean],
    "MAE std": [mae_std],
    "MAPE (%)": [mape_mean],
    "MAPE std": [mape_std],
    "ME (g)": [me_mean],
    "ME std": [me_std],
    "R2": [r2_mean],
    "R2 std": [r2_std],
    "Pearson correlation": [corr]
}

metrics_df = pd.DataFrame(metrics_dict)
metrics_csv_path = os.path.join(summary_dir, "output_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Cross-validation metrics saved as '{metrics_csv_path}'")
