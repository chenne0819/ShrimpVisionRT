import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import joblib
import pandas as pd
import numpy as np
import os

# Set English font
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Define output directories
output_dir = "./shrimp weight regression with both length and width/shrimp_weight_regression_LW_3D"
plots_dir = os.path.join(output_dir, "plots")
models_dir = os.path.join(output_dir, "models")
formulas_dir = os.path.join(output_dir, "formulas")

# Create directories if they do not exist
for d in [plots_dir, models_dir, formulas_dir]:
    os.makedirs(d, exist_ok=True)

# Load training data
df = pd.read_excel(r"shrimp_len_wid_wei.xlsx")
print("Columns in dataset:", df.columns)

# Extract features and target variable
X = df['length (mm)'].to_numpy()  # Length (mm)
M = df['width (mm)'].to_numpy()  # Width (mm)
y = df['weight (g)'].to_numpy()  # Weight (g)

# Combine features into multi-feature matrix [Length, Width]
X_multi = np.column_stack((X, M))

# Initialize polynomial regression model (degree 3)
degree = 3
model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())

# 5-fold cross-validation
cv = 5
kf = KFold(n_splits=cv, shuffle=True, random_state=42)

# Compute cross-validated MSE and R²
mse_scores = cross_val_score(model, X_multi, y, cv=kf, scoring='neg_mean_squared_error')
r2_scores = cross_val_score(model, X_multi, y, cv=kf, scoring='r2')
mse_mean = -np.mean(mse_scores)
mse_std = np.std(mse_scores)
r2_mean = np.mean(r2_scores)
r2_std = np.std(r2_scores)
rmse_mean = np.sqrt(mse_mean)
rmse_std = mse_std / (2 * np.sqrt(mse_mean))

# Manual CV for MAE, MAPE, ME
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

# Output CV metrics
print(f"X_multi shape: {X_multi.shape}")
print(f"y shape: {y.shape}")
print(f"CV folds: {cv}")
print(f"MSE: {mse_mean:.4f} ± {mse_std:.4f} g^2")
print(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f} g")
print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f} g")
print(f"MAPE: {mape_mean:.4f} ± {mape_std:.4f} %")
print(f"ME: {me_mean:.4f} ± {me_std:.4f} g")
print(f"Mean R²: {r2_mean:.4f} ± {r2_std:.4f}")

# Get CV predictions and Pearson correlation
predicted = cross_val_predict(model, X_multi, y, cv=cv)
corr, _ = pearsonr(y, predicted)
print(f"Pearson correlation coefficient: {corr:.4f}")

# Save multi-feature model
joblib.dump(model, os.path.join(models_dir, 'multi_feature_model.pkl'))

# Fit model for plotting
model.fit(X_multi, y)

# Define 3D curve along diagonal of length and width
t = np.linspace(0, 1, 100)
x_curve = X.min() + t * (X.max() - X.min())
m_curve = M.min() + t * (M.max() - M.min())
X_curve = np.column_stack((x_curve, m_curve))
y_curve = model.predict(X_curve)

# 3D plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, M, y, alpha=0.6, label='Data Points')
ax.plot(x_curve, m_curve, y_curve, color='red', linewidth=2, label='Fitted Curve')
ax.set_xlabel('Length (mm)')
ax.set_ylabel('Width (mm)')
ax.set_zlabel('Weight (g)')
ax.set_title('3D Length-Width-Weight with Fitted Curve')
ax.legend()
plt.savefig(os.path.join(plots_dir, "3d_length_width_weight_curve.png"), dpi=300, bbox_inches='tight')
plt.show()

# Fit separate 1D models for Length and Width
X_reshaped = X.reshape(-1,1)
model_X = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
model_X.fit(X_reshaped, y)
joblib.dump(model_X, os.path.join(models_dir, 'length_model.pkl'))

M_reshaped = M.reshape(-1,1)
model_M = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
model_M.fit(M_reshaped, y)
joblib.dump(model_M, os.path.join(models_dir, 'width_model.pkl'))

# 2D plots with fit curves
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_range_X = model_X.predict(X_range)
M_range = np.linspace(M.min(), M.max(), 100).reshape(-1,1)
y_range_M = model_M.predict(M_range)

plt.figure(figsize=(8,6))
plt.scatter(X, y, alpha=0.6, color='blue', label='Length vs Weight')
plt.scatter(M, y, alpha=0.6, color='green', label='Width vs Weight')
plt.plot(X_range, y_range_X, 'r-', label='Length Fit Curve')
plt.plot(M_range, y_range_M, 'r--', label='Width Fit Curve')
plt.xlabel('Length / Width (mm)')
plt.ylabel('Weight (g)')
plt.title('Length and Width vs Weight with Fit Curves')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "length_width_vs_weight.png"), dpi=300, bbox_inches='tight')
plt.close()

# Residual plot
residuals = y - predicted
plt.figure(figsize=(8,6))
plt.scatter(y, residuals, alpha=0.6)
for i in range(len(y)):
    plt.plot([y[i], y[i]], [0, residuals[i]], color='gray', linestyle='--', alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Actual Weight (g)")
plt.ylabel("Residuals (g)")
plt.title("Residual Plot")
plt.savefig(os.path.join(plots_dir, "residual_plot.png"), dpi=300, bbox_inches='tight')
plt.close()

# Boxplot of Length, Width, Weight
plt.figure(figsize=(8,6))
plt.boxplot([X, M, y], labels=['Length (mm)', 'Width (mm)', 'Weight (g)'])
plt.title("Boxplot of Length, Width, Weight")
plt.savefig(os.path.join(plots_dir, "boxplot_length_width_weight.png"), dpi=300, bbox_inches='tight')
plt.close()

# Export polynomial regression formulas (multi-feature)
poly_features = model.named_steps['polynomialfeatures']
linear_model = model.named_steps['linearregression']
feature_names = poly_features.get_feature_names_out(input_features=['L','W'])
coefficients = linear_model.coef_
intercept = linear_model.intercept_

text_formula = f"Weight (g) = {intercept:.4f}"
latex_formula = r"\[ \text{Weight (g)} = " + f"{intercept:.4f}"
for coef, name in zip(coefficients, feature_names):
    if coef != 0:
        sign = '+' if coef > 0 else '-'
        text_formula += f" {sign} {abs(coef):.4f} * {name}"
        latex_formula += f" {sign} {abs(coef):.4f} \cdot {name}"
latex_formula += r" \]"

with open(os.path.join(formulas_dir, 'multi_feature_formula.tex'), 'w', encoding='utf-8') as f:
    f.write(latex_formula)
print(f"Multi-feature LaTeX formula saved at {formulas_dir}")
print("Polynomial regression formula (text):", text_formula)
