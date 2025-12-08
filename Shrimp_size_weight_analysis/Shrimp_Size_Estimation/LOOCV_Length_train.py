import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib

# Set font for English display
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Create main output folders
output_dir = 'LOOCV_Length_predict'
os.makedirs(output_dir, exist_ok=True)

# Subfolders for organization
train_dir = os.path.join(output_dir, 'train_leave_one_out')
plots_dir = os.path.join(output_dir, 'plots')
summary_dir = os.path.join(output_dir, 'summary')
final_model_dir = os.path.join(output_dir, 'final_model')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)
os.makedirs(final_model_dir, exist_ok=True)

# Read Excel file
data = pd.read_excel(r"data.xlsx")

# Extract features and target
X = data[['Image Estimated Length (mm)']].values
y = data['Actual Length (mm)'].values

# Check data size
n_samples = len(data)
print(f"Total data points: {n_samples}")

# Initialize storage for metrics and predictions
metrics_all_runs = []
all_predictions = []
used_test_indices = []

# Leave-One-Out Cross-Validation (LOOCV)
for run in range(1, n_samples + 1):
    print(f"\nRun: {run}")

    # Select available test index
    available_indices = [i for i in range(n_samples) if i not in used_test_indices]
    if not available_indices:
        print(f"Error: No available test indices for run {run}. Stopping.")
        break
    test_index = available_indices[0]
    used_test_indices.append(test_index)

    # Split train/test data
    test_mask = np.zeros(n_samples, dtype=bool)
    test_mask[test_index] = True
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    print(f"Test index: {test_index}")

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict test sample
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1e-10, y_test))) * 100
    me = np.mean(y_test - y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'Run': run,
        'Slope': model.coef_[0],
        'Intercept': model.intercept_,
        'MSE (mm²)': mse,
        'RMSE (mm)': rmse,
        'MAE (mm)': mae,
        'MAPE (%)': mape,
        'ME (mm)': me,
        'R²': r2
    }
    metrics_all_runs.append(metrics)

    # Store predictions
    for x, y_true, y_p in zip(X_test.flatten(), y_test, y_pred):
        all_predictions.append({
            'Image Estimated Length (mm)': x,
            'Actual Length (mm)': y_true,
            'Predicted Length (mm)': y_p
        })

    # Print run metrics
    print(f"Slope: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"MSE (mm²): {mse:.4f}")
    print(f"RMSE (mm): {rmse:.4f}")
    print(f"MAE (mm): {mae:.4f}")
    print(f"MAPE (%): {mape:.4f}")
    print(f"ME (mm): {me:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Create plots for this run
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Regression line plot
    ax1.scatter(X_test, y_test, label='Test Data')
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    ax1.plot(X_line, y_line, color='red', linewidth=2, label='Predicted Regression Line')
    ax1.set_xlabel('Image Estimated Length (mm)')
    ax1.set_ylabel('Actual Length (mm)')
    ax1.set_title(f'Test Data vs Predicted Regression Line (Run {run})')
    ax1.legend()
    ax1.grid(True)

    # Error plot
    errors = y_test - y_pred
    ax2.scatter(X_test, errors, label='Prediction Error')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    for x, err in zip(X_test.flatten(), errors):
        ax2.vlines(x=x, ymin=0, ymax=err, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Image Estimated Length (mm)')
    ax2.set_ylabel('Prediction Error (mm)')
    ax2.set_title(f'Prediction Error of Test Data (Run {run})')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'combined_plot_run_{run}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save prediction results for this run
    results = pd.DataFrame({
        'Image Estimated Length (mm)': X_test.flatten(),
        'Actual Length (mm)': y_test,
        'Predicted Length (mm)': y_pred,
        'Error (mm)': errors
    })
    results.to_excel(os.path.join(train_dir, f'predicted_results_run_{run}.xlsx'), index=False)

# Combine metrics and calculate mean/std
metrics_df = pd.DataFrame(metrics_all_runs)
mean_metrics = metrics_df.mean(numeric_only=True)
std_metrics = metrics_df.std(numeric_only=True)

# Label mean and std rows properly (跟 Width 寫法一致)
summary_df = pd.DataFrame([
    {
        'Run': 'Mean',
        'Slope': mean_metrics['Slope'],
        'Intercept': mean_metrics['Intercept'],
        'MSE (mm²)': mean_metrics['MSE (mm²)'],
        'RMSE (mm)': mean_metrics['RMSE (mm)'],
        'MAE (mm)': mean_metrics['MAE (mm)'],
        'MAPE (%)': mean_metrics['MAPE (%)'],
        'ME (mm)': mean_metrics['ME (mm)'],
        'R²': mean_metrics['R²']
    },
    {
        'Run': 'Std',
        'Slope': std_metrics['Slope'],
        'Intercept': std_metrics['Intercept'],
        'MSE (mm²)': std_metrics['MSE (mm²)'],
        'RMSE (mm)': std_metrics['RMSE (mm)'],
        'MAE (mm)': std_metrics['MAE (mm)'],
        'MAPE (%)': std_metrics['MAPE (%)'],
        'ME (mm)': std_metrics['ME (mm)'],
        'R²': std_metrics['R²']
    }
])

# Save full metrics
final_df = pd.concat([metrics_df, summary_df], ignore_index=True)
final_df.to_excel(os.path.join(summary_dir, f'metrics_{n_samples}_runs.xlsx'), sheet_name='Metrics', index=False)

# Save all predictions
predictions_df = pd.DataFrame(all_predictions)
predictions_df.to_excel(os.path.join(summary_dir, 'all_loocv_predictions.xlsx'), index=False)

# Train final model on all data
final_model = LinearRegression()
final_model.fit(X, y)
joblib.dump(final_model, os.path.join(final_model_dir, 'final_linear_model_length.pkl'))

# Save final model metrics
final_metrics = {'Slope': final_model.coef_[0], 'Intercept': final_model.intercept_}
pd.DataFrame([final_metrics]).to_excel(os.path.join(final_model_dir, 'final_model_metrics_length.xlsx'), index=False)

# Evaluate final model on training data
y_pred_train = final_model.predict(X)
errors_train = y - y_pred_train
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_train, y, alpha=0.6, label='Training Data')
min_val = min(y_pred_train.min(), y.min())
max_val = max(y_pred_train.max(), y.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='y=x')
plt.xlabel('Predicted Length (mm)')
plt.ylabel('Actual Length (mm)')
plt.title('Final Model Regression')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'final_model_regression_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_train, errors_train, alpha=0.6, label='Prediction Error')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
for x, err in zip(y_pred_train.flatten(), errors_train):
    plt.vlines(x=x, ymin=0, ymax=err, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Predicted Length (mm)')
plt.ylabel('Prediction Error (mm)')
plt.title('Final Model Prediction Error')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'final_model_error_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
