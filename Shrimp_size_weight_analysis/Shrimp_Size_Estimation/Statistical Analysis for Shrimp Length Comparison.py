import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load LOOCV Length Prediction Results
df_path = os.path.join('LOOCV_Length_predict', 'summary', 'all_loocv_predictions.xlsx')
df = pd.read_excel(df_path)

# Extract actual and predicted values
actual = df['Actual Length (mm)'].values
predicted = df['Predicted Length (mm)'].values

# Calculate differences
differences = predicted - actual

print("=" * 70)
print("Statistical Analysis Results: Manual vs LOOCV Prediction (Length)")
print("=" * 70)

# 1. Paired t-test
print("\n[1. Paired t-test]")
print("-" * 70)
t_stat, p_value_t = stats.ttest_rel(predicted, actual)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value_t:.4f}")
print(f"Degrees of freedom: {len(actual) - 1}")
print(f"Mean difference: {np.mean(differences):.4f} mm")

if p_value_t > 0.05:
    print("Conclusion: p > 0.05, no significant difference between methods ✓")
else:
    print("Conclusion: p ≤ 0.05, significant difference detected")

# 2. Wilcoxon signed-rank test
print("\n[2. Wilcoxon signed-rank test (Non-parametric)]")
print("-" * 70)
w_stat, p_value_w = stats.wilcoxon(predicted, actual)
print(f"Wilcoxon statistic: {w_stat:.4f}")
print(f"p-value: {p_value_w:.4f}")

if p_value_w > 0.05:
    print("Conclusion: p > 0.05, no significant difference between methods ✓")
else:
    print("Conclusion: p ≤ 0.05, significant difference detected")

# 3. Bland-Altman Analysis
print("\n[3. Bland-Altman Analysis]")
print("-" * 70)
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)
upper_loa = mean_diff + 1.96 * std_diff
lower_loa = mean_diff - 1.96 * std_diff

print(f"Mean Difference: {mean_diff:.4f} mm")
print(f"Standard Deviation (SD): {std_diff:.4f} mm")
print(f"Upper Limit of Agreement (LoA): {upper_loa:.4f} mm")
print(f"Lower Limit of Agreement (LoA): {lower_loa:.4f} mm")
print(f"95% Limits of Agreement: [{lower_loa:.4f}, {upper_loa:.4f}] mm")

within_loa = np.sum((differences >= lower_loa) & (differences <= upper_loa))
percentage = (within_loa / len(differences)) * 100
print(f"Samples within 95% LoA: {within_loa}/{len(differences)} ({percentage:.1f}%)")

# 4. Descriptive Statistics
print("\n[4. Descriptive Statistics]")
print("-" * 70)
print(f"Sample size: {len(actual)}")
print(f"RMSE: {np.sqrt(np.mean(differences**2)):.4f} mm")
print(f"MAE: {np.mean(np.abs(differences)):.4f} mm")
print(f"Pearson correlation: {stats.pearsonr(actual, predicted)[0]:.4f}")

# 5. Create output folder for plots
output_folder = 'LOOCV_Length_statistical_analysis_results'
os.makedirs(output_folder, exist_ok=True)

mean_values = (actual + predicted) / 2
box_data = [actual, predicted]
abs_errors = np.abs(differences)

# Bland-Altman Plot
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(mean_values, differences, alpha=0.6, s=60, color='steelblue', edgecolor='black', linewidth=0.5)
ax1.axhline(mean_diff, color='blue', linestyle='-', linewidth=2.5, label=f'Mean Diff = {mean_diff:.2f} mm')
ax1.axhline(upper_loa, color='red', linestyle='--', linewidth=2, label=f'Upper LoA = {upper_loa:.2f} mm')
ax1.axhline(lower_loa, color='red', linestyle='--', linewidth=2, label=f'Lower LoA = {lower_loa:.2f} mm')
ax1.axhline(0, color='gray', linestyle=':', linewidth=1.5)
ax1.set_xlabel('Mean of Actual and Predicted Length (mm)')
ax1.set_ylabel('Difference (Predicted - Actual) (mm)')
ax1.set_title('Bland-Altman Plot')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '1_bland_altman_plot.png'), dpi=300)
plt.close()

# Scatter Plot with Regression Line
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(actual, predicted, alpha=0.6, s=60, color='steelblue', edgecolor='black', linewidth=0.5)
ax2.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', linewidth=2.5, label='Perfect Agreement (y=x)')
z = np.polyfit(actual, predicted, 1)
p = np.poly1d(z)
ax2.plot(actual, p(actual), "g-", linewidth=2.5, alpha=0.7, label=f'Regression (y={z[0]:.3f}x+{z[1]:.2f})')
ax2.set_xlabel('Actual Length (mm)')
ax2.set_ylabel('Predicted Length (mm)')
r_value = stats.pearsonr(actual, predicted)[0]
ax2.set_title(f'Actual vs Predicted Length (r = {r_value:.4f})')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '2_actual_vs_predicted.png'), dpi=300)
plt.close()

# Residual Distribution Histogram
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.hist(differences, bins=15, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
ax3.axvline(0, color='red', linestyle='--', linewidth=2.5, label='Zero Difference')
ax3.axvline(mean_diff, color='blue', linestyle='-', linewidth=2.5, label=f'Mean = {mean_diff:.2f} mm')
ax3.set_xlabel('Difference (Predicted - Actual) (mm)')
ax3.set_ylabel('Frequency')
ax3.set_title('Residual Distribution')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '3_residual_distribution.png'), dpi=300)
plt.close()

# Q-Q Plot for Normality Check
fig4, ax4 = plt.subplots(figsize=(8, 6))
stats.probplot(differences, dist="norm", plot=ax4)
ax4.set_title('Q-Q Plot (Normality Check)')
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '4_qq_plot.png'), dpi=300)
plt.close()

# Box Plot Comparison
fig5, ax5 = plt.subplots(figsize=(8, 6))
bp = ax5.boxplot(box_data, labels=['Actual', 'Predicted'], patch_artist=True,
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))
ax5.set_ylabel('Length (mm)')
ax5.set_title('Distribution Comparison')
ax5.grid(True, alpha=0.3, axis='y')
ax5.text(0.5, 0.97, f'Paired t-test: p = {p_value_t:.4f}\nWilcoxon test: p = {p_value_w:.4f}',
         transform=ax5.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '5_distribution_comparison.png'), dpi=300)
plt.close()

# Absolute Error Distribution
fig6, ax6 = plt.subplots(figsize=(8, 6))
ax6.hist(abs_errors, bins=15, color='coral', alpha=0.7, edgecolor='black', linewidth=1.2)
ax6.axvline(np.mean(abs_errors), color='blue', linestyle='-', linewidth=2.5, label=f'MAE = {np.mean(abs_errors):.2f} mm')
ax6.axvline(np.sqrt(np.mean(differences**2)), color='green', linestyle='--', linewidth=2.5, label=f'RMSE = {np.sqrt(np.mean(differences**2)):.2f} mm')
ax6.set_xlabel('Absolute Error (mm)')
ax6.set_ylabel('Frequency')
ax6.set_title('Absolute Error Distribution')
ax6.legend(loc='best')
ax6.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '6_absolute_error_distribution.png'), dpi=300)
plt.close()

# Normality Test (Shapiro-Wilk)
print("\n[5. Normality Test (Shapiro-Wilk)]")
print("-" * 70)
shapiro_stat, shapiro_p = stats.shapiro(differences)
print(f"Shapiro-Wilk statistic: {shapiro_stat:.4f}")
print(f"p-value: {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("Conclusion: p > 0.05, differences follow normal distribution ✓")
    print("Paired t-test is appropriate")
else:
    print("Conclusion: p ≤ 0.05, differences do not follow normal distribution")
    print("Wilcoxon test is recommended")

print("\n" + "=" * 70)
print("Analysis completed! All plots saved to folder:", output_folder)
print("=" * 70)
