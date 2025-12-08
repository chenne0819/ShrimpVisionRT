import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
import os

# Set English font
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Define main output folder
output_dir = r"./linear_relationship/linear_length_comparison"
plots_dir = os.path.join(output_dir, "plots")
summary_dir = os.path.join(output_dir, "summary")

# Create folders if they do not exist
for d in [plots_dir, summary_dir]:
    os.makedirs(d, exist_ok=True)

# Load dataset
data = pd.read_excel(r"./Shrimp_Size_Estimation/data.xlsx")

# Extract image estimated length and actual length
image_length = data['Image Estimated Length (mm)']
actual_length = data['Actual Length (mm)']

# Compute Pearson correlation
pearson_corr, _ = pearsonr(image_length, actual_length)
print(f"Pearson correlation: {pearson_corr:.4f}")

# Save data to Excel in summary folder
output_data = pd.DataFrame({
    'Image Estimated Length (mm)': image_length,
    'Actual Length (mm)': actual_length
})
output_data.to_excel(os.path.join(summary_dir, 'shrimp_length_data.xlsx'), index=False)

# Plot scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(image_length, actual_length, alpha=0.6, label='Data Points')
plt.xlabel('Image Estimated Length (mm)')
plt.ylabel('Actual Length (mm)')
plt.title('Shrimp Image Estimated Length vs Actual Length')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'scatter_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(image_length, actual_length)

# Save regression results to Excel in summary folder
regression_results = pd.DataFrame({
    'Metric': ['Slope', 'Intercept', 'R', 'R²', 'P-value'],
    'Value': [slope, intercept, r_value, r_value**2, p_value]
})
regression_results.to_excel(os.path.join(summary_dir, 'regression_results.xlsx'), index=False)

# Print regression results
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R: {r_value:.4f}")
print(f"R²: {r_value**2:.4f}")
print(f"P-value: {p_value:.4e}")

# Plot regression line with scatter
plt.figure(figsize=(8, 6))
plt.scatter(image_length, actual_length, alpha=0.6, label='Data Points')
plt.plot(image_length, slope * image_length + intercept, color='red', label='Regression Line')
plt.xlabel('Image Estimated Length (mm)')
plt.ylabel('Actual Length (mm)')
plt.title('Image Estimated Length vs Actual Length with Regression Line')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'regression_plot.png'), dpi=300, bbox_inches='tight')
plt.close()
