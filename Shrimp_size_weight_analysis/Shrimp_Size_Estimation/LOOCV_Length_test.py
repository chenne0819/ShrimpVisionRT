import numpy as np
import joblib
import os

# Set output directory
output_dir = 'LOOCV_Length_Prediction'  # updated folder name
test_dir = os.path.join(output_dir, 'test_results')
os.makedirs(test_dir, exist_ok=True)

# Load the final model from the same structure as width
model_path = os.path.join('LOOCV_Length_predict', 'final_model', 'final_linear_model_length.pkl')
final_model = joblib.load(model_path)
print(f"Loaded model from '{model_path}'")

# Input a single parameter for prediction
input_length = float(115.3)  # Example input: 115.3
X_input = np.array([[input_length]])

# Perform prediction using the model
y_pred = final_model.predict(X_input)[0]
print(f"\nPredicted Actual Length (mm): {y_pred:.4f}")

# # Optional: Save prediction result to Excel
# result_df = pd.DataFrame({
#     'Image Estimated Length (mm)': [input_length],
#     'Predicted Actual Length (mm)': [y_pred]
# })
# output_file = os.path.join(test_dir, f'prediction_for_input_length_{input_length:.2f}.xlsx')
# result_df.to_excel(output_file, index=False)
# print(f"Prediction result saved to '{output_file}'")
