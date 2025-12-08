# Points grid is 22 x 16
import cv2
import numpy as np
import pandas as pd
import os

# Set paths
csv_file = r"point22_16.csv"  # Path to CSV file
input_dir = r'img'             # Input image folder
output_dir = r'undistort_img'  # Output folder for undistorted images

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Check if CSV file exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file {csv_file} does not exist. Please check the path.")

# Read CSV file
try:
    df = pd.read_csv(csv_file, header=None)
except Exception as e:
    print(f"Failed to read CSV file: {e}")
    raise

# Extract image points (columns 2 and 3, index 1 and 2)
img_points = df.iloc[:, [1, 2]].values.astype(np.float32)  # Points in image (x, y)

# Image dimensions (from columns 5 and 6, index 4 and 5)
img_width = int(df.iloc[0, 4])  # 800
img_height = int(df.iloc[0, 5])  # 450

# Sort points by y coordinate, then by x coordinate
rows, cols = 16, 22
sorted_indices = np.argsort(img_points[:, 1])
sorted_img_points = img_points[sorted_indices]
groups = np.array_split(sorted_img_points, rows)
sorted_groups = [group[np.argsort(group[:, 0])] for group in groups]
sorted_img_points = np.vstack(sorted_groups)

# Generate ideal object points grid (22x16, z=0)
obj_points = np.zeros((rows * cols, 3), np.float32)
x_ideal = np.linspace(0, img_width, cols)
y_ideal = np.linspace(0, img_height, rows)
x_ideal, y_ideal = np.meshgrid(x_ideal, y_ideal)
obj_points[:, :2] = np.vstack([x_ideal.ravel(), y_ideal.ravel()]).T

# Prepare data for calibration
obj_points = [obj_points]         # Single image, one set of object points
img_points = [sorted_img_points]  # Single image, one set of image points

# Initialize camera intrinsic matrix
focal_length = max(img_width, img_height)
camera_matrix = np.array([[focal_length, 0, img_width/2],
                          [0, focal_length, img_height/2],
                          [0, 0, 1]], dtype=np.float32)

# Initialize distortion coefficients
dist_coeffs = np.zeros((5, 1), np.float32)  # k1, k2, p1, p2, k3

# Perform camera calibration
try:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (img_width, img_height), camera_matrix, dist_coeffs,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL)
except cv2.error as e:
    print(f"Calibration failed: {e}")
    raise

# Process all images in the input folder
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.png', '.bmp')):  # Supported image formats
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"corrected_{filename}")

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image {img_path}, skipping.")
            continue

        # Check if image size matches CSV-specified size
        h, w = image.shape[:2]
        if w != img_width or h != img_height:
            print(f"Image {img_path} size is {w}x{h}, expected {img_width}x{img_height}. Resizing.")
            image = cv2.resize(image, (img_width, img_height))

        # Undistort image
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

        # Save undistorted image
        cv2.imwrite(output_path, undistorted_image)
        print(f"Image {filename} corrected and saved as {output_path}")

print("All images have been undistorted!")
