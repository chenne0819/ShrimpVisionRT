# Underwater Image Calibration & Distortion Correction

This directory contains the preprocessing module for the **Shrimp Size & Weight Analysis** project. It focuses on correcting optical distortions caused by the underwater environment to ensure the accuracy of subsequent length and width measurements.

---

## Problem Statement: Underwater Distortion

In underwater photography, light refraction at the water-glass-air interface (camera housing) often introduces significant **radial and tangential distortions**. These distortions cause straight lines to appear curved (often resulting in pincushion or barrel distortion) and alter the spatial relationship between pixels.

For our research, which relies on precise **pixel-to-millimeter conversion** to estimate shrimp size and weight, these distortions can introduce substantial errors. If left uncorrected, the geometric deformation would lead to inaccurate length and width estimations, ultimately affecting the weight regression model.

**Objective:**
To minimize measurement error, we perform **camera calibration** using a reference grid plate *before* conducting the main biological experiments. This script calculates the distortion coefficients and un-distorts all experimental images to restore their geometric fidelity.

---

##  Folder Structure

All calibration resources are located in the `image_calibration` directory:

<pre>
image_calibration/
│
├── undistort_all_img.py          # Main script for performing calibration and correction
├── point22_16.csv                # Annotated grid points (22x16) used for calibration
├── grid_plate_for_labeling.jpg   # The reference image of the grid plate used to generate the CSV
│
├── img/                          # [Input] Place raw, distorted images here
│   └── (e.g., sample_shrimp.jpg)
│
└── undistort_img/                # [Output] Corrected images will be saved here
    └── (e.g., corrected_sample_shrimp.jpg)
</pre>

---

## How It Works

1.  **Reference Data**:
    A **Grid Plate** (`grid_plate_for_labeling.jpg`) was placed underwater and manually annotated the intersection points of the grid (22 columns x 16 rows) and saved them to `point22_16.csv`.

2.  **Calibration Process (`undistort_all_img.py`)**:
    * The script reads the distorted points from the CSV.
    * It generates a corresponding "Ideal Grid" (perfectly spaced mathematical coordinates).
    * Using `cv2.calibrateCamera`, it calculates the **Camera Matrix** and **Distortion Coefficients** required to map the distorted points back to the ideal grid.

3.  **Batch Correction**:
    * The script iterates through all images in the `img/` folder.
    * It applies the calculated correction matrix to each image using `cv2.undistort`.
    * The corrected results are saved to `undistort_img/`.

---

## Usage

### 1. Prerequisites
Ensure you have the required Python libraries installed:
```bash
   pip install opencv-python numpy pandas
```

### 2. Prepare Data
- Ensure point22_16.csv is present (this provides the calibration logic).

- Place the images you want to correct into the img/ folder.

### 3. Run the Script
Execute the Python script from the terminal:

```bash
    cd image_calibration
    python undistort_all_img.py
```

### 4. Check Results
The script will process the images and output the message:

Image [filename] corrected and saved as undistort_img/corrected_[filename]

Navigate to the undistort_img/ folder to view the calibrated images ready for size estimation.

---

### Notes
Image Dimensions: The script assumes the calibration was performed on images of size 800x450. If input images in img/ differ, the script will automatically resize them to match the calibration profile.

Grid Specs: The calibration logic is based on a 22x16 intersection grid.

---

## Visual Results

### Img
<img src=".\img\2025-05-07-16_24_34_frame_0.jpg" width="600">

### Undistort Img
<img src=".\undistort_img\corrected_2025-05-07-16_24_34_frame_0.jpg" width="500">