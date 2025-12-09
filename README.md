# Real-time Shrimp Weight Estimation System based on Computer Vision

This repository contains a comprehensive framework for **non-invasive, real-time shrimp monitoring**. It integrates underwater image calibration, biological data analysis, and advanced deep learning models (YOLOv5-OBB, YOLOv8-Seg, Norfair Tracking) to detect, track, and estimate the weight of shrimp in aquaculture environments.

## 🌟 Key Features
- **Bottom-Up Acquisition Strategy**: We utilize a unique camera angle positioned beneath the tank to capture high-contrast ventral profiles. This setup prioritizes shrimp on the **tank floor (fixed depth)**, effectively filtering out floating subjects and surface disturbances. This geometric consistency is critical for minimizing scale deviation and ensuring high-precision biometric estimation.
- **Real-time Weight Estimation**: Instantly calculates individual shrimp weight (g) from video feeds.
- **Water Clarify Model**: Automatically detects water turbidity to prevent processing unusable footage, significantly **saving power and storage**.
- **Robust Tracking**: Utilizes **Norfair + SIFT ReID** to track shrimp identities even under occlusion.
- **High Precision**: Combines **Oriented Bounding Boxes (OBB)** for detection(Capture the shrimp that is closest to the camera to make sure the precision of length estimation) and **Segmentation** for precise width measurement.
- **Underwater Correction**: Includes a dedicated module for correcting optical distortion caused by water.
- **Automated Logging**: Exports biological data (Length, Width, Weight) to CSV for statistical analysis.

---

## 📸 Demo

|             Detection & Tracking & Length, Width, and Weight Estimation              |
|:------------------------------------------------------------------------------------:|
| <img src=".\shrimp_OBB\data\demo\demo.gif" width="600" alt="Weight Estimation Demo"> |

---

## ⚡ Energy & Storage Optimization

This system is designed for real-world deployment where resources are limited. We implemented two key mechanisms to optimize efficiency:

### 1. Water Quality Gating (Power Saving)
Before running the computationally expensive detection models (YOLOv5/YOLOv8), the system performs a lightweight check using `logistic_water.py`.
- **Logic**: If the water is classified as **"Turbid"**, the system immediately aborts processing for that video.
- **Benefit**: Prevents wasting GPU/CPU cycles on low-visibility footage, extending battery life for edge devices.

### 2. Intelligent Data Storage (Storage Saving)
- **Turbid Video Handling**: Unusable turbid videos are automatically moved to a separate folder and excluded from the analysis pipeline.
- **Shrimp-Only Export**: The system generates a condensed video containing **only the frames where shrimp are detected**, filtering out empty frames to drastically reduce file size.

---

## 📂 Project Workflow & Structure

The project is organized into three main modules. For the best results, please follow the implementation order below:

```text
STEP 1: image_calibration/
      │
      ▼ (Output: Undistorted Images)
      │
STEP 2: Shrimp_size_weight_analysis/
      │
      ▼ (Output: Trained Regression Models .pkl)
      │
STEP 3: shrimp_OBB/
      │
      ▼ (Action: Real-time Inference)
      │
FINAL OUTPUT: Annotated Video & CSV Logs
```

1. Image Calibration (image_calibration/)
Start here. Underwater environments introduce optical distortion. This module calculates the camera matrix and distortion coefficients to correct images before analysis.

- Goal: Minimize geometric error to ensure pixel measurements represent true physical size.

- Key File: undistort_all_img.py

2. Research & Modeling (Shrimp_size_weight_analysis/)
The Scientific Foundation. This module analyzes the biological relationship between shrimp dimensions (Length/Width) and Weight.

- Components:

    - linear_relationship: Validates the linearity between pixel and physical measurements.

    - Shrimp_Size_Estimation: Uses LOOCV to train models for converting Pixels → mm.

    - shrimp_weight_regression: Trains the final regression models (Linear/Polynomial) to convert mm → grams.

- Output: Generates the .pkl models used in the real-time application.

3. Real-time Application (shrimp_OBB/)
The Deployment. This is the main execution engine that runs on video streams.

- Methodology:

    - Detection: YOLOv5-OBB (Oriented Bounding Box).

    - Refinement: Crop & Rotate -> YOLOv8-Seg (Precise Width).

    - Tracking: Norfair + SIFT ReID.

    - Estimation: Applies the regression models from Step 2 (using image-based pred_length and pred_width).
    
    - Water clarify: Features logistic_water.py to filter out turbid water, saving computational resources and storage.

---

## 🔗 Acknowledgments & References
Our codebase is built upon the YOLOv5-OBB repository and the Norfair tracking library.

The core integration is implemented within `shrimp_OBB/detect_norfair_optimize.py`, where we added Norfair tracking with SIFT Re-Identification (ReID) and embedded the regression logic to predict the actual length, width, and weight of the shrimp in `shrimp_OBB/utils/plots.py`.

Researchers interested in the underlying methodology can utilize or modify to the code provided in this repository. 

Full research paper will be updated once it is accepted.

---

## Author
```text
Bing Chian 
email: cucuchen105@gmail.com
```
