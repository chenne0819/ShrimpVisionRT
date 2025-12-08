# Real-time Shrimp OBB Detection, Tracking, and Weight Estimation System

This project is a comprehensive solution for **real-time shrimp monitoring**. It integrates **YOLOv5-OBB** for oriented detection, **Norfair** combined with **SIFT** for robust multi-object tracking, and **YOLOv8-Seg** for precise width measurement. 

The core capability of this system is to **estimate the weight of shrimp in real-time**. By fusing visual data with statistical regression models, it provides immediate weight feedback (in grams) for aquaculture analysis.  

---

## 🔗 Acknowledgments & References
Our codebase is built upon the [YOLOv5-OBB](https://github.com/hukaixuan19970627/yolov5_obb) repository and the **Norfair** tracking library. The core integration is implemented within `detect_norfair_optimize.py` (modified from the original `detect.py`), where we added **Norfair tracking with SIFT Re-Identification (ReID)** and './utils/plots.py' embedded the regression logic to predict the actual length, width, and weight of the shrimp. 

Researchers interested in the underlying methodology can refer to the code provided in this repository.   

**We will update this section with a link to our full research paper once it is accepted.**

---

## 🏗️ System Architecture & Workflow

The pipeline processes video feeds through the following stages:

1.  **Water Quality Verification**:
    -   The system first checks the water turbidity using `logistic_water.py`.
    -   If the water is classified as "turbid", the video is **skipped and moved to a `turbid_water` folder to ensure data integrity**.

2.  **Oriented Detection (YOLOv5-OBB)**:
    -   Detects shrimp using Oriented Bounding Boxes (OBB) to accurately capture their shape regardless of rotation.

3.  **Precise Feature Extraction**:
    -   **Crop & Rotate**: The detected OBB region is cropped and mathematically rotated to a horizontal alignment.
    -   **Width Estimation**: The aligned image is passed to a **YOLOv8 Segmentation** model (`seg_shrimp`) to measure the precise body width.

4.  **Robust Tracking (Norfair + SIFT)**:
    -   We utilize a hybrid tracking approach combining **Norfair** (Kalman Filter) with **SIFT (Scale-Invariant Feature Transform)**.
    -   **SIFT** features are extracted to calculate `embedding_distance`, allowing the system to re-identify (ReID) shrimp even if they move erratically or are temporarily occluded.

5.  **Real-time Weight Estimation**:
    -   **This is the final and most critical step.**
    -   The system converts pixel dimensions to physical metrics (mm) and immediately calculates the weight (g) using pre-trained regression models loaded in `utils/plots.py`.
    -   Supports Linear, Polynomial, and Multi-feature (Length+Width) regression models.

---

## 📂 Folder Structure

We provide sample videos (one clear, one turbid) in `shrimp_video/` for testing the water quality filter and detection capabilities.

```text
Shrimp_OBB
 ├── shrimp_video/                              # Sample Videos
 │      ├── 2024-01-01-00_11_15.mp4             # Clear water sample (For execution)
 │      └── 2024-01-08-06_53_42.mp4             # Turbid water sample (For testing filter)
 ├── runs/train/exp_OBB/weights/best.pt"        # Shrimp OBB model
 ├── Model/
 │    ├── final_linear_model_length.pkl         # Length Regression Model
 │    ├── logistic_regression_model.pth         # Water Quality Model
 │    ├── polynomial_regression_model_degree3.pkl # Weight Regression Model
 │    ├── final_linear_model_width.pkl          # Width Regression Model
 │    ├── multi_feature_model.pkl               # Combined L+W Weight Model
 │    └── seg_shrimp/
 │           └── weights/
 │                  └── best.pt                 # YOLOv8 Weights for Width Segmentation
 │
 ├── norfair/                                   # Norfair tracking library
 ├── utils/
 │    └── plots.py                              # Handles model loading & metric drawing
 ├── detect_norfair_optimize.py                 # [MAIN] Detection, Tracking & Logic Script
 ├── detect_norfair_optimize_elec_time.py       # Performance/Time optimization variant
 └── logistic_water.py                          # Water quality classification script
                            # Dependencies                       # Dependencies
```
---
## 🛠️ Requirements
To run the code, please install the necessary dependencies. You can create a requirements.txt file with the content below or install them directly.

Required Packages:

```bash
    pip install numpy opencv-python torch torchvision pandas scikit-learn scipy ultralytics norfair matplotlib seaborn tqdm pyyaml
```

---

## 🚀 Usage
This project have provided all necessary pre-trained models (YOLOv5-OBB, YOLOv8-Seg, and Regression models). You can run the inference directly on the provided sample video.

1. Enter the Directory
```bash
    cd Shrimp_OBB
```

2. Execute Detection on Clear Video/Turbid Video
Run the following command to test the system on the provided clear video sample:
```Bash
    python detect_norfair_optimize.py --weights ./runs/train/exp_OBB/weights/best.pt --source ./video/sample.mp4    
```

Key Arguments  

- --weights: Path to the YOLOv5-OBB model weights.

- --source: Input source. Can be a file path, img/video path, or 0 for webcam.

- --conf-thres: Confidence threshold (default 0.6).

- --track-points: Tracking method. Options: bbox (default).

(Training Note: If you wish to learn how to train the YOLOv5-OBB model yourself, please refer to the original repository and tutorial: https://github.com/hukaixuan19970627/yolov5_obb)

---

## 📊 Outputs
The system generates structured data and visual results:

1. Annotated Video:

    - Saved in runs/detect/expX/.
    
    - Displays the OBB, ID, and real-time estimated weight (g).

2. Shrimp-Only Video:

    - Saved in shrimp_only_frames/.
    
    - A condensed video file containing only the frames where shrimp were detected, optimizing storage.

3. Data Logs (CSV):

    - Saved in csv_data/.
    
    - Filename matches the video name.
    
    - Columns: Start Time, End Time, Max Length (mm), Max Width (mm), Max Weight (g).

4. Turbid Water Handling:

    - Videos with poor visibility are automatically moved to the turbid_water/ directory and excluded from analysis.

---

## 📝 Technical Notes
- Regression Logic: The mapping from pixels to grams is handled inside utils/plots.py. Ensure the path to the .pkl files in the Model/ directory is correct relative to the execution path.

- SIFT ReID: The embedding_distance function in the main script calculates the similarity between the current detection and past tracks using SIFT feature matching.

- YOLOv8 Integration: The global model global_yolo_model is initialized at the start of the script to handle width detection batches efficiently.
---

## ⚡ Energy & Performance Benchmarking Script

The `detect_norfair_optimize_elec_time.py` script is a specialized variant of the main detector. It is designed to profile the system's efficiency by logging **execution time**, **GPU power consumption**, and **resource utilization** for every frame.

## 🚀 Usage

```bash
   python detect_norfair_optimize_elec_time.py --weights ./runs/train/exp_OBB/weights/best.pt --source ./shrimp_video/2024-01-01-00_11_15.mp4
```
## 📊 System Outputs
When running this script, you will see detailed performance metrics in two places: the Console Logs and the Generated CSV Reports.

1. Real-time Console Output (Per Frame)
For every processed frame, the system prints a performance breakdown:

    - Time Breakdown: Shows exactly how many milliseconds each stage (Detection, Width Seg, Tracking) took.

    - Power Readings: Displays real-time GPU power draw (in Watts) if supported by the hardware.

2. Final Performance Summary
At the end of the video execution, a comprehensive summary is displayed

3. Generated Data Files
The script saves two additional CSV files in the runs/detect/expX/ directory for analysis:

    - {video_name}_performance.csv:
       - Raw data log containing row-by-row metrics for every single frame. 
       - Columns: frame, yolov5_time, gpu_power, cpu_util, yolov5_gpu_power, etc. 
       - Useful for creating time-series plots of power usage.

    - {video_name}_performance_summary.csv:

       - A concise file containing the calculated averages and totals (Mean Time, Total Wh, etc.).
    
       - Useful for comparing different models or hardware setups.

## 🛠️ Hardware Requirements for Power Monitoring
- NVIDIA GPU: Required for power monitoring.

- Python Library: pip install nvidia-ml-py3 (or pynvml).

- Note: If a GPU is not detected or does not support power reporting, the script will automatically disable power logs and only report execution time.