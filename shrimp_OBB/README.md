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
shrimp_OBB
 ├── shrimp_video/                              # Sample Videos
 │      ├── 2024-01-01-00_11_15.mp4             # Clear water sample (For execution)
 │      └── 2024-01-08-06_53_42.mp4             # Turbid water sample (For testing filter)
 ├── runs/train/exp_OBB/weights/best.pt         # Shrimp OBB model
 ├── data/bottom_shrimp.example.yaml           # Dataset configuration example (not study data)
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
The repository includes YOLOv5-OBB and YOLOv8-Seg weights and sample videos. Full weight estimation additionally requires the following files in `shrimp_OBB/Model/`; these files are currently **not included** in the repository:

- `final_linear_model_length.pkl`
- `final_linear_model_width.pkl`
- `polynomial_regression_model_degree3.pkl`
- `multi_feature_model.pkl`
- `logistic_regression_model.pth`

Obtain the matching trained files before running the full pipeline. Fixing a file path does not supply a missing model. For regression training, the referenced measurement spreadsheets, including `shrimp_len_wid_wei.xlsx`, must also be supplied separately.

1. Enter the Directory
```bash
    cd shrimp_OBB
```

2. Execute Detection on Clear Video/Turbid Video
Run the following command to test the system on the provided clear video sample:
```Bash
    python detect_norfair_optimize.py
```

Key Arguments  

- --weights: Path to the YOLOv5-OBB model weights.

- --source: Input source. The default is `shrimp_video/2024-01-01-00_11_15.mp4` under `shrimp_OBB/`. Explicit sources are passed through to the existing input loader.

- --conf-thres: Confidence threshold (default 0.6).

- --track-points: Tracking method. Options: bbox (default).

(Training Note: If you wish to learn how to train the YOLOv5-OBB model yourself, please refer to the original repository and tutorial: https://github.com/hukaixuan19970627/yolov5_obb)

### Portable inference paths

For `detect.py`, `detect_norfair_optimize.py`, and `detect_norfair_optimize_elec_time.py`, built-in model paths, the default sample video, and default output directories are resolved from the location of `shrimp_OBB/`, using `Path(__file__).resolve()`. Moving the repository does not require editing these paths. After supplying the missing models, you can also run from the repository root:

```bash
python shrimp_OBB/detect_norfair_optimize.py
```

Paths explicitly supplied through `--weights`, `--source`, and `--project` retain their normal behavior: relative filesystem paths are relative to your current working directory, while absolute paths are used as supplied. Camera IDs, URLs, and globs are passed through unchanged. For example, from the repository root:

```bash
python shrimp_OBB/detect_norfair_optimize.py --source shrimp_OBB/shrimp_video/2024-01-01-00_11_15.mp4 --project results
```

The `--project` option controls the annotated inference results. Auxiliary CSV, turbid-video, and shrimp-only-video outputs remain under `shrimp_OBB/` as documented below. The research/calibration scripts and upstream DOTA helper examples have their own path settings; this inference path convention does not change those scripts.

### What is `bottom_shrimp.yaml`?

`train.py` refers to `data/bottom_shrimp.yaml`, but the original dataset configuration and DOTA annotations are not included in this repository. A YAML dataset configuration is a small text file describing the dataset root (`path`), image splits (`train`, `val`, optionally `test`), class count (`nc`), and class names (`names`). It contains neither the images nor their bounding-box annotations, and is not a trained model.

`data/bottom_shrimp.example.yaml` is a new single-class **example**, not the original study configuration. Copy it to `data/bottom_shrimp.yaml` and adapt the paths and class names to your own dataset. The existing training loader resolves its dataset `path` relative to the **working directory**, so run training from `shrimp_OBB/` when using this example and pass `--data data/bottom_shrimp.yaml`. External datasets may also use an absolute `path` of your choice.

The example expects this layout under `shrimp_OBB/`:

```text
dataset/bottom_shrimp/
├── train/
│   ├── images/       # e.g. shrimp_001.jpg
│   └── labelTxt/     # matching shrimp_001.txt
└── val/
    ├── images/
    └── labelTxt/
```

This repository's loader maps each `images/<name>.<extension>` to `labelTxt/<name>.txt`. Each object annotation has ten fields in DOTA format:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

The four corners use pixel coordinates; `class_name` must match an entry in `names`. For example, a synthetic annotation could be `10 20 50 20 50 40 10 40 shrimp 0`. Sharing the YAML alone is not sufficient to reproduce training: the matching images, annotations, and original split/settings are also needed.

### Train an OBB detector on your own data

Use the original [YOLOv5-OBB installation guide](https://github.com/hukaixuan19970627/yolov5_obb/blob/master/docs/install.md) and [training instructions](https://github.com/hukaixuan19970627/yolov5_obb/blob/master/docs/GetStart.md) to prepare a matching PyTorch/CUDA environment. In particular, rotated NMS requires a compiled extension. This repository contains its Python wrapper and build script, but not the C++/CUDA sources; obtain `utils/nms_rotated/src/` from the upstream project and follow its build instructions for your platform before running training. Python package installation alone does not build that extension.

From the repository root, enter the detector directory and install its Python dependencies:

```bash
cd shrimp_OBB
python -m pip install -r requirements.txt
```

Copy the example to the exact filename used by the training command:

```bash
# Linux/macOS
cp data/bottom_shrimp.example.yaml data/bottom_shrimp.yaml
```

```powershell
# Windows PowerShell
Copy-Item data/bottom_shrimp.example.yaml data/bottom_shrimp.yaml
```

Edit the new `data/bottom_shrimp.yaml`: set `path`, `train`, and `val` to your own image directories, and set `nc` and `names` to match your DOTA annotations. Prepare the matching `labelTxt/` directories shown above. The current training defaults treat the dataset as a single shrimp class.

The repository now includes `data/hyps/obb/hyp.finetune_dota.yaml` from upstream commit `b00c3f245e50e7a80460a1b949d22ea7dfeb27a0`. These are **upstream starting hyperparameters**, not a recovered configuration from our shrimp study. Adapt them to your data and camera setup.

For the single-class example, run from `shrimp_OBB/`:

```bash
python train.py --data data/bottom_shrimp.yaml --hyp data/hyps/obb/hyp.finetune_dota.yaml --weights weight/yolov5n.pt --epochs 100 --batch-size 8 --imgsz 864 --device 0 --name shrimp_custom
```

The epoch count here is illustrative. Training outputs are saved under `runs/train/shrimp_custom/` (with an incremented suffix if that name already exists). This trains the OBB detector; segmentation, pixel-to-size calibration, weight regression, and water clarity classification are separate models/tasks.

**Initial weights:** if `weight/yolov5n.pt` is absent, the downloader attempts to retrieve it from the official Ultralytics **v6.0** release and creates the `weight/` directory. The release is pinned to match this older codebase instead of following the latest release. These are COCO pretrained weights used to initialize training, not our trained shrimp detector.

If automatic downloading fails, download `yolov5n.pt` from the [official v6.0 release](https://github.com/ultralytics/yolov5/releases/tag/v6.0) and save it as `shrimp_OBB/weight/yolov5n.pt`. From `shrimp_OBB/`, the equivalent commands are:

```bash
# Linux/macOS
mkdir -p weight
curl -L --fail https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt -o weight/yolov5n.pt
```

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force weight
Invoke-WebRequest https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt -OutFile weight/yolov5n.pt
```

You can also pass the path to a compatible checkpoint with `--weights`. Use trusted checkpoints: legacy YOLO `.pt` files contain serialized Python model objects. Their loader explicitly supports this format, including PyTorch 2.6's changed loading default.

**No biometric checkpoints are needed for OBB training or validation.** Generic polygon/class annotations are separate from shrimp size/weight annotations. The regression models are loaded and cached only when the tracking/measurement pipeline requests a prediction. If a required measurement model is absent, that inference call reports the missing file rather than substituting an estimate.

Component regression tests are in `tests/test_training_setup.py` (run `python -m pytest tests/test_training_setup.py` from the repository root after installing pytest and the dependencies). They cover model-free plotting, cached measurement inference, dataset parsing, weight download routing, and a synthetic OBB optimization step. These tests use a fail-fast stub for the compiled NMS extension; they do not validate rotated NMS, a complete training epoch, or reproduction of the paper's results.

---

## 📊 Outputs
The system generates structured data and visual results. The default locations below are relative to `shrimp_OBB/`:

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
- Regression Logic: The mapping from pixels to grams is handled inside `utils/plots.py`. Place the required `.pkl` files in `shrimp_OBB/Model/`; this location is independent of the working directory. The water classifier likewise loads its `.pth` file from `shrimp_OBB/Model/`.

- SIFT ReID: The embedding_distance function in the main script calculates the similarity between the current detection and past tracks using SIFT feature matching.

- YOLOv8 Integration: The global model global_yolo_model is initialized at the start of the script to handle width detection batches efficiently.
---

## ⚡ Energy & Performance Benchmarking Script

The `detect_norfair_optimize_elec_time.py` script is a specialized variant of the main detector. It is designed to profile the system's efficiency by logging **execution time**, **GPU power consumption**, and **resource utilization** for every frame.

## 🚀 Usage

```bash
   # Run from shrimp_OBB/ after supplying the missing models:
   python detect_norfair_optimize_elec_time.py
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
