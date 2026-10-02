# Portable Inference Paths Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the shrimp inference entry points locate their bundled assets independently of the working directory and document the missing training dataset configuration.

**Architecture:** Resolve built-in asset and output paths from each Python file using `pathlib.Path(__file__).resolve()`. Preserve explicit CLI paths, URLs, globs, and camera sources. Keep model contents and estimation logic unchanged.

**Tech Stack:** Python, pathlib, argparse, joblib, PyTorch, YOLOv5-OBB, YOLOv8-Seg.

---

### Task 1: Fix inference asset paths

**Files:** `shrimp_OBB/utils/plots.py`, `shrimp_OBB/logistic_water.py`, `shrimp_OBB/detect.py`, `shrimp_OBB/detect_norfair_optimize.py`, `shrimp_OBB/detect_norfair_optimize_elec_time.py`.

1. Use `Path(__file__).resolve().parents[1] / 'Model'` for the four regression models in `utils/plots.py`.
2. Use `Path(__file__).resolve().parent / 'Model'` for the water model.
3. Keep the detector `ROOT` anchored to the script directory; resolve segmentation weights from `ROOT / 'Model/seg_shrimp/weights/best.pt'`.
4. Align function and CLI defaults with `runs/train/exp_OBB/weights/best.pt` and `shrimp_video/2024-01-01-00_11_15.mp4` under `ROOT`.

### Task 2: Stabilize output paths and explain dataset configuration

**Files:** the three detectors above, `shrimp_OBB/README.md`, `shrimp_OBB/data/bottom_shrimp.example.yaml`.

1. Anchor default inference, CSV, turbid-video, and shrimp-only-video directories to `ROOT`.
2. Replace slash-based filename parsing with `Path(path).stem` for Windows and names containing dots.
3. Correct the README commands, explain default versus explicit CLI paths, and list the missing regression/water checkpoints.
4. Add a clearly labeled single-class dataset example, documenting the existing training loader's working-directory-relative dataset path and DOTA `labelTxt` format. Do not present it as the original study configuration or include unverified model/data files.

### Task 3: Validate and publish

1. Compile the changed Python source without importing optional runtime dependencies.
2. Exercise the actual path expressions, prediction functions with a recording model loader, and argument parsers from the repository root, detector directory, and an unrelated temporary directory. Check explicit relative paths, absolute paths, camera IDs, URLs, and globs remain unchanged.
3. Parse the YAML example and verify its image-to-label mapping against the repository's dataset loader.
4. Run `git diff --check`, review the scoped diff, commit the changes, push `main`, and verify the remote commit. Full video inference requires the absent checkpoints and a configured CV environment.
