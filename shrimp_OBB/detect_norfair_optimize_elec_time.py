# YOLOv5 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python detect_norfair_optimize_elec_time.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""
import time
import argparse
import os
import sys
from pathlib import Path
import csv
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import shutil
import psutil
import pandas as pd
from collections import defaultdict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression_obb, print_args, scale_polys, strip_optimizer)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import rbox2poly
from norfair.filter import OptimizedKalmanFilterFactory

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import List, Optional, Union
import norfair
from norfair import Paths, Video
from norfair.tracker import Detection, Tracker
from norfair import draw_points, draw_tracked_objects, get_cutout

DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000

from ultralytics import YOLO
from logistic_water import predict_image_class

# GPU 
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    print("GPU monitoring enabled")
    # GPU 
    try:
        test_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(test_handle)
        print(f"Detected GPU: {gpu_name}")
        # 
        test_power = pynvml.nvmlDeviceGetPowerUsage(test_handle)
        print(f"Test power reading: {test_power/1000.0:.2f}W")
    except Exception as e:
        print(f"GPU test failed: {e}")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"Warning: pynvml not available. GPU power monitoring disabled.")
    print(f"Error: {e}")
    print("Install with: pip install nvidia-ml-py3")

# Load YOLOv8 model globally for width detection
global_yolo_model = YOLO('./Model/seg_shrimp/weights/best.pt')

@torch.no_grad()
def width_detect_batch(crop_imgs, model=global_yolo_model):
    """
    Batch process cropped images to detect width using YOLOv8
    """
    if not crop_imgs:  # If empty list, return empty
        return []
    results = model(crop_imgs)  # Batch inference
    widths = []
    for i, r in enumerate(results):
        boxes = r.boxes.xywh
        confidence = r.boxes.conf
        if boxes.tolist():
            max_conf_idx = confidence.argmax()
            max_conf_box = boxes[max_conf_idx].tolist()
            max_conf_value = confidence[max_conf_idx].item()
            w, h = max_conf_box[2], max_conf_box[3]
            width = min(w, h)
            widths.append(width)
        else:
            widths.append(None)
    return widths

def crop_rotate_img(img, points):
    """
    Crop and rotate a region from an image based on given points.
    """
    points = np.array(points, dtype=np.int32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


def embedding_distance(matched_not_init_trackers, unmatched_trackers):
    """
    Calculate the distance between two tracked objects using SIFT feature embeddings.
    """
    snd_embedding = unmatched_trackers.last_detection.embedding
    
    if snd_embedding is None:
        for detection in reversed(unmatched_trackers.past_detections):
            if detection.embedding is not None:
                snd_embedding = detection.embedding
                break
        else:
            return 1
    
    for detection_fst in matched_not_init_trackers.past_detections:
        if detection_fst.embedding is None:
            continue
        
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(snd_embedding, detection_fst.embedding)
        matches = sorted(matches, key=lambda x: x.distance)
        match_rate = (2 * len(matches)) / (len(detection_fst.embedding) + len(snd_embedding))
        distance = 1 - match_rate
        
        if distance < 0.3:
            return distance
    
    return 1

def sift_feature(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

def calculate_embedding(cut: np.ndarray) -> Optional[np.ndarray]:
    if cut.shape[0] > 0 and cut.shape[1] > 0:
        return sift_feature(cut)
    else:
        return None

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',
        source=ROOT / 'data/images',
        imgsz=(864, 864),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=True,
        save_conf=True,
        save_crop=True,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=True,
        half=False,
        dnn=False,
        track_points='bbox',
        ):
    start_time = time.time()
    
    # 
    performance_data = {
        'frame': [],
        'yolov5_time': [],
        'yolov8_time': [],
        'tracking_time': [],
        'total_frame_time': [],
        'gpu_power': [],
        'gpu_util': [],
        'cpu_util': [],
        'memory_util': [],
        'yolov5_gpu_power': [],  # YOLOv5  GPU 
        'yolov8_gpu_power': [],  # YOLOv8  GPU 
    }
    
    # GPU 
    if GPU_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            print(f"GPU monitoring initialized: {gpu_name}")
            
            # 
            try:
                test_power = pynvml.nvmlDeviceGetPowerUsage(handle)
                print(f"GPU power reading available: {test_power/1000.0:.2f}W")
                power_available = True
            except pynvml.NVMLError_NotSupported:
                print("Warning: GPU power reading not supported on this device")
                power_available = False
            except Exception as e:
                print(f"Warning: GPU power reading error: {e}")
                power_available = False
                
        except Exception as e:
            handle = None
            power_available = False
            print(f"Warning: Could not get GPU handle: {e}")
    else:
        handle = None
        power_available = False
    
    # CPU/Memory 
    process = psutil.Process()
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Total frames
    source_cap = cv2.VideoCapture(source)
    total_frames = int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_cap.release()

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)
    dt, seen = [0.0, 0.0, 0.0], 0
    distance_function = "iou" if track_points == "bbox" else "euclidean"
    distance_threshold = DISTANCE_THRESHOLD_BBOX if track_points == "bbox" else DISTANCE_THRESHOLD_CENTROID
    tracker = Tracker(
        initialization_delay=1,
        distance_function=distance_function,
        hit_counter_max=15,
        filter_factory=OptimizedKalmanFilterFactory(),
        distance_threshold=distance_threshold,
        past_detections_length=5,
        reid_distance_function=embedding_distance,
        reid_distance_threshold=0.5,
        reid_hit_counter_max=15,
    )
    save_csv = {}
    store_path = None
    empty_det = 0
    
    # For storing frames with shrimp detections
    frames_with_shrimp = []
    frame_count = 0
    water_quality_checked = False

    for path, im, im0s, vid_cap, s in dataset:
        #  frame 
        frame_start_time = time.time()
        
        store_path = path
        
        # Water quality check on first frame only
        if not water_quality_checked:
            water_quality, confidence = predict_image_class(im0s)
            print(f'Predicted Class: {water_quality}')
            
            if water_quality == 'turbid':
                turbid_folder = "./turbid_water"
                # curr_dir = os.getcwd()
                turbid_folder_path = os.path.join(f'./{turbid_folder}')
                if not os.path.exists(turbid_folder_path):
                    os.makedirs(turbid_folder_path)
                
                source_path = Path(source)
                destination_path = os.path.join(turbid_folder_path, source_path.name)
                shutil.copy(str(source_path), str(destination_path))
                print(f"Water quality is turbid. Copied video to {destination_path}")
                print("Skipping processing for this video.")
                return
            
            water_quality_checked = True
        
        # YOLOv5 
        t1 = time_sync()
        
        # YOLOv5  GPU 
        yolov5_power_start = 0
        if handle and power_available:
            try:
                yolov5_power_start = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except:
                pass
        
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det)
        t4 = time_sync()
        dt[2] += t4 - t3
        
        # YOLOv5  GPU
        yolov5_power_end = 0
        if handle and power_available:
            try:
                yolov5_power_end = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except:
                pass
        
        # YOLOv5
        yolov5_power = (yolov5_power_start + yolov5_power_end) / 2.0
        
        # YOLOv5 
        yolov5_time = t4 - t1

        detections = []
        crop_imgs = []
        
        for i, det in enumerate(pred):
            pred_poly = rbox2poly(det[:, :5])
            count = len(pred_poly)
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            store_path = path.split('/')[-1].split('.')[0]
            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                pred_poly = scale_polys(im.shape[2:], pred_poly, (450, 800))
                det = torch.cat((pred_poly, det[:, -2:]), dim=1)

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                img = im0.copy()
                num_shown = 1
                for j, (*poly, conf, cls) in enumerate(reversed(det)):
                    points = [(poly[num].item(), poly[num + 1].item()) for num in range(0, 8, 2)]
                    min_x = min(point[0] for point in points)
                    max_x = max(point[0] for point in points)
                    min_y = min(point[1] for point in points)
                    max_y = max(point[1] for point in points)
                    points_array = np.array([[min_x, min_y], [max_x, max_y]])
                    cropedImg = crop_rotate_img(img, points)
                    crop_imgs.append(cropedImg)
                    embedding = calculate_embedding(cropedImg)
                    detection = Detection(points=points_array, embedding=embedding, label=int(cls), frame=frame)
                    detections.append(detection)

                    if save_txt:
                        poly = poly.tolist()
                        line = (cls, *poly, conf) if save_conf else (cls, *poly)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # YOLOv8 
                t5 = time_sync()
                
                # YOLOv8  GPU 
                yolov8_power_start = 0
                if handle and power_available:
                    try:
                        yolov8_power_start = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except:
                        pass
                
                yolov8_wids = width_detect_batch(crop_imgs)
                t6 = time_sync()
                
                # YOLOv8 GPU 
                yolov8_power_end = 0
                if handle and power_available:
                    try:
                        yolov8_power_end = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except:
                        pass
                
                #  YOLOv8 
                yolov8_power = (yolov8_power_start + yolov8_power_end) / 2.0
                
                yolov8_time = t6 - t5
                
                # 
                t7 = time_sync()
                
                for j, (*poly, conf, cls) in enumerate(reversed(det)):
                    yolov8_wid = yolov8_wids[j]
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.poly_label(poly=poly, width=yolov8_wid, label=label, color=colors(c, True))
                    shr_weight = annotator.weight
                    shr_len = annotator.length
                    shr_wid = annotator.width
                    detections[j].length = shr_len
                    detections[j].weight = shr_weight
                    detections[j].width = shr_wid
                    width_str = f"{yolov8_wid:.2f}" if yolov8_wid is not None else "N/A"
                    print(f'[Frame {frame}] Detection {j}: cls={names[c]}, conf={conf:.2f}, width={width_str}, length={shr_len:.2f}, weight={shr_weight:.2f}')

                tracked_objects = tracker.update(detections=detections)
                
                for obj in tracked_objects:
                    csv_frame = obj.start_frame
                    csv_sec = obj.start_frame // 15
                    csv_min, csv_sec = divmod(csv_sec, 60)
                    start_frame = f'{str(csv_min).zfill(2)}:{str(csv_sec).zfill(2)}_{csv_frame}'
                    csv_frame = obj.end_frame
                    csv_sec = obj.end_frame // 15
                    csv_min, csv_sec = divmod(csv_sec, 60)
                    end_frame = f'{str(csv_min).zfill(2)}:{str(csv_sec).zfill(2)}_{csv_frame}'
                    print(f'[Frame {frame}] Tracked Object ID={obj.id}: start={start_frame}, end={end_frame}, max_length={obj.max_length:.2f}, max_weight={obj.max_weight:.2f}')
                    if obj.id in save_csv:
                        save_csv[obj.id] = [start_frame, end_frame, obj.max_length, obj.max_width, obj.max_weight]
                    else:
                        save_csv[obj.id] = [start_frame, end_frame, obj.max_length, obj.max_width, obj.max_weight]
                
                # plot tracking bbox
                if track_points == "bbox":
                    # norfair.draw_boxes(im0, detections)  
                    norfair.draw_tracked_boxes(im0, tracked_objects, draw_labels=True, id_size=1)  # ID
                    # pass
                
                t8 = time_sync()
                tracking_time = t8 - t7
            else:
                empty_det += 1
                yolov8_time = 0
                tracking_time = 0
                yolov8_power = 0

            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            im0 = annotator.result()
            
            # Store frame if it has detections
            if len(det):
                frames_with_shrimp.append((frame_count, im0.copy()))
            
            # total frame time
            frame_end_time = time.time()
            total_frame_time = frame_end_time - frame_start_time
            
            # 
            cpu_percent = process.cpu_percent()
            memory_percent = process.memory_percent()
            
            gpu_power = 0
            gpu_util = 0
            gpu_memory_used = 0
            gpu_memory_total = 0
            
            if handle and power_available:
                try:
                    # GPU W
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    gpu_power = power_mw / 1000.0
                except pynvml.NVMLError_NotSupported:
                    gpu_power = -1  # 
                except Exception as e:
                    print(f"Error reading GPU power: {e}")
                    gpu_power = 0
            
            if handle:
                try:
                    # GPU 
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    
                    # GPU 
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used = mem_info.used / (1024**3)  # GB
                    gpu_memory_total = mem_info.total / (1024**3)
                except Exception as e:
                    print(f"Error reading GPU utilization: {e}")
            
            # 
            performance_data['frame'].append(frame_count)
            performance_data['yolov5_time'].append(yolov5_time * 1000)  # ms
            performance_data['yolov8_time'].append(yolov8_time * 1000)
            performance_data['tracking_time'].append(tracking_time * 1000)
            performance_data['total_frame_time'].append(total_frame_time * 1000)
            performance_data['gpu_power'].append(gpu_power)
            performance_data['gpu_util'].append(gpu_util)
            performance_data['cpu_util'].append(cpu_percent)
            performance_data['memory_util'].append(memory_percent)
            performance_data['yolov5_gpu_power'].append(yolov5_power)  #  YOLOv5 
            performance_data['yolov8_gpu_power'].append(yolov8_power)  # YOLOv8 
            
            # 
            print(f'\n[Frame {frame_count}] Performance:')
            print(f'  YOLOv5: {yolov5_time*1000:.2f}ms | YOLOv8: {yolov8_time*1000:.2f}ms | '
                  f'Tracking: {tracking_time*1000:.2f}ms | Total: {total_frame_time*1000:.2f}ms')
            if handle:
                power_str = f"{gpu_power:.2f}W" if gpu_power >= 0 else "N/A (not supported)"
                print(f'  GPU Power: {power_str} | GPU Util: {gpu_util}% | '
                      f'GPU Mem: {gpu_memory_used:.2f}/{gpu_memory_total:.2f}GB')
                if yolov5_power > 0 or yolov8_power > 0:
                    print(f'  YOLOv5 Power: {yolov5_power:.2f}W | YOLOv8 Power: {yolov8_power:.2f}W')
                print(f'  CPU: {cpu_percent:.1f}% | System Memory: {memory_percent:.1f}%')
            
            frame_count += 1

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if vid_writer[-1]:
        vid_writer[-1].release()

    #  GPU 
    if GPU_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    t = tuple(x / seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)

    print(f'empty_det: {empty_det}')
    
    # Save video with only frames containing shrimp
    if frames_with_shrimp:
        shrimp_only_path = "./shrimp_only_frames"
        os.makedirs(shrimp_only_path, exist_ok=True)
        
        source_path = Path(source)
        store_path_name = source_path.stem
        shrimp_only_video = os.path.join(shrimp_only_path, f"{store_path_name}_shrimp_only.mp4")
        
        # Re-open video to get correct FPS
        temp_cap = cv2.VideoCapture(source)
        fps = temp_cap.get(cv2.CAP_PROP_FPS)
        original_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_cap.release()
        
        if fps <= 0 or fps > 120:
            fps = 30
        
        # Use original video dimensions
        w, h = original_width, original_height
        
        print(f"Creating shrimp_only video: {w}x{h} @ {fps}fps")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(shrimp_only_video, fourcc, fps, (w, h))
        
        if not out.isOpened():
            print(f"Error: Could not open VideoWriter for {shrimp_only_video}")
        else:
            written_count = 0
            failed_count = 0
            
            for frame_num, frame in frames_with_shrimp:
                # Ensure frame is correct size and type
                if frame.shape[1] != w or frame.shape[0] != h:
                    print(f"Warning: Frame {frame_num} has wrong size {frame.shape[1]}x{frame.shape[0]}, resizing to {w}x{h}")
                    frame = cv2.resize(frame, (w, h))
                
                # Ensure frame is uint8
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Ensure frame is BGR (3 channels)
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                try:
                    out.write(frame)
                    written_count += 1
                except:
                    failed_count += 1
                    print(f"Warning: Failed to write frame {frame_num}")
        
        out.release()
        print(f"\nShrimp-only video creation summary:")
        print(f"  Successfully written: {written_count} frames")
        print(f"  Failed: {failed_count} frames")
        print(f"  Total frames with shrimp: {len(frames_with_shrimp)} out of {frame_count}")
        
        # Verify file was created
        if os.path.exists(shrimp_only_video):
            file_size = os.path.getsize(shrimp_only_video)
            print(f"  Video saved to: {shrimp_only_video}")
            print(f"  File size: {file_size / (1024*1024):.2f} MB")
        else:
            print("  Warning: Video file was not created!")
    
    # Save CSV only if there are detections
    if save_csv:    
    
        os.makedirs("./csv_data", exist_ok=True)
        # CSV output path
        csv_file_path = f"./csv_data/{store_path_name}.csv"
            
        # Database Connection
        # import pymysql
        # connection = pymysql.connect(
        #     host='ip address',  # your ip address
        #     user='user name',   # user name
        #     password='password',  # db password
        #     database='db name',  # db name
        # )
    
        try:
            # Write CSV
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
              
                # SQL Insert Template                 
                # with connection.cursor() as cursor:
                #
                #     sql = """
                #         INSERT INTO newObjects 
                #         (VideoName, start_time, end_time, max_length, max_width, max_weight) 
                #         VALUES (%s, %s, %s, %s, %s, %s)
                #     """
                #     batch_data = []
                
                # Loop all saved items
                for values in save_csv.values():
    
                    writer.writerow(values)
    
                    # Prepare DB Insert
                    # video_name = f"{store_path_name}.mp4"
                    #
                    # start_time_frame = values[0]
                    # end_time_frame = values[1]
                    # max_length = values[2]
                    # max_width = values[3]
                    # max_weight = values[4]
                    #
                    # batch_data.append(
                    #     (video_name, start_time_frame, end_time_frame,
                    #      max_length, max_width, max_weight)
                    # )

                # Execute SQL Insert                
                # cursor.executemany(sql, batch_data)
                # connection.commit()
                # print("Data inserted successfully!")
                
        except Exception as e:
            print(f"Error writing CSV: {e}")
    
        # DB Close (Disabled) 
        # finally:
        #     connection.close()

    # 
    end_time = time.time()
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    
    # DataFrame
    df = pd.DataFrame(performance_data)
    
    # 
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\nTotal Runtime: {minutes} minutes {seconds} seconds ({total_seconds:.2f}s)")
    print(f"Total Frames Processed: {len(df)}")
    print(f"Average FPS: {len(df)/total_seconds:.2f}")
    
    print("\n--- Timing Statistics (ms) ---")
    print(f"YOLOv5 Detection:")
    print(f"  Mean: {df['yolov5_time'].mean():.2f} | Std: {df['yolov5_time'].std():.2f}")
    print(f"  Min: {df['yolov5_time'].min():.2f} | Max: {df['yolov5_time'].max():.2f}")
    
    print(f"\nYOLOv8 Width Detection:")
    print(f"  Mean: {df['yolov8_time'].mean():.2f} | Std: {df['yolov8_time'].std():.2f}")
    print(f"  Min: {df['yolov8_time'].min():.2f} | Max: {df['yolov8_time'].max():.2f}")
    
    print(f"\nTracking:")
    print(f"  Mean: {df['tracking_time'].mean():.2f} | Std: {df['tracking_time'].std():.2f}")
    print(f"  Min: {df['tracking_time'].min():.2f} | Max: {df['tracking_time'].max():.2f}")
    
    print(f"\nTotal Frame Time:")
    print(f"  Mean: {df['total_frame_time'].mean():.2f} | Std: {df['total_frame_time'].std():.2f}")
    print(f"  Min: {df['total_frame_time'].min():.2f} | Max: {df['total_frame_time'].max():.2f}")
    
    if handle:
        print("\n--- Resource Usage ---")
        
        # 
        valid_power_readings = df[df['gpu_power'] > 0]['gpu_power']
        valid_yolov5_power = df[df['yolov5_gpu_power'] > 0]['yolov5_gpu_power']
        valid_yolov8_power = df[df['yolov8_gpu_power'] > 0]['yolov8_gpu_power']
        
        if len(valid_power_readings) > 0:
            print(f"Overall GPU Power Consumption:")
            print(f"  Mean: {valid_power_readings.mean():.2f}W | Max: {valid_power_readings.max():.2f}W")
            total_energy = valid_power_readings.sum() * (total_seconds/len(df)) / 3600
            print(f"  Total Energy: {total_energy:.4f} Wh ({total_energy * 1000:.2f} mWh)")
            
            # YOLOv5 
            if len(valid_yolov5_power) > 0:
                yolov5_mean_power = valid_yolov5_power.mean()
                yolov5_total_time = df['yolov5_time'].sum() / 1000  # s
                yolov5_energy = yolov5_mean_power * yolov5_total_time / 3600  # Wh
                print(f"\nYOLOv5 OBB Power Consumption:")
                print(f"  Mean Power: {yolov5_mean_power:.2f}W | Max: {valid_yolov5_power.max():.2f}W")
                print(f"  Total Processing Time: {yolov5_total_time:.2f}s")
                print(f"  Total Energy: {yolov5_energy:.4f} Wh ({yolov5_energy * 1000:.2f} mWh)")
                print(f"  Energy per Frame: {(yolov5_energy / len(df)) * 1000:.4f} mWh")
            
            # YOLOv8 
            if len(valid_yolov8_power) > 0:
                yolov8_mean_power = valid_yolov8_power.mean()
                yolov8_total_time = df['yolov8_time'].sum() / 1000  # sec
                yolov8_energy = yolov8_mean_power * yolov8_total_time / 3600  # Wh
                print(f"\nYOLOv8 SEG Power Consumption:")
                print(f"  Mean Power: {yolov8_mean_power:.2f}W | Max: {valid_yolov8_power.max():.2f}W")
                print(f"  Total Processing Time: {yolov8_total_time:.2f}s")
                print(f"  Total Energy: {yolov8_energy:.4f} Wh ({yolov8_energy * 1000:.2f} mWh)")
                print(f"  Energy per Frame: {(yolov8_energy / len(df)) * 1000:.4f} mWh")
            
            # 
            if len(valid_yolov5_power) > 0 and len(valid_yolov8_power) > 0:
                total_model_energy = yolov5_energy + yolov8_energy
                print(f"\nEnergy Distribution:")
                print(f"  YOLOv5: {(yolov5_energy/total_model_energy)*100:.1f}%")
                print(f"  YOLOv8: {(yolov8_energy/total_model_energy)*100:.1f}%")
                print(f"  Other Processing: {((total_energy-total_model_energy)/total_energy)*100:.1f}%")
        else:
            print(f"GPU Power Consumption: Not supported on this device")
            print(f"  Note: Some GPUs (especially mobile/laptop GPUs) don't support power monitoring")
        
        print(f"\nGPU Utilization:")
        print(f"  Mean: {df['gpu_util'].mean():.1f}% | Max: {df['gpu_util'].max():.1f}%")
        
        print(f"\nCPU Utilization:")
        print(f"  Mean: {df['cpu_util'].mean():.1f}% | Max: {df['cpu_util'].max():.1f}%")
        
        print(f"\nMemory Usage:")
        print(f"  Mean: {df['memory_util'].mean():.1f}% | Max: {df['memory_util'].max():.1f}%")
    
    # CSV
    source_path = Path(source)
    perf_csv_path = save_dir / f"{source_path.stem}_performance.csv"
    df.to_csv(perf_csv_path, index=False)
    print(f"\nDetailed performance data saved to: {perf_csv_path}")
    
    # 
    summary_metrics = ['Total Runtime (s)', 'Total Frames', 'Average FPS',
                       'YOLOv5 Mean (ms)', 'YOLOv8 Mean (ms)', 'Tracking Mean (ms)',
                       'Total Frame Mean (ms)', 'GPU Power Mean (W)', 'GPU Power Max (W)',
                       'Total Energy (Wh)', 'GPU Util Mean (%)', 'CPU Util Mean (%)',
                       'Memory Util Mean (%)']
    
    summary_values = [total_seconds, len(df), len(df)/total_seconds,
                      df['yolov5_time'].mean(), df['yolov8_time'].mean(),
                      df['tracking_time'].mean(), df['total_frame_time'].mean(),
                      df['gpu_power'].mean() if handle else 0,
                      df['gpu_power'].max() if handle else 0,
                      df['gpu_power'].sum() * (total_seconds/len(df)) / 3600 if handle else 0,
                      df['gpu_util'].mean() if handle else 0,
                      df['cpu_util'].mean(), df['memory_util'].mean()]
    
    # YOLOv5  YOLOv8 
    if handle and len(df[df['yolov5_gpu_power'] > 0]) > 0:
        yolov5_mean_power = df[df['yolov5_gpu_power'] > 0]['yolov5_gpu_power'].mean()
        yolov5_total_time = df['yolov5_time'].sum() / 1000
        yolov5_energy = yolov5_mean_power * yolov5_total_time / 3600
        
        summary_metrics.extend(['YOLOv5 Power Mean (W)', 'YOLOv5 Total Energy (Wh)', 
                                'YOLOv5 Energy per Frame (mWh)'])
        summary_values.extend([yolov5_mean_power, yolov5_energy, 
                               (yolov5_energy / len(df)) * 1000])
    
    if handle and len(df[df['yolov8_gpu_power'] > 0]) > 0:
        yolov8_mean_power = df[df['yolov8_gpu_power'] > 0]['yolov8_gpu_power'].mean()
        yolov8_total_time = df['yolov8_time'].sum() / 1000
        yolov8_energy = yolov8_mean_power * yolov8_total_time / 3600
        
        summary_metrics.extend(['YOLOv8 Power Mean (W)', 'YOLOv8 Total Energy (Wh)', 
                                'YOLOv8 Energy per Frame (mWh)'])
        summary_values.extend([yolov8_mean_power, yolov8_energy, 
                               (yolov8_energy / len(df)) * 1000])
    
    summary_data = {
        'metric': summary_metrics,
        'value': summary_values
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = save_dir / f"{source_path.stem}_performance_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Performance summary saved to: {summary_csv_path}")
    print("="*80)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / './runs/train/exp_OBB/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=r'video.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[864, 864], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=50, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', default=False, help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', default=False, help='augmented inference')
    parser.add_argument('--visualize', nargs='*', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--track-points', type=str, default='bbox', help='Track points: centroid or bbox')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(FILE.stem, opt)
    return opt

def main():
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(parse_opt()))

if __name__ == "__main__":
    main()