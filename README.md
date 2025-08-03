# Tennis Vision System

A comprehensive computer vision system for tennis analysis combining synthetic data generation, multi-camera calibration, and real-time tennis ball detection using YOLOX.

## System Overview

This project implements a complete end-to-end tennis analysis pipeline featuring:
- **Synthetic Data Generation** for camera calibration
- **Multi-Camera Calibration** with bundle adjustment optimization
- **High-Performance Tennis Ball Detection** using optimized YOLOX

## Core Components

### 1. Synthetic Data Generation
- **Purpose**: Generate court and ball trajectories under different camera angles
- **Output**: Synthetic tennis scenes with ground truth annotations
```
Location: data\simulated
```

### 2. Multi-Camera Calibration System

Advanced calibration pipeline using tennis court geometry and bundle adjustment optimization.

#### Key Features
- **Multi-camera bundle adjustment** for simultaneous calibration of 3 cameras
- **Tennis court keypoint detection** using standardized court geometry
- **Performance visualization** with court overlay assessment

#### Court Geometry Standards
- **Court dimensions**: 12.0m × 6.0m (simplified regulation)
- **Service lines**: 3.0m from net center
- **Keypoint network**: 13 strategic court intersections
- **3D coordinate system**: Court-centered with z=0 at ground level

#### Calibration Performance (Distortion Ignored)
- **Reprojection accuracy**: Sub-pixel precision achieved
- **Multi-camera consistency**: Bundle adjustment ensures geometric coherence
- **Visual assessment**: Court overlay verification system

### 3. Tennis Ball Detection (YOLOX Optimized)

Tennis ball detection system using YOLOX architecture, optimized for real-time sports analysis and tracking applications.

#### Project Overview

This project implements tennis ball detection using a customized YOLOX model trained on multiple datasets to achieve robust performance across various court conditions, lighting scenarios, and ball sizes.

#### Performance Results

##### Original Baseline Results by using Primary Dataset and Default yolox_custom.py file 
Training on initial tennis ball dataset:

| Metric | Value | Description |
|--------|-------|-------------|
| **Inference Speed** | 3.50 ms | Average total inference time |
| **Forward Time** | 3.21 ms | Model forward pass |
| **NMS Time** | 0.29 ms | Non-maximum suppression |
| **AP@0.50:0.95** | 42.9% | Main COCO metric (strict IoU) |
| **AP@0.50** | 83.8% | Precision at IoU=0.5 threshold |
| **AP@0.75** | 40.2% | Precision at IoU=0.75 threshold |
| **Average Recall** | 50.0% | Maximum achievable recall |

**Analysis**: Good detection capability (83.8% AP@0.5) but limited recall ceiling (50%) and imprecise localization at strict IoU thresholds.


##### Results after Improvement Strategies Implemented
| Metric | Baseline | Improved | Change | Status |
|--------|----------|----------|---------|---------|
| **AP@0.50:0.95** | 42.9% | **63.7%** | **+20.8%** | **Target Exceeded** |
| **AP@0.50** | 83.8% | **90.2%** | **+6.4%** | **Excellent** |
| **AP@0.75** | 40.2% | **64.3%** | **+24.1%** | **Major Improvement** |
| **Average Recall** | 50.0% | **66.3%** | **+16.3%** | **Ceiling Raised** |

##### Size-Specific Performance Breakthrough

| Object Size | AP@0.50:0.95 | AR@0.50:0.95 | Improvement |
|-------------|--------------|--------------|-------------|
| **Small** | 45.0% | 51.4% | Solid baseline performance |
| **Medium** | **85.2%** | **86.7%** | **Excellent detection** |
| **Large** | **90.4%** | **92.0%** | **Outstanding performance** |

#### Improvement Strategies Implemented

##### 1) Multi-Scale Training Enhancement
- **Dynamic input sizes**: 448-832px range to handle varying tennis ball sizes
- **Multi-scale interval**: Scale changes every epoch for robust size adaptation
- **Target**: Address "all small objects" limitation in original results

##### 2) Advanced Data Augmentation Pipeline
```python
# Enhanced augmentation strategy
mosaic_prob = 0.8      # Increased mosaic for better context learning
mixup_prob = 0.3       # Added mixup for boundary precision
hsv_prob = 0.8         # Aggressive color augmentation
```
- **Mosaic augmentation**: Improved context understanding
- **MixUp technique**: Better boundary localization learning
- **Enhanced HSV**: Robust performance across lighting conditions

##### 3) Optimized Training Schedule
- **Extended training**: 100 → 200 epochs for better convergence
- **Longer warmup**: 5 → 10 epochs for training stability
- **No-augmentation fine-tuning**: Last 20 epochs for precision refinement
- **Learning rate optimization**: Improved scheduling with minimum ratio

##### 4) Detection Parameter Optimization
```python
test_conf = 0.25       # Lower confidence threshold for better recall
nmsthre = 0.45         # Optimized NMS for precision-recall balance
```

##### 5) Evaluation Framework Enhancement
- **Robust evaluation pipeline**: Fixed distributed training compatibility
- **Comprehensive metrics tracking**: Detailed performance monitoring
- **Failure case analysis**: Systematic improvement identification

## Dataset Information

### Primary Dataset: Tennis Ball Detection
- **Location**: https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection
- **Format**: COCO JSON format
- **Classes**: 1 (tennis_ball)
- **Characteristics**: Standard tennis court scenarios, various lighting conditions

### Enhanced Dataset: Larger Tennis Balls
- **Location**: https://universe.roboflow.com/tennis-3ll0a/tennis-ball-icifx
- **Format**: COCO JSON format
- **Description**: Close-up tennis ball shots, professional tournament footage, etc.
- **Key improvements**: Better coverage of medium/large tennis balls 


## Quick Start
Clone yolox repo (git clone https://github.com/Megvii-BaseDetection/YOLOX.git) and replace YOLOX/exps/example/custom/yolox_custom.py with yolox_custom.py uploaded in this repo.

### Training
```bash
# Train with improved configuration
python tools/train.py -f exps/example/custom/yolox_custom.py -d 1 -b 8 --fp16

### Evaluation
```bash
# Test on custom images
python tools/demo.py image -f exps/example/custom/yolox_custom.py --path assets/tennis_images/ --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
```

### Inference
```python
from yolox.utils import postprocess
import torch

# Load model
model = torch.load('best_ckpt.pth')['model']
model.eval()

# Inference on image
with torch.no_grad():
    outputs = model(img_tensor)
    predictions = postprocess(outputs, num_classes=1, conf_thre=0.25, nms_thre=0.45)
```

