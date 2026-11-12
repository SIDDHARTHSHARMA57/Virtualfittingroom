# VirtualFittingRoom — Stage 1

## Overview
This repository contains Stage 1 of the **VirtualFittingRoom** project:
1. **Milestone 1 (Complete)**: Robust real-time pose tracking system using camera capture and dual model backends (YOLO11 Pose on CUDA and Google MediaPipe Tasks).
2. **Milestone 2 (Upcoming)**: 3D Body Reconstruction evaluation and pipeline setup (SMPL/SMPL-X, HMR2.0, SMPLer-X).
3. **Milestone 3 (Upcoming)**: Live avatar animation & pose retargeting pipeline.
4. **Milestone 4 (Upcoming)**: Unreal Engine 5 integration.
5. **Milestone 5 (Upcoming)**: Controlled 360-degree capture workflow.

---

## Environment & Hardware Status
- **OS**: Ubuntu Linux (Kernel x86_64)
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- **Driver**: 595.99.02, CUDA 13.2
- **PyTorch**: 2.13.0+cu132 (Verified CUDA accelerated)
- **Python Env**: `/media/siddharth-sharam/Workspace/VirtualFittingRoom/envs/stage1` (Python 3.12.15)
- **Unreal Engine**: Found at `/media/siddharth-sharam/Workspace/Unreal Engine/Linux_Unreal_Engine_5.8.3`

---

## Project Structure
```
/media/siddharth-sharam/Workspace/VirtualFittingRoom/
├── config.yaml          # Camera, model backend, and UI configuration
├── models/              # Pretrained weights (stored on Workspace drive)
│   ├── yolo11n-pose.pt  # Ultralytics YOLO11 Nano pose model
│   └── pose_landmarker_lite.task # Google MediaPipe pose task
├── src/
│   ├── camera.py        # OpenCV camera manager with reconnection logic
│   ├── tracker.py       # Modular pose tracking (YOLO & MediaPipe)
│   ├── visualizer.py    # UI overlay, skeleton renderer, HUD & telemetry
│   └── main.py          # Application entry point with benchmark mode
└── README.md
```

---

## Running Milestone 1

### 1. Launch Interactive Live Window
Activate the virtual environment and run:
```bash
/media/siddharth-sharam/Workspace/VirtualFittingRoom/envs/stage1/bin/python src/main.py
```

### Controls:
- **`Q` or `ESC`**: Quit application
- **`B`**: Switch tracker backend live between **YOLO (CUDA)** and **MediaPipe (CPU Tasks)**
- **`M`**: Toggle mirror mode

### 2. Run Headless Benchmark
To benchmark processing FPS and latency over N frames:
```bash
/media/siddharth-sharam/Workspace/VirtualFittingRoom/envs/stage1/bin/python src/main.py 50
```

---

## Measured Performance on RTX 4050 Laptop GPU
- **YOLO11 Pose (CUDA)**: ~12–21 ms inference latency per frame (~45–80 FPS model throughput, camera stream rate bounded by 30 FPS camera hardware).
- **MediaPipe Pose (CPU Tasks)**: ~38–41 ms latency per frame (~25 FPS model throughput).
