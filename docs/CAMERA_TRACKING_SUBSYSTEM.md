# VirtualFittingRoom — Camera & Tracking Subsystem

This document provides complete instructions for running, testing, and troubleshooting the modular camera capture and full-body motion tracking service.

---

## 1. Architecture Overview

Located in `src/camera/`:
- `capture.py`: Dedicated background capture thread (`ThreadedCamera`) reading exclusively from `/dev/video0`. Implements a thread-safe latest-frame buffer, stale frame detection, and automatic reconnection.
- `tracking.py`: Multi-model inference engine (`UnifiedTracker`) executing MediaPipe Pose (33 landmarks + 3D world points), Hands (up to 42 landmarks), and Face (478 mesh points, transformation matrix, and 52 blendshapes). Features 1-Euro temporal filtering for low jitter.
- `protocol.py`: Serializer for the 127.0.0.1:8766 localhost stream matching Blender receiver (`blender_pose_receiver.py`) schema.
- `pipeline.py`: Non-blocking coordinator managing camera frames, inference execution, socket transmission with timeout guards, and optional GUI preview.
- `monitor.py`: Rate-limited telemetry logger reporting capture/inference FPS, frame age, model latencies, frame drops, and network health.
- `main.py`: CLI entry point supporting customizable resolutions, device indices, and model toggles.

---

## 2. Environment Activation

Always activate the project conda environment before running:
```bash
source /media/siddharth-sharam/Workspace/miniforge3/etc/profile.d/conda.sh
conda activate /media/siddharth-sharam/Workspace/VirtualFittingRoom/envs/smplx
cd /media/siddharth-sharam/Workspace/VirtualFittingRoom
```

---

## 3. Terminal Commands

### A. Run Automated Unit Tests (No Camera / No Blender Needed)
```bash
python -m unittest tests/test_camera_subsystem.py
```

### B. Hardware Camera Smoke Test
Verify `/dev/video0` can deliver 30 frames cleanly at 640x480 MJPG:
```bash
python -c "
import cv2, time
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
frames = 0; t0 = time.time()
while frames < 30:
    ok, f = cap.read()
    if ok and f is not None: frames += 1
cap.release()
print(f'Camera smoke test ok: {frames} frames in {time.time()-t0:.2f}s')
"
```

### C. Run Tracking Subsystem (Headless / Production Mode)
Streams tracking data directly to Blender at `127.0.0.1:8766` without GUI overhead:
```bash
python src/camera/main.py
```

### D. Run Tracking Subsystem with GUI Preview
Displays live webcam feed with skeleton overlay and FPS counter:
```bash
python src/camera/main.py --preview
```

### E. Run High-Performance Lightweight Tracking (Pose Only)
Disables hand and face landmarker inference for minimal CPU usage:
```bash
python src/camera/main.py --no-hands --no-face
```

---

## 4. Troubleshooting Guide

### 1. Camera preview freezes or stops updating
- **Cause**: In single-threaded loops, MediaPipe latency or socket blocking delays `cv2.waitKey()`, freezing the OS window manager event loop.
- **Remedy**: The new subsystem decouples capture and inference on `ThreadedCamera`. Run without `--preview` for production, or ensure you do not add blocking network calls to the preview loop.

### 2. Device `/dev/video0` busy or inaccessible
- Check if another application or background process holds the camera:
  ```bash
  fuser -v /dev/video*
  ```
- Kill lingering sender processes if necessary.

### 3. Blender Receiver Disconnected / Port 8766 Error
- The new sender automatically handles disconnections and reconnects gracefully every 1 second without stalling camera capture.
- Verify Blender receiver is listening:
  ```bash
  netstat -tlpn | grep 8766
  ```

---

## 5. Graphical Camera & Skeleton Tracking Dashboard

A dedicated, lightweight graphical application is provided in `src/camera/dashboard.py`.

### Features
1. **Live Camera Viewport**: Real-time video with customizable skeleton and landmark overlays.
2. **Overlay Controls**:
   - `Skeleton Connections`: Full limb & torso kinematic connections.
   - `Body Joint Landmarks`: 33 MediaPipe pose landmarks with highlighted shoulder/hip/wrist anchors.
   - `Hands & Fingers`: 21 keypoints per hand with distinct color coding per handedness.
   - `Dense Face Mesh`: 478 face landmarks and expression mesh.
3. **Inference Switches**: Dynamically toggle Hand or Face inference on and off without restarting to conserve CPU/battery.
4. **Hardware & Resolution Configuration**: Switch between 640x480, 1280x720, or 640x360 MJPEG modes.
5. **Mode Switching**:
   - `Stream to Blender`: Sends Schema 2 packets directly to `127.0.0.1:8766`.
   - `Preview Only`: Standalone preview with zero network overhead.
6. **Live Telemetry HUD**: Shows Camera FPS, Inference FPS, combined latency (ms), stale frames, and Blender connection status.
7. **Control Actions**: Start, Stop, and Reconnect buttons.

### Launching the Dashboard

```bash
source /media/siddharth-sharam/Workspace/miniforge3/etc/profile.d/conda.sh
conda activate /media/siddharth-sharam/Workspace/VirtualFittingRoom/envs/smplx
cd /media/siddharth-sharam/Workspace/VirtualFittingRoom

python src/camera/dashboard.py
```

> **Note on Camera Ownership**:
> Linux V4L2 permits only one process to hold `/dev/video0` at a time. If the background CLI sender (`python src/camera/main.py`) is already running in another terminal, stop it (Ctrl+C) before starting the dashboard, or the dashboard will alert you that the camera is in use.

---

## 6. Dense Anatomical Body Tracking & 3D SMPL-X Surface Reconstruction

The system features dense anatomical body region tracking and 3D parametric surface fitting:

### 1. Distinct Data Categories
- **Observed Landmarks**: Directly measured from the camera feed (MediaPipe 33 pose, 42 hand, 478 face).
- **Inferred Anatomical Contours**: Derived body contours with explicit uncertainty:
  - **Torso**: Left/right clavicles, sternum & manubrium, chest & rib cage contours, navel, waist boundaries, pelvis.
  - **Arms**: Upper arm (bicep/tricep) contours, forearm contours.
  - **Legs**: Thigh contours (front, inner, outer), calf & shin contours.
  - **Head**: Neck junction and head orientation.
- **Parametric 3D Mesh Vertices**: Fitted 10,475-vertex SMPL-X neutral body mesh (`SMPLX_NEUTRAL.pkl`).

### 2. Dual-Viewport Graphical Dashboard
- **Viewport 1 (Left)**: Live camera feed with customizable overlays:
  - Yellow: Skeleton connections
  - Green/Orange: Directly observed joints
  - Cyan: Inferred anatomical contours & chest/waist lines
  - Magenta/Cyan: Left/Right hands & fingers
  - Lavender: Dense face mesh
- **Viewport 2 (Right)**: Real-time 3D parametric mesh viewport with **Front**, **Side**, and **Rear** orthographic perspectives.
- **Dynamic Controls**:
  - Toggle `SMPL-X 3D Parametric Mesh` in real time to enable or disable dense GPU fitting.
  - Toggle `Inferred Contours`, `Skeleton`, `Joints`, `Hands`, and `Face` independently.
  - Telemetry HUD showing observed landmark counts, inferred surface points, and 10,475 mesh vertex status.

---

## 7. SMPL-X Orientation & Live Motion Retargeting Architecture

### Coordinate Frame Alignment
- **SMPL-X 3D Space**: $+X$ = Lateral Right, $+Y$ = Vertical Up (Head $Y=+0.27$, Pelvis $Y=-0.35$), $+Z$ = Forward.
- **MediaPipe 3D World Landmarks**: $+X$ = Lateral Right, $+Y$ = Vertical Down, $+Z$ = Forward.
- **Screen Viewport Space**: $+X$ = Right, $+Y$ = Down (top of image is $y=0$).
- **Projection Rule in [drawing.py](file:///media/siddharth-sharam/Workspace/VirtualFittingRoom/src/camera/drawing.py)**:
  - $\text{Screen}_Y = -\text{proj}_V$
  - Head ($+Y$ in 3D) correctly projects with the lowest screen $y$ coordinate (top of viewport), and feet ($-Y$ in 3D) project with the highest screen $y$ coordinate (bottom of viewport) in all **Front**, **Side**, and **Rear** perspectives.

### Real-Time Articulation Retargeting
- Implemented in [smplx_fitter.py](file:///media/siddharth-sharam/Workspace/VirtualFittingRoom/src/camera/smplx_fitter.py):
  - Solves joint rotations using Rodrigues axis-angle formulations from tracked limb vectors (Upper arms, Forearms, Thighs, Calves).
  - Keeps body shape $\beta$ parameters constant, eliminating shape drift during motion.
  - Applies temporal pose smoothing ($\alpha=0.40$) for jitter-free real-time response.
  - Telemetry HUD displays active status: **LIVE MOTION** (green) vs **REST POSE** (grey), tracked joint count (`/12`), and fit latency (`ms`).

---

## 8. Coordinate Correction & Full-Body Pose Retargeting

### Coordinate Conventions & Depth Boundary Correction

| System | Lateral Axis | Vertical Axis | Depth Axis | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **MediaPipe World 3D** | $+X$ = Lateral Right | $+Y$ = Downward | $+Z$ = Away from camera | Camera looks directly AT user |
| **SMPL-X Model Space** | $+X$ = Lateral Right | $+Y$ = Upward | $+Z$ = Forward (in front of chest/body) | Natural anatomical coordinate frame |
| **Screen Raster Viewport**| $+X$ = Lateral Right | $+Y$ = Downward | N/A | $y=0$ at top edge |

#### The Depth Direction Fix
- When the user reaches toward the camera, distance from the camera decreases, which corresponds to **$-Z$ in MediaPipe world coordinates**.
- In the SMPL-X model space, the hand extending forward in front of the human chest requires **$+Z$ in SMPL-X coordinates**.
- **Boundary Mapping in [smplx_fitter.py](file:///media/siddharth-sharam/Workspace/VirtualFittingRoom/src/camera/smplx_fitter.py)**:
  $$\vec{v}_{\text{smplx}} = \begin{bmatrix} dx_{\text{mp}} \\ -dy_{\text{mp}} \\ -dz_{\text{mp}} \end{bmatrix}$$
  This ensures:
  1. Reaching toward the camera translates the SMPL-X hand forward in $+Z$ (towards the front).
  2. Withdrawing the hand translates the SMPL-X hand backward in $-Z$.
  3. Head, spine, pelvis, knees, and feet share this identical depth orientation.

### Kinematic Articulation Coverage

1. **Torso & Spine**:
   - Computes trunk vector $\vec{v}_{\text{trunk}} = \text{Mid}_{\text{shoulders}} - \text{Mid}_{\text{hips}}$.
   - Distributes bending and twisting across `spine1` (lumbar), `spine2` (mid-thoracic), and `spine3` (upper-thoracic).
2. **Head & Neck**:
   - Tracks head orientation from nose/ears relative to mid-shoulders across `neck` and `head` joints.
3. **Arms & Forearms**:
   - Left & Right `shoulder` and `elbow` joints articulate with Rodrigues axis-angle formulations.
4. **Hips & Thighs (Standing vs. Sitting)**:
   - Tracks hip-to-knee vectors. In a sitting posture, thighs flex forward in $+Z$, correctly bending the SMPL-X model at the hip joints.
5. **Knees & Calves**:
   - Tracks knee flexion and lower leg extension from knee-to-ankle vectors.
6. **Ankles & Feet**:
   - Tracks ankle-to-toe orientation across left and right feet.

### Viewport Indicators
- **3D Coordinate Tripod**: Displays active axes in the bottom-left corner of the 3D viewport ($+Z$ Forward in cyan, $+Y$ Up in green).
