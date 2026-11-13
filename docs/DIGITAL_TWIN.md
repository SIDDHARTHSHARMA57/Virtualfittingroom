# SMPL-X Digital Twin: anatomy, collision, live camera background

Status 2026-11-13. Built and tested on the **SMPL-X twin** (Windows, Blender 5.3). The MakeHuman
`Human`/`Human.rig` scene is not in this repo; retargeting onto it is pending (see Limitations).

## Data flow

```
Camera --OpenCV--> ThreadedCamera (capture thread, latest frame)
   |                    |
   |  VFR-CameraFeed    |  VFR-ServiceWorker: MediaPipe pose/hands/face (person-locked)
   |  thread, ~30 fps   |        |
   v                    |        v  VFR-SMPLXFit thread
shared memory           |  SMPLXFittingEngine.solve(): LBFGS on bone directions
"vfr_camera_frame"      |     + joint-limit penalty   (body_model.py)
(seqlock, latest only)  |     + self-collision penalty (body_collision.py)
   |                    |     -> hard clamp into limits -> hands -> camera-space translation
   |                    v
   |             TCP 127.0.0.1:8766, JSON line per frame, key "smplx"
   |             {global_orient, body_pose, hand poses, betas, transl, camera, penetrations}
   v                    v
Blender (src/blender_smplx_receiver.py, main-thread timer @60 Hz)
   VFR_CameraFeed image (updated in place) -> camera background (viewport) + emission plane (render)
   SMPLX_Twin_rig pose bones <- rotations; Limit Rotation constraints (safety net)
   SMPLX_Twin shape key "VFR_PoseCorrectives" <- worker thread (SMPL-X pose blend shapes)
   VFR_Camera <- intrinsics; rig location <- transl
```

## Files

| File | Role |
|---|---|
| `config/body_model.json` | Joint ranges (deg, Euler order per joint), collision weights/pairs. Shared by solver and Blender. |
| `src/camera/body_model.py` | Limits: Blender-convention Euler math, soft penalty (torch), hard clamp (numpy). Self-test: `python src/camera/body_model.py` |
| `src/camera/body_collision.py` | Capsule proxies measured from the SMPL-X mesh/skin weights; batched differentiable penetration penalty. |
| `src/camera/smplx_fitter.py` | Solver: limits + collision in the loss, `_finalize_pose` clamp + penetration check, `_solve_translation`. |
| `src/camera/frame_share.py` | Shared-memory latest-frame writer/reader (numpy + stdlib; Blender imports it). |
| `src/camera/service.py` | Feed thread, intrinsics loading (`load_intrinsics`), extra packet fields. |
| `src/blender_smplx_receiver.py` | Builds and drives the Blender scene. |
| `scripts/calibrate_camera.py` | Checkerboard calibration -> `config/camera_intrinsics.json`. |
| `tests/test_body_anatomy.py`, `tests/test_smplx_fitter.py`, `tests/blender_twin_check.py` | Tests (below). |

## Joint limits

* Ranges live in `config/body_model.json` (`joint_limits_deg`), left side + centre joints; right side is
  mirrored (x kept, y/z negated). Frames: SMPL-X local, world-aligned at rest (+X body-left, +Y up, +Z forward).
* Each joint has its own Euler order with the smallest-range axis in the middle (gimbal-safe):
  shoulder `YXZ` (twist in the middle), elbow `XZY`, others `XYZ`.
* Enforcement: soft penalty inside the solve (`limit_weight`), then an exact clamp of the final pose.
  Blender's Limit Rotation constraints (LOCAL space, same order/ranges) are a safety net; the test shows
  they produce the same matrices as the solver's clamp (max diff 2e-7), so they never fight the solver.
* Tune by editing the JSON; both sides reload it on restart.

## Collision prevention

* Proxies: distal upper arm ("elbow"), forearm, hand sphere, thigh (lower 75%), shin; torso = two side-by-side
  capsules (elliptic cross-section). Radii = median distance of each bone's skinned vertices to its segment
  x `radius_scale` (0.9). Measured: forearm 3.0 cm, elbow 4.9, hand 3.7, thigh 7.0-7.3, shin 4.9-5.0,
  torso 10.9 cm (+2.9 cm side offset).
* Checked pairs (`collision.pairs`): forearms/hands/elbows vs torso, thigh vs thigh, shin vs shin, shins vs torso.
  Forearm vs forearm is not checked (crossed arms legitimately touch).
* Penetration is a smooth penalty in the same solve (`weight` 3000), so corrections keep the user's intended
  direction and inherit the temporal smoothing (no oscillation). `slack_m` (1 cm) of skin contact is allowed.
  Every fit reports remaining penetration (`penetrations` in the packet; 0.5 cm tolerance).

## Mesh deformation

The twin uses SMPL-X's own skin weights plus its **pose corrective blend shapes** (shape key, recomputed
per pose on a Blender worker thread). Result: identical to the SMPL-X model (0.13 mm max vertex error),
i.e. no candy-wrapper elbows/shoulders beyond what SMPL-X itself does. No per-frame mesh regeneration.
"Preserve Volume" (dual quaternion) is not used: it would deviate from the SMPL-X deformation the
correctives are trained for. No gravity/rigid body on the body (garments later).

## Camera background and calibration

* No second camera stream: the service's capture thread feeds both tracking and Blender.
* Blender: `VFR_Camera` uses the intrinsics from the packet; render resolution = camera resolution;
  lens = fx * 36 / width; principal point -> shift_x/shift_y. The emission plane `VFR_Background` is sized to
  fill the frustum exactly at 12 m. Test: the twin's pelvis projects within 0.000 px of the pinhole model.
* Placement: with rotation solved, the camera-space translation is a linear least-squares fit of the posed
  model joints to the 2D landmarks.
* **Calibrate** (until then a 60 deg horizontal FOV is assumed, which may scale/offset the overlay):
  close the dashboard, print a checkerboard, run
  `python scripts/calibrate_camera.py --cols 9 --rows 6 --square 0.025`, capture >= 12 varied views (SPACE),
  press C. Restart the dashboard. Lens distortion is measured but not yet undistorted.

## Running

1. `launch_dashboard.bat` -> Start.
2. Click **Open Digital Twin in Blender** (or `launch_blender_twin.bat`). Blender opens in camera view with
   the live feed behind the twin. Errors: `outputs/logs/blender_twin.log`.

## Test results (actual, this machine: i7-10th gen, no GPU)

| Test | Result |
|---|---|
| Fit accuracy, synthetic, front/side/rear + 360 deg turn | 0.5 deg static, 1.9 deg mean / 2.5 deg p95 in motion |
| Hyperextended knee / elbow request | clamped to 0 deg / -5 deg (the limit) |
| 12 random limb-into-torso requests (up to 6.7 cm deep) | 0.00 cm after fit |
| Crossed arms | no penetration, no limit violation |
| Hand driven through abdomen over 30 frames | 0.00 cm, max 4.3 deg/frame change (request 3.8) |
| Blender mesh vs SMPL-X (with correctives) | 0.13 mm max |
| Blender constraints vs solver clamp | 2e-7 |
| Camera frame -> Blender image | correct orientation + RGB |
| Camera disconnect | "no signal", pose updates continue |
| Blender soak, 600 ticks, new frame + pose each | tick p50 5.8 ms / p95 7.6 ms, memory 225 -> 225 MB |
| Live run (webcam + service + Blender GUI, 25 s) | camera 30 fps, inference 3-6 fps, Blender feed 22 fps, fit 230-720 ms, CPU 200-390% (4 cores saturated), service RSS ~575 MB stable |

Not measured: GPU usage (no GPU), Blender viewport FPS, camera-to-render latency end-to-end,
and the movement table with a real person (crossing arms, knees, walking) — run these live and report.

## Limitations / next steps

1. **MakeHuman retarget** pending: copy the MakeHuman `.blend` into the repo; then inspect its bone axes and
   map the (already limited, collision-free) SMPL-X rotations onto it.
2. **CPU-bound on this laptop**: with Blender open the fit drops to ~1-4 Hz. Options: `pose_landmarker_lite`,
   disable face mesh, fewer LBFGS iterations, or run on the GPU machine.
3. Background runs ahead of the pose by the inference latency (smooth video vs exact sync trade-off;
   see `_feed_loop`).
4. Body size is the SMPL-X template (betas = 0) -> overlay scale/depth is approximate until Phase 1
   shape estimation; lens distortion not undistorted.
5. Finger joint limits and finger collision are not implemented; collision is limb-vs-body only.
