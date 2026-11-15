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

## Feet, head, fingers (added 2026-11-13)

* **Feet**: MediaPipe's heel (29/30) and foot-index (31/32) are skin points, so the model now has matching
  skin points (`FOOT_POINTS` in `smplx_fitter.py`: SMPL-X heel vertex on the ankle, big+small toe tips on the
  foot joint) and fits the heel->toe segment (yaw + pitch) plus ankle->heel (heel lift). Previously the
  toe tip was matched to the foot *joint*, biasing yaw/pitch by up to 11 deg. Foot roll is not observable from
  MediaPipe's 2 foot points (ankle limits keep it plausible). `tests/test_feet.py`.
* **Head**: the face landmarker's facial transformation matrix drives the head's full orientation (nod, tilt,
  turn) as a solver term, accepted only when it is a proper rotation and agrees < 45 deg with the ears->eyes
  heading. Live: agreement median 3.7 deg, max 6.2 deg (convention confirmed).
* **Fingers**: per-segment bend capped at `finger_max_bend_deg` (110).

## Person selection (multi-person)

`src/camera/person_tracker.py`: YOLO11n (`models/yolo11n.pt`) + ByteTrack (ultralytics; needs `lap`) give
temporary IDs; `PersonSelector` is the state machine (IDLE, DETECTING, SELECTING, LOCKED, TEMPORARILY_LOST,
REACQUIRING, LOST, RESET). Only the selected person is fed to MediaPipe: the frame is masked outside their
(expanded) box, so other people's pose/hand/face landmarks cannot mix in; Blender's background still gets
the unmasked frame. Selection: tap/click a box in the dashboard preview, or pick an ID + Select; Reset clears.
One person in view is auto-selected (`TrackingService.auto_select_single`); several people never are.
While the target is missing the last pose is held; a *new* track ID is re-acquired only if it overlaps the
last box, matches the clothing-colour histogram, was never on screen together with the target, and is the
only such candidate. After `reacquire_timeout_s` (5 s) -> LOST: paused until someone is selected explicitly.
No face recognition; nothing stored beyond the session. Without ultralytics/yolo11n.pt the service falls
back to the previous MediaPipe-only person lock. `tests/test_person_selector.py` (19 scenario checks).

## Accuracy round 2 (overlay, silhouettes, body shape)

* **Silhouette masking**: `models/yolo11n-seg.pt` (falls back to `yolo11n.pt` boxes). MediaPipe sees only
  the selected person's dilated silhouette; other people's silhouettes are removed even inside the
  target's box (fixes a bystander's hand being fitted as the customer's arm).
* **2D reprojection**: the solver also minimises the angular reprojection error of the posed model's
  keypoints against MediaPipe's 2D landmarks, with the camera-space position as a variable
  (`reprojection_weight` 150), and down-weights the depth component of MediaPipe's 3D directions
  (`world_depth_weight` 0.5) because depth is its weakest axis. Weights > 150 overfit 2D and hurt 3D.
  Dashboard "SMPL-X Status" shows the live overlay error in px.
* **Body shape**: `src/camera/body_shape.py` estimates SMPL-X betas from median lengths of 10 segments
  (limbs, shoulders, torso; not hip width) every 30 frames; the solver re-rigs on change > 0.1 and the
  Blender twin reshapes in place (`apply_shape`). Shape is kept until a different person is selected.
  `estimate_body_shape` in the config turns it off. Absolute size follows MediaPipe's metric estimate.

| Test | Result |
|---|---|
| Overlay, MediaPipe-like noise (2D +-1.5 px, depth +-6 cm) | image error 22.9 -> 9.1 px, 3D bone error 19.2 -> 12.7 deg |
| Body shape, 3 random bodies, 90 frames | segment lengths within 0.83 cm, stature within 1.6 cm (template: 4.8-7.0 cm off) |
| Bystander hand inside target box (masking) | masked; target torso kept |
| Blender twin with non-template shape | 0.13 mm vs SMPL-X |

## Body-shape calibration (personalised body)

Shape (layer A: SMPL-X betas) and pose (layer B: joint rotations) are separate parameters; the pose
solve never changes the betas. Module: `src/camera/body_shape.py`, settings: `body_shape` in
`config/body_model.json`. Only the selected person is measured (MediaPipe only sees their silhouette);
measurements stop while they are lost; a newly selected person starts a fresh calibration and the
previous person's samples and shape are discarded.

1. **Guided calibration** (starts on selection; dashboard "Recalibrate" restarts it). Each frame is
   checked for a neutral pose: facing the camera (< 25 deg), upright (< 12 deg), arms >= 20 deg away from
   the torso side and below the shoulders, legs straight if visible. The failed check is shown as the
   instruction ("Turn to face the camera", "Move your arms slightly away from your body", ...). After 25
   good frames the shape is solved once (median per measurement).
2. **Evidence**: 11 segment lengths (limbs, shoulders, torso, shoulders->head) from MediaPipe world
   landmarks; front waist and hip widths from the segmentation outline, as a ratio to the image shoulder
   width (camera distance cancels), minus a 2 cm clothing allowance. Chest width is NOT measured from the
   outline: the upper arm touches the torso below the armpit in any comfortable pose (+3 cm bias in tests).
3. **Refinement**: lengths keep accumulating from any frame (segments pointing at the camera rejected),
   widths only from neutral-pose frames; every 60 frames the shape is re-solved and applied only if some
   measurement changes by > 1.5 cm. Shape changes are rate-limited (0.15 per beta per frame), so the
   body morphs over ~1 s; Blender reshapes in place (~15 ms per step). Collision proxies, heel/toe points
   and joint positions are rebuilt from the reshaped mesh.
4. **Report** (dashboard preview overlay): each measurement with source and confidence.
   *observed*: segment lengths, front waist/hip widths. *model*: height, chest/waist/hip/thigh/calf/neck
   circumferences, read off the fitted mesh; circumferences need a side view or depth camera to be measured.

| Test (`tests/test_body_shape.py`, rendered synthetic people) | Result |
|---|---|
| Height, 3 random bodies | 0.2-1.9 cm (template 2.1-8.6 cm) |
| Segment lengths | max 0.4-1.4 cm |
| Front waist/hip widths | 0.6-1.9 cm with silhouette vs up to 4.3 cm lengths-only, 5.4 cm template |
| 70 frames of arms up/down, bending, knee raises after calibration | max measurement change 0.00 cm |
| Switch A (179 cm) -> B (167 cm) | 168.4 cm, no carry-over |
| Pose gate | A-pose accepted; arms at sides, turned 50 deg, leaning 20 deg, arms raised rejected with the right message |

Limits: left/right asymmetry is not modelled (SMPL-X shape space is symmetric); absolute size follows
MediaPipe's metric estimate until the camera is calibrated or a height is entered; loose clothing widens
the measured waist/hips beyond the allowance.

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
| Feet: forward / outward / one foot out / inward / heel + toe raise / seated | yaw within 1 deg, worst yaw/pitch error 4.2 deg (was 11.2) |
| Head nod+tilt+turn (synthetic) | 2.4 deg pose-only -> 0.3 deg with face matrix |
| Person selector scenarios (one, three, tap middle, others move/approach, occlusion, new ID, impostor, timeout, reselect, reset) | 19/19 pass |
| Live, 1 person, 20 s with person detection | auto-LOCKED; detector 58 ms, pose 83 ms, fit 296 ms, inference 4.3 fps, camera 30 fps, process RSS 645 MB |
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
