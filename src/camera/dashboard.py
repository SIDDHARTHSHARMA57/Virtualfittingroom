"""
VirtualFittingRoom — Dense Anatomical Body Tracking & Live SMPL-X Motion Retargeting.
Features:
- Dual-Viewport Architecture:
    1. Primary Live Camera View with toggleable anatomical regions & contours.
    2. Real-time 3D Parametric Mesh Viewport (Front, Side, Rear perspectives of 10,475-vertex SMPL-X).
- Verified Upright Orientation:
    - Fixed inverted Y screen projection (+Y SMPL-X height correctly maps to top of viewport).
- Real-time Articulation:
    - SMPL-X kinematic joints update continuously from user motion (arms, elbows, hips, knees).
- Telemetry HUD:
    - Sequence/packet counts, live fitting latency, valid joint counts, and mode status.
"""
import os
import sys
import glob
import time
import shutil
import subprocess
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.camera.service import TrackingService
from src.camera.drawing import render_tracking_overlays, render_3d_orthographic_view

logger = logging.getLogger("VFR-Dashboard")

class CameraDashboardApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VirtualFittingRoom — Dense Anatomical Body Tracking & SMPL-X Live Retargeting")
        self.root.geometry("1280x820")
        self.root.minsize(1000, 680)
        self.root.configure(bg="#1e1e24")

        # Models
        # full model: much better 3D (world) landmarks than lite; falls back if not downloaded
        self.pose_model = os.path.join(PROJECT_ROOT, "models/pose_landmarker_full.task")
        if not os.path.isfile(self.pose_model):
            self.pose_model = os.path.join(PROJECT_ROOT, "models/pose_landmarker_lite.task")
        self.hand_model = os.path.join(PROJECT_ROOT, "models/hand_landmarker.task")
        self.face_model = os.path.join(PROJECT_ROOT, "models/face_landmarker.task")
        self.smplx_dir = os.path.join(PROJECT_ROOT, "models")

        # Service
        self.service: Optional[TrackingService] = None

        # Device & Streaming State
        self.var_device = tk.StringVar(value="0 (/dev/video0)")
        self.var_resolution = tk.StringVar(value="640x480 (30 FPS MJPG)")
        self.var_stream_blender = tk.BooleanVar(value=True)
        self.var_smooth = tk.BooleanVar(value=True)

        # Anatomical & Overlay Toggles
        self.var_draw_skeleton = tk.BooleanVar(value=True)
        self.var_draw_joints = tk.BooleanVar(value=True)
        self.var_draw_anatomy = tk.BooleanVar(value=True)
        self.var_draw_hands = tk.BooleanVar(value=True)
        self.var_draw_face = tk.BooleanVar(value=True)

        # 3D Viewport Controls
        self.var_smplx_enable = tk.BooleanVar(value=True)
        self.var_3d_view = tk.StringVar(value="front")

        # Telemetry Labels
        self.stat_cam_fps = tk.StringVar(value="--")
        self.stat_inf_fps = tk.StringVar(value="--")
        self.stat_latency = tk.StringVar(value="-- ms")
        self.stat_obs_pts = tk.StringVar(value="0")
        self.stat_surf_pts = tk.StringVar(value="0")
        self.stat_smplx_mode = tk.StringVar(value="Rest Pose")
        self.stat_smplx_joints = tk.StringVar(value="0/12")
        self.stat_blender = tk.StringVar(value="Connecting...")
        self.stat_status = tk.StringVar(value="Ready to Start")

        # Photo references
        self.current_cam_photo: Optional[ImageTk.PhotoImage] = None
        self.current_3d_photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(33, self._ui_update_loop)

    def _build_ui(self):
        # Header Bar
        header = tk.Frame(self.root, bg="#2a2a32", height=50)
        header.pack(fill=tk.X, side=tk.TOP, padx=8, pady=(8, 4))

        title = tk.Label(
            header,
            text="3D Virtual Fitting Room — Live SMPL-X Anatomical Body Retargeting",
            font=("Helvetica", 14, "bold"),
            fg="#61afef",
            bg="#2a2a32"
        )
        title.pack(side=tk.LEFT, padx=14, pady=10)

        self.status_badge = tk.Label(
            header,
            textvariable=self.stat_status,
            font=("Helvetica", 10, "bold"),
            fg="#98c379",
            bg="#333842",
            padx=10,
            pady=4
        )
        self.status_badge.pack(side=tk.RIGHT, padx=14)

        # Main Workspace Container
        main_container = tk.Frame(self.root, bg="#1e1e24")
        main_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Dual Viewports Container
        viewport_container = tk.Frame(main_container, bg="#1e1e24")
        viewport_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # Viewport 1: Live Camera + Anatomical Contours
        self.cam_frame = tk.LabelFrame(
            viewport_container,
            text=" Live Camera & Anatomical Contours ",
            font=("Helvetica", 10, "bold"),
            fg="#abb2bf",
            bg="#121216"
        )
        self.cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.cam_frame.pack_propagate(False)  # image size must not dictate frame size

        self.cam_canvas = tk.Label(self.cam_frame, bg="#121216")
        self.cam_canvas.pack(fill=tk.BOTH, expand=True)

        # Viewport 2: 3D Reconstructed Body Surface (Front/Side/Rear)
        self.view3d_frame = tk.LabelFrame(
            viewport_container,
            text=" 3D Parametric Body Surface (SMPL-X) ",
            font=("Helvetica", 10, "bold"),
            fg="#abb2bf",
            bg="#121216",
            width=350
        )
        # before=cam_frame: fixed-width panels get their space first, camera takes the remainder
        self.view3d_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(4, 0), before=self.cam_frame)
        self.view3d_frame.pack_propagate(False)

        # 3D View Angle Controls
        view_ctrls = tk.Frame(self.view3d_frame, bg="#18181f")
        view_ctrls.pack(fill=tk.X, side=tk.TOP, pady=4, padx=4)

        tk.Label(view_ctrls, text="Perspective:", fg="#abb2bf", bg="#18181f", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=4)
        for angle in ("front", "side", "rear"):
            tk.Radiobutton(
                view_ctrls,
                text=angle.capitalize(),
                variable=self.var_3d_view,
                value=angle,
                fg="#61afef",
                bg="#18181f",
                selectcolor="#121216",
                activebackground="#18181f"
            ).pack(side=tk.LEFT, padx=2)

        self.canvas_3d = tk.Label(self.view3d_frame, bg="#121216")
        self.canvas_3d.pack(fill=tk.BOTH, expand=True)

        # Right Sidebar: Controls & Live Diagnostics
        sidebar = tk.Frame(main_container, bg="#25252c", width=330)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0), before=viewport_container)
        sidebar.pack_propagate(False)

        # 1. Device & Mode Setup Card
        c_setup = tk.LabelFrame(sidebar, text=" Camera & Network Setup ", font=("Helvetica", 10, "bold"),
                                fg="#abb2bf", bg="#25252c", padx=10, pady=6)
        c_setup.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(c_setup, text="Device:", fg="#abb2bf", bg="#25252c").grid(row=0, column=0, sticky="w", pady=2)
        cb_dev = ttk.Combobox(c_setup, textvariable=self.var_device, values=["0 (/dev/video0)"], state="readonly", width=18)
        cb_dev.grid(row=0, column=1, sticky="e", pady=2)

        tk.Label(c_setup, text="Resolution:", fg="#abb2bf", bg="#25252c").grid(row=1, column=0, sticky="w", pady=2)
        cb_res = ttk.Combobox(c_setup, textvariable=self.var_resolution,
                              values=["640x480 (30 FPS MJPG)", "1280x720 (30 FPS MJPG)", "640x360 (30 FPS MJPG)"],
                              state="readonly", width=18)
        cb_res.grid(row=1, column=1, sticky="e", pady=2)

        chk_stream = tk.Checkbutton(c_setup, text="Stream to Blender (127.0.0.1:8766)",
                                    variable=self.var_stream_blender, fg="#98c379", bg="#25252c",
                                    selectcolor="#1e1e24", activebackground="#25252c")
        chk_stream.grid(row=2, column=0, columnspan=2, sticky="w", pady=3)

        # 2. Controls Action Buttons
        btn_frame = tk.Frame(sidebar, bg="#25252c")
        btn_frame.pack(fill=tk.X, padx=8, pady=4)

        self.btn_start = tk.Button(btn_frame, text="▶ Start", bg="#98c379", fg="#1e1e24",
                                   font=("Helvetica", 10, "bold"), width=8, command=self.action_start)
        self.btn_start.pack(side=tk.LEFT, padx=2)

        self.btn_stop = tk.Button(btn_frame, text="⏹ Stop", bg="#e06c75", fg="#1e1e24",
                                  font=("Helvetica", 10, "bold"), width=8, state=tk.DISABLED, command=self.action_stop)
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        self.btn_recon = tk.Button(btn_frame, text="🔄 Reconnect", bg="#e5c07b", fg="#1e1e24",
                                   font=("Helvetica", 10, "bold"), width=9, command=self.action_reconnect)
        self.btn_recon.pack(side=tk.LEFT, padx=2)

        self.btn_twin = tk.Button(sidebar, text="Open Digital Twin in Blender", bg="#61afef", fg="#1e1e24",
                                  font=("Helvetica", 10, "bold"), command=self.action_open_twin)
        self.btn_twin.pack(fill=tk.X, padx=10, pady=(0, 4))

        # 3. Anatomical & Regional Overlays
        c_vis = tk.LabelFrame(sidebar, text=" Layers & Kinematic Toggles ", font=("Helvetica", 10, "bold"),
                              fg="#abb2bf", bg="#25252c", padx=10, pady=6)
        c_vis.pack(fill=tk.X, padx=8, pady=4)

        tk.Checkbutton(c_vis, text="Skeleton Kinematic Chain", variable=self.var_draw_skeleton,
                       fg="#e5c07b", bg="#25252c", selectcolor="#1e1e24", activebackground="#25252c").pack(anchor="w")
        tk.Checkbutton(c_vis, text="Observed Body Joints", variable=self.var_draw_joints,
                       fg="#98c379", bg="#25252c", selectcolor="#1e1e24", activebackground="#25252c").pack(anchor="w")
        tk.Checkbutton(c_vis, text="Inferred Contours (Clavicle/Chest)", variable=self.var_draw_anatomy,
                       fg="#61afef", bg="#25252c", selectcolor="#1e1e24", activebackground="#25252c").pack(anchor="w")
        tk.Checkbutton(c_vis, text="Hands & Finger Joints", variable=self.var_draw_hands,
                       fg="#d19a66", bg="#25252c", selectcolor="#1e1e24", activebackground="#25252c").pack(anchor="w")
        tk.Checkbutton(c_vis, text="Dense Face Mesh", variable=self.var_draw_face,
                       fg="#c678dd", bg="#25252c", selectcolor="#1e1e24", activebackground="#25252c").pack(anchor="w")
        tk.Checkbutton(c_vis, text="SMPL-X Live Motion Retargeting", variable=self.var_smplx_enable,
                       command=self._sync_smplx_flag, fg="#98c379", bg="#25252c",
                       selectcolor="#1e1e24", activebackground="#25252c").pack(anchor="w")

        # 4. Live Telemetry & Motion Diagnostics
        c_stats = tk.LabelFrame(sidebar, text=" Telemetry & Live Motion Diagnostics ", font=("Helvetica", 10, "bold"),
                                fg="#abb2bf", bg="#25252c", padx=10, pady=6)
        c_stats.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        def add_stat_row(parent, label, tvar, row):
            tk.Label(parent, text=label, fg="#5c6370", bg="#25252c", font=("Helvetica", 9)).grid(row=row, column=0, sticky="w", pady=1)
            tk.Label(parent, textvariable=tvar, fg="#61afef", bg="#25252c", font=("Helvetica", 9, "bold")).grid(row=row, column=1, sticky="e", pady=1)

        c_stats.columnconfigure(0, weight=1)
        c_stats.columnconfigure(1, weight=1)
        add_stat_row(c_stats, "Camera FPS:", self.stat_cam_fps, 0)
        add_stat_row(c_stats, "Inference FPS:", self.stat_inf_fps, 1)
        add_stat_row(c_stats, "Pipeline Latency:", self.stat_latency, 2)
        add_stat_row(c_stats, "Observed Landmarks:", self.stat_obs_pts, 3)
        add_stat_row(c_stats, "Inferred Surface Contours:", self.stat_surf_pts, 4)
        add_stat_row(c_stats, "SMPL-X Status:", self.stat_smplx_mode, 5)
        add_stat_row(c_stats, "Tracked Joints in Fit:", self.stat_smplx_joints, 6)
        add_stat_row(c_stats, "Blender Transport:", self.stat_blender, 7)

    def _sync_smplx_flag(self):
        if self.service:
            self.service.enable_smplx = self.var_smplx_enable.get()

    def action_start(self):
        dev_str = self.var_device.get()
        dev_id = 0 if "0" in dev_str else 0

        res_str = self.var_resolution.get()
        w, h = 640, 480
        if "1280x720" in res_str:
            w, h = 1280, 720
        elif "640x360" in res_str:
            w, h = 640, 360

        stream_blender = self.var_stream_blender.get()

        self.service = TrackingService(
            pose_model_path=self.pose_model,
            hand_model_path=self.hand_model,
            face_model_path=self.face_model,
            smplx_model_dir=self.smplx_dir,
            device_id=dev_id,
            width=w,
            height=h,
            fps=30,
            fourcc="MJPG",
            stream_to_blender=stream_blender
        )
        self.service.enable_smplx = self.var_smplx_enable.get()

        ok = self.service.start_service()
        if ok:
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.stat_status.set("Live Motion Tracking Active")
            self.status_badge.config(fg="#98c379")
        else:
            if self.service.camera_busy:
                messagebox.showerror(
                    "Camera In Use / Device Busy",
                    f"Could not open /dev/video{dev_id}.\n\n"
                    "Another process is currently holding the camera.\n\n"
                    "Check running processes with: fuser -v /dev/video0"
                )
            self.stat_status.set("Device Unavailable")
            self.status_badge.config(fg="#e06c75")

    def action_stop(self):
        if self.service:
            self.service.stop_service()
            self.service = None
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.stat_status.set("Stopped")
        self.status_badge.config(fg="#abb2bf")

    def action_reconnect(self):
        self.action_stop()
        time.sleep(0.3)
        self.action_start()

    @staticmethod
    def _find_blender():
        """BLENDER env var, PATH, the team's E:\\Apps\\Blender, then Program Files."""
        cands = [os.environ.get("BLENDER"), shutil.which("blender"), r"E:\Apps\Blender\blender.exe"]
        cands += sorted(glob.glob(os.path.join(os.environ.get("ProgramFiles", ""), "Blender Foundation",
                                               "Blender*", "blender.exe")), reverse=True)
        return next((c for c in cands if c and os.path.isfile(c)), None)

    def action_open_twin(self):
        """Launch Blender with the SMPL-X twin receiver; the running service streams the pose to it."""
        proc = getattr(self, "_blender_proc", None)
        if proc is not None and proc.poll() is None:
            messagebox.showinfo("Digital Twin", "Blender is already running the digital twin.")
            return
        blender = self._find_blender()
        if blender is None:
            messagebox.showerror("Digital Twin", "Blender not found.\nSet the BLENDER environment variable "
                                                 "to blender.exe, or add Blender to PATH.")
            return
        script = os.path.join(PROJECT_ROOT, "src", "blender_smplx_receiver.py")
        env = dict(os.environ, VFR_ROOT=PROJECT_ROOT)
        self._blender_proc = subprocess.Popen([blender, "--python", script], cwd=PROJECT_ROOT, env=env)
        self.var_stream_blender.set(True)  # twin needs the pose stream
        if self.service:
            self.service.stream_to_blender = True

    def _ui_update_loop(self):
        try:
            if self.service and self.service._running:
                frame, tracking, anatomy, verts, stats, status_msg = self.service.get_ui_snapshot()

                # Telemetry
                self.stat_cam_fps.set(f"{stats.get('capture_fps', 0.0):.1f}")
                self.stat_inf_fps.set(f"{stats.get('inference_fps', 0.0):.1f}")
                lat_tot = stats.get('latency_ms', {}).get('total', 0.0)
                smplx_lat = stats.get('smplx_latency_ms', 0.0)
                self.stat_latency.set(f"{lat_tot:.0f}ms (+{smplx_lat:.0f}ms fit)" if smplx_lat > 0 else f"{lat_tot:.1f}ms")

                counts = anatomy.get("summary_counts", {})
                self.stat_obs_pts.set(str(counts.get("observed_landmarks", 0)))
                self.stat_surf_pts.set(str(counts.get("estimated_surface_points", 0)))

                is_live = stats.get("smplx_is_live", False)
                valid_j = stats.get("smplx_valid_joints", 0)
                self.stat_smplx_mode.set("LIVE MOTION" if is_live else "Rest Pose")
                self.stat_smplx_joints.set(f"{valid_j}/12 + {stats.get('smplx_hands', 0)} hands")

                net = stats.get("network", {})
                if not self.var_stream_blender.get():
                    self.stat_blender.set("Disabled (Preview Only)")
                elif net.get("connected"):
                    self.stat_blender.set(f"Connected ({net.get('packets_sent', 0)} pkts)")
                else:
                    self.stat_blender.set("Disconnected (Retrying)")

                # Viewport 1: Render Camera + Anatomical Overlays
                if frame is not None:
                    annotated = render_tracking_overlays(
                        frame=frame,
                        tracking_data=tracking,
                        draw_skeleton=self.var_draw_skeleton.get(),
                        draw_body_points=self.var_draw_joints.get(),
                        draw_hands=self.var_draw_hands.get(),
                        draw_face=self.var_draw_face.get(),
                        draw_anatomy=self.var_draw_anatomy.get(),
                        anatomical_data=anatomy
                    )

                    vw = max(self.cam_frame.winfo_width(), 320)
                    vh = max(self.cam_frame.winfo_height(), 240)
                    fh, fw = annotated.shape[:2]
                    scale = min(vw / fw, vh / fh)
                    nw, nh = int(fw * scale), int(fh * scale)

                    resized = cv2.resize(annotated, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    photo = ImageTk.PhotoImage(image=img)
                    self.cam_canvas.config(image=photo)
                    self.current_cam_photo = photo

                # Viewport 2: Render 3D Parametric Mesh with HUD
                v3d_w = max(self.view3d_frame.winfo_width() - 10, 200)
                v3d_h = max(self.view3d_frame.winfo_height() - 50, 240)
                mesh_3d_img = render_3d_orthographic_view(
                    vertices=verts,
                    view_angle=self.var_3d_view.get(),
                    width=v3d_w,
                    height=v3d_h,
                    is_live_fitted=is_live,
                    latency_ms=smplx_lat,
                    valid_joints=valid_j
                )
                rgb_3d = cv2.cvtColor(mesh_3d_img, cv2.COLOR_BGR2RGB)
                img_3d = Image.fromarray(rgb_3d)
                photo_3d = ImageTk.PhotoImage(image=img_3d)
                self.canvas_3d.config(image=photo_3d)
                self.current_3d_photo = photo_3d

        except Exception as e:
            logger.error(f"UI update loop error: {e}")
        finally:
            self.root.after(33, self._ui_update_loop)

    def _on_close(self):
        self.action_stop()
        self.root.destroy()

def launch_dashboard():
    root = tk.Tk()
    app = CameraDashboardApp(root)
    root.mainloop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
    launch_dashboard()
