"""
Reconstruction Pipeline Orchestrator
Captures input image/video/webcam or test data, runs 2D pose estimation,
fits personalized SMPL mesh, and exports OBJ / GLB assets for UE5.
"""

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

from .fitter import SMPLFitter
from .mesh_exporter import export_mesh_obj, export_mesh_glb, export_reconstruction_summary

class ReconstructionPipeline:
    def __init__(self, yolo_model_path="yolov8n-pose.pt", smpl_dir="/media/siddharth-sharam/Workspace/VirtualFittingRoom/models/smpl"):
        print("[Pipeline] Initializing YOLOv8 Pose model...")
        self.detector = YOLO(yolo_model_path)
        print("[Pipeline] Initializing SMPL Fitter...")
        self.fitter = SMPLFitter(model_dir=smpl_dir)

    def reconstruct_from_frame(self, frame_bgr, output_dir="/media/siddharth-sharam/Workspace/VirtualFittingRoom/outputs/reconstruction", prefix="avatar"):
        """Run pose detection, optimize SMPL, and export mesh."""
        os.makedirs(output_dir, exist_ok=True)
        h, w = frame_bgr.shape[:2]
        
        t0 = time.time()
        results = self.detector(frame_bgr, verbose=False)
        t_detect = (time.time() - t0) * 1000

        if len(results) == 0 or len(results[0].keypoints) == 0 or results[0].keypoints.xy.shape[1] == 0:
            print("[Pipeline] Warning: No person detected in frame!")
            return None

        # Take the most confident person detected
        kps_xy = results[0].keypoints.xy[0].cpu().numpy() # (17, 2)
        kps_conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)

        print(f"[Pipeline] Pose detected in {t_detect:.1f}ms. Running 3D body fitting...")
        t1 = time.time()
        fit_results = self.fitter.fit_from_coco_keypoints(kps_xy, kps_conf, img_w=w, img_h=h, num_iters=100)
        t_fit = (time.time() - t1) * 1000
        print(f"[Pipeline] Body fitted in {t_fit:.1f}ms with loss: {fit_results['loss']:.4f}")

        # Export meshes
        obj_path = os.path.join(output_dir, f"{prefix}.obj")
        glb_path = os.path.join(output_dir, f"{prefix}.glb")
        meta_path = os.path.join(output_dir, f"{prefix}_metadata.json")

        export_mesh_obj(fit_results["vertices"], fit_results["faces"], obj_path)
        export_mesh_glb(fit_results["vertices"], fit_results["faces"], glb_path)

        meta = {
            "betas": fit_results["betas"],
            "global_orient": fit_results["global_orient"],
            "body_pose": fit_results["body_pose"],
            "transl": fit_results["transl"],
            "fit_loss": fit_results["loss"],
            "detect_time_ms": t_detect,
            "fit_time_ms": t_fit,
            "image_resolution": [w, h],
            "num_vertices": len(fit_results["vertices"]),
            "num_faces": len(fit_results["faces"]),
            "obj_mesh": obj_path,
            "glb_mesh": glb_path
        }
        export_reconstruction_summary(meta, meta_path)

        # Save annotated input image
        annotated_img = results[0].plot()
        annotated_path = os.path.join(output_dir, f"{prefix}_pose.jpg")
        cv2.imwrite(annotated_path, annotated_img)
        print(f"[Pipeline] Annotated pose saved: {annotated_path}")

        return meta

    def reconstruct_from_image_file(self, image_path, output_dir="/media/siddharth-sharam/Workspace/VirtualFittingRoom/outputs/reconstruction", prefix="avatar"):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        return self.reconstruct_from_frame(img, output_dir, prefix)

