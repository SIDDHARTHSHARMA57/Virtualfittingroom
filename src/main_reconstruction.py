"""
VirtualFittingRoom — Milestone 2 Main Entrypoint
CLI tool to run 3D human body reconstruction from live webcam or static image,
compute anthropometric evaluation metrics, and export 3D assets for Unreal Engine 5.
"""

import os
import sys
import argparse
import json

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.reconstruction.pipeline import ReconstructionPipeline
from src.reconstruction.evaluation import compute_body_proportions
import trimesh

def main():
    parser = argparse.ArgumentParser(description="Milestone 2: 3D Human Reconstruction Pipeline")
    parser.add_argument("--source", type=str, default="webcam", help="'webcam' or path to an image file")
    parser.add_argument("--output_dir", type=str, default="/media/siddharth-sharam/Workspace/VirtualFittingRoom/outputs/reconstruction")
    parser.add_argument("--prefix", type=str, default="user_avatar")
    args = parser.parse_args()

    print("==================================================")
    print(" VirtualFittingRoom — 3D Reconstruction Pipeline ")
    print("==================================================")

    pipeline = ReconstructionPipeline(
        yolo_model_path=os.path.join(project_root, "yolov8n-pose.pt"),
        smpl_dir=os.path.join(project_root, "models/smpl")
    )

    if args.source == "webcam":
        import cv2
        print("[Main] Capturing reference frame from /dev/video0...")
        cap = cv2.VideoCapture(0)
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("[Main] Error: Could not capture from webcam!")
            return
        meta = pipeline.reconstruct_from_frame(frame, output_dir=args.output_dir, prefix=args.prefix)
    else:
        meta = pipeline.reconstruct_from_image_file(args.source, output_dir=args.output_dir, prefix=args.prefix)

    if meta is None:
        print("[Main] Reconstruction failed or no person was visible in the frame.")
        return

    # Load resulting mesh and compute anthropometric metrics
    mesh = trimesh.load(meta["obj_mesh"])
    proportions = {
        "estimated_height_cm": float(abs((mesh.bounds[1][1] - mesh.bounds[0][1]) * 100.0)),
        "bounding_box_width_cm": float(abs((mesh.bounds[1][0] - mesh.bounds[0][0]) * 100.0)),
        "bounding_box_depth_cm": float(abs((mesh.bounds[1][2] - mesh.bounds[0][2]) * 100.0)),
    }

    print("\n--- Reconstruction Metrics & Proportions ---")
    print(f"• Detection latency: {meta['detect_time_ms']:.1f} ms")
    print(f"• 3D Fitting latency: {meta['fit_time_ms']:.1f} ms")
    print(f"• Fitting Optimization Loss: {meta['fit_loss']:.4f}")
    print(f"• Vertices: {meta['num_vertices']} | Faces: {meta['num_faces']}")
    print(f"• Estimated Mesh Height: {proportions['estimated_height_cm']:.1f} cm")
    print(f"• Estimated Mesh Width: {proportions['bounding_box_width_cm']:.1f} cm")
    print("\n--- Exported Assets for Unreal Engine 5.8.3 ---")
    print(f"• OBJ Mesh: {meta['obj_mesh']}")
    print(f"• GLB Mesh: {meta['glb_mesh']}")
    print(f"• Metadata: {os.path.join(args.output_dir, args.prefix + '_metadata.json')}")
    print("==================================================")

if __name__ == "__main__":
    main()
