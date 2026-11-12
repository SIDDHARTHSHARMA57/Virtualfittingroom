"""
Mesh and Skeleton Exporter
Exports SMPL mesh to OBJ, GLTF/GLB, and metadata JSON for Unreal Engine 5.8.3.
"""

import os
import json
import numpy as np
import trimesh

def export_mesh_obj(vertices, faces, output_path):
    """Export 3D vertices and triangular faces to an .obj file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(output_path, file_type='obj')
    print(f"[MeshExporter] Saved OBJ mesh: {output_path} ({len(vertices)} verts, {len(faces)} faces)")
    return output_path

def export_mesh_glb(vertices, faces, output_path):
    """Export 3D vertices and triangular faces to a .glb binary file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(output_path, file_type='glb')
    print(f"[MeshExporter] Saved GLB mesh: {output_path}")
    return output_path

def export_reconstruction_summary(metadata, output_path):
    """Save reconstruction metadata (betas, pose parameters, loss, metrics) as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert numpy types to list/float
    serializable = {}
    for k, v in metadata.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            serializable[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            serializable[k] = int(v)
        else:
            serializable[k] = v

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[MeshExporter] Saved reconstruction summary: {output_path}")
    return output_path
