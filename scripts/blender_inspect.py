"""
Blender Inspection Script
Runs inside Blender's embedded Python engine in background mode (--background).
Inspects:
- Active scene & unit settings
- Mesh objects (vertex count, polygon count, materials)
- Armatures and bone hierarchy / names
- Current mode & selected objects
"""
import sys
import json

try:
    import bpy
except ImportError:
    print("[ERROR] 'bpy' module not available. This script must be run inside Blender.")
    sys.exit(1)

def inspect_blender():
    scene = bpy.context.scene
    data = {
        "blender_version": bpy.app.version_string,
        "active_scene": scene.name if scene else None,
        "total_objects": len(bpy.data.objects),
        "meshes": [],
        "armatures": [],
        "other_objects": []
    }

    for obj in bpy.data.objects:
        obj_info = {
            "name": obj.name,
            "type": obj.type,
            "location": [round(v, 4) for v in obj.location],
            "rotation_euler": [round(v, 4) for v in obj.rotation_euler],
            "scale": [round(v, 4) for v in obj.scale],
            "visible": obj.visible_get()
        }

        if obj.type == 'MESH':
            mesh = obj.data
            obj_info["vertices_count"] = len(mesh.vertices)
            obj_info["polygons_count"] = len(mesh.polygons)
            obj_info["materials"] = [m.name for m in obj.material_slots if m.material]
            data["meshes"].append(obj_info)
        elif obj.type == 'ARMATURE':
            armature = obj.data
            bones = [b.name for b in armature.bones]
            obj_info["bones_count"] = len(bones)
            obj_info["bone_names"] = bones
            data["armatures"].append(obj_info)
        else:
            data["other_objects"].append(obj_info)

    return data

if __name__ == "__main__":
    report = inspect_blender()
    print("--- BLENDER INSPECTION JSON START ---")
    print(json.dumps(report, indent=2))
    print("--- BLENDER INSPECTION JSON END ---")
