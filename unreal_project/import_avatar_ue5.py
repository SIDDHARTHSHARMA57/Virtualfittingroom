"""
Unreal Engine 5 Python Asset Importer Script
Imports the reconstructed avatar OBJ/GLB mesh into the Unreal Engine Content Browser.
Can be executed directly within Unreal Editor (Python terminal) or via UE commandlet.
"""

import os
try:
    import unreal
except ImportError:
    unreal = None

def import_avatar_mesh(obj_file_path, destination_path="/Game/Avatars"):
    if unreal is None:
        print("[UE5 Importer] unreal module not available outside UE editor environment.")
        return False

    if not os.path.exists(obj_file_path):
        print(f"[UE5 Importer] File not found: {obj_file_path}")
        return False

    task = unreal.AssetImportTask()
    task.filename = obj_file_path
    task.destination_path = destination_path
    task.destination_name = os.path.splitext(os.path.basename(obj_file_path))[0]
    task.replace_existing = True
    task.automated = True
    task.save = True

    # Use Static Mesh options for OBJ
    options = unreal.FbxImportUI()
    options.import_mesh = True
    options.import_textures = False
    options.import_materials = True
    options.import_as_skeletal = False
    task.options = options

    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    print(f"[UE5 Importer] Successfully imported {obj_file_path} into {destination_path}")
    return True

if __name__ == "__main__":
    avatar_obj = "/media/siddharth-sharam/Workspace/VirtualFittingRoom/outputs/reconstruction/demo_user.obj"
    import_avatar_mesh(avatar_obj)
