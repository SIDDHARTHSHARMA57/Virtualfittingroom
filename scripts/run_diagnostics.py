#!/usr/bin/env python3
"""
Diagnostic Script for Real-Time Camera-Driven SMPL-X Avatar Project.
Checks:
1. Project directories and Python virtual environments.
2. OpenCV and MediaPipe package availability in python environments.
3. Webcam accessibility and frame capture via OpenCV.
4. Blender 5.2.2 executable launch and version verification.
5. Blender Python script inspection of scene, meshes, armatures, and bone names.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/media/siddharth-sharam/Workspace/VirtualFittingRoom")
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BLENDER_EXEC = Path("/media/siddharth-sharam/Workspace/Office files/blender-5.2.2-linux-x64/blender")
STAGE1_PYTHON = PROJECT_ROOT / "envs" / "stage1" / "bin" / "python"
SMPLX_PYTHON = PROJECT_ROOT / "envs" / "smplx" / "bin" / "python"

def log_header(title):
    return f"\n{'='*60}\n{title}\n{'='*60}\n"

def run_diagnostics():
    report = {
        "timestamp": datetime.now().isoformat(),
        "system": {},
        "directories": {},
        "environments": {},
        "webcam": {},
        "blender": {},
        "issues_and_recommendations": []
    }

    print(log_header("1. CHECKING DIRECTORIES & PYTHON ENVIRONMENTS"))
    # Check Project Root
    dirs_to_check = {
        "project_root": PROJECT_ROOT,
        "logs": LOGS_DIR,
        "models": PROJECT_ROOT / "models",
        "models_smplx": PROJECT_ROOT / "models" / "smplx",
        "envs": PROJECT_ROOT / "envs",
        "scripts": PROJECT_ROOT / "scripts"
    }

    for name, path in dirs_to_check.items():
        exists = path.exists() and path.is_dir()
        report["directories"][name] = {"path": str(path), "exists": exists}
        print(f"[{'PASS' if exists else 'FAIL'}] Directory {name}: {path}")

    # Check Python environments
    envs_to_check = {
        "stage1": STAGE1_PYTHON,
        "smplx": SMPLX_PYTHON
    }

    for env_name, py_path in envs_to_check.items():
        exists = py_path.exists() and os.access(py_path, os.X_OK)
        report["environments"][env_name] = {"path": str(py_path), "exists": exists}
        print(f"[{'PASS' if exists else 'FAIL'}] Python environment '{env_name}': {py_path}")

    print(log_header("2. CHECKING PACKAGES (OPENCV, MEDIAPIPE, SMPL-X, TORCH)"))
    target_py = STAGE1_PYTHON if STAGE1_PYTHON.exists() else Path(sys.executable)

    for env_name, py_path in envs_to_check.items():
        if not py_path.exists():
            continue
        print(f"\n--- Environment: {env_name} ({py_path}) ---")
        code = """
import sys, json
results = {}
for pkg in ['cv2', 'mediapipe', 'torch', 'ultralytics', 'smplx']:
    try:
        mod = __import__(pkg)
        results[pkg] = {
            'status': 'installed',
            'version': str(getattr(mod, '__version__', 'unknown'))
        }
    except Exception as e:
        results[pkg] = {
            'status': 'missing_or_error',
            'error': str(e)
        }
print('__RESULT__' + json.dumps(results))
"""
        try:
            res = subprocess.run([str(py_path), "-c", code], capture_output=True, text=True, check=True)
            for line in res.stdout.splitlines():
                if line.startswith("__RESULT__"):
                    pkg_data = json.loads(line[len("__RESULT__"):])
                    report["environments"][env_name]["packages"] = pkg_data
                    for pkg, pinfo in pkg_data.items():
                        status_str = "PASS" if pinfo["status"] == "installed" else "WARN"
                        print(f"  [{status_str}] {pkg}: {pinfo.get('version', pinfo.get('error'))}")
        except Exception as e:
            print(f"  [FAIL] Error querying packages for {env_name}: {e}")
            report["environments"][env_name]["error"] = str(e)

    print(log_header("3. CHECKING WEBCAM ACCESSIBILITY"))
    # Test camera via OpenCV using stage1 python or current python
    cam_test_code = """
import cv2, json
res = {'cameras_found': []}
for dev_idx in [0, 1]:
    cap = cv2.VideoCapture(dev_idx)
    is_opened = cap.isOpened()
    frame_read = False
    width = 0
    height = 0
    if is_opened:
        ret, frame = cap.read()
        frame_read = ret and (frame is not None) and (frame.size > 0)
        if frame_read:
            height, width = frame.shape[:2]
        cap.release()
    res['cameras_found'].append({
        'index': dev_idx,
        'opened': is_opened,
        'frame_read': frame_read,
        'resolution': f'{width}x{height}' if frame_read else None
    })
print('__CAM__' + json.dumps(res))
"""
    try:
        res = subprocess.run([str(target_py), "-c", cam_test_code], capture_output=True, text=True, timeout=15)
        cam_info = None
        for line in res.stdout.splitlines():
            if line.startswith("__CAM__"):
                cam_info = json.loads(line[len("__CAM__"):])
                break
        if cam_info:
            report["webcam"] = cam_info
            any_success = False
            for c in cam_info["cameras_found"]:
                status = "PASS" if c["frame_read"] else ("WARN" if c["opened"] else "FAIL")
                print(f"[{status}] Camera Index {c['index']}: Opened={c['opened']}, Frame Captured={c['frame_read']}, Res={c['resolution']}")
                if c["frame_read"]:
                    any_success = True
            if not any_success:
                report["issues_and_recommendations"].append("No camera device successfully returned an active frame. Check USB connection or V4L2 permissions.")
        else:
            print(f"[FAIL] Camera test produced no structured output. Stderr: {res.stderr}")
            report["webcam"]["error"] = res.stderr
    except Exception as e:
        print(f"[FAIL] Failed executing camera probe: {e}")
        report["webcam"]["error"] = str(e)

    print(log_header("4. CHECKING BLENDER EXECUTABLE"))
    if not BLENDER_EXEC.exists():
        print(f"[FAIL] Blender binary not found at: {BLENDER_EXEC}")
        report["blender"]["executable_found"] = False
        report["issues_and_recommendations"].append(f"Blender executable missing at {BLENDER_EXEC}")
    else:
        report["blender"]["executable_found"] = True
        try:
            ver_res = subprocess.run([str(BLENDER_EXEC), "--version"], capture_output=True, text=True, check=True)
            ver_output = ver_res.stdout.splitlines()[0] if ver_res.stdout else "Unknown"
            print(f"[PASS] Blender launches successfully: {ver_output}")
            report["blender"]["version_output"] = ver_output
        except Exception as e:
            print(f"[FAIL] Error launching Blender: {e}")
            report["blender"]["version_error"] = str(e)

    print(log_header("5. CHECKING BLENDER PYTHON SCENE & SKELETON INSPECTION"))
    inspect_script = PROJECT_ROOT / "scripts" / "blender_inspect.py"
    if BLENDER_EXEC.exists() and inspect_script.exists():
        try:
            # Run Blender headless in background mode to inspect default/blank scene
            cmd = [str(BLENDER_EXEC), "--background", "--python", str(inspect_script)]
            b_res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            
            json_blob = ""
            recording = False
            for line in b_res.stdout.splitlines():
                if "--- BLENDER INSPECTION JSON START ---" in line:
                    recording = True
                    continue
                if "--- BLENDER INSPECTION JSON END ---" in line:
                    recording = False
                    continue
                if recording:
                    json_blob += line + "\n"

            if json_blob.strip():
                inspection_data = json.loads(json_blob)
                report["blender"]["inspection"] = inspection_data
                print(f"[PASS] Blender embedded python ran successfully!")
                print(f"       Blender Version: {inspection_data.get('blender_version')}")
                print(f"       Active Scene: {inspection_data.get('active_scene')}")
                print(f"       Meshes Found: {len(inspection_data.get('meshes', []))}")
                for m in inspection_data.get("meshes", []):
                    print(f"         - Mesh '{m['name']}': {m['vertices_count']} vertices, {m['polygons_count']} polygons")
                print(f"       Armatures Found: {len(inspection_data.get('armatures', []))}")
                for arm in inspection_data.get("armatures", []):
                    print(f"         - Armature '{arm['name']}': {arm['bones_count']} bones ({arm['bone_names'][:5]}...)")
            else:
                print("[WARN] Blender executed but output did not contain inspection JSON.")
                report["blender"]["raw_stdout"] = b_res.stdout
                report["blender"]["raw_stderr"] = b_res.stderr
        except Exception as e:
            print(f"[FAIL] Blender python inspection failed: {e}")
            report["blender"]["inspection_error"] = str(e)
    else:
        print("[FAIL] Skipping Blender inspection because executable or script is missing.")

    # Save output log files
    log_json = LOGS_DIR / f"diagnostic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    latest_json = LOGS_DIR / "diagnostic_results_latest.json"
    latest_txt = LOGS_DIR / "diagnostic_report_latest.txt"

    with open(log_json, "w") as f:
        json.dump(report, f, indent=2)
    with open(latest_json, "w") as f:
        json.dump(report, f, indent=2)

    with open(latest_txt, "w") as f:
        f.write(f"VIRTUAL FITTING ROOM DIAGNOSTIC REPORT\nTimestamp: {report['timestamp']}\n\n")
        f.write(f"Directories:\n{json.dumps(report['directories'], indent=2)}\n\n")
        f.write(f"Environments & Packages:\n{json.dumps(report['environments'], indent=2)}\n\n")
        f.write(f"Webcam:\n{json.dumps(report['webcam'], indent=2)}\n\n")
        f.write(f"Blender:\n{json.dumps(report['blender'], indent=2)}\n\n")

    print(log_header("DIAGNOSTICS COMPLETE"))
    print(f"Full results saved to:")
    print(f" - {latest_json}")
    print(f" - {latest_txt}")
    return report

if __name__ == "__main__":
    run_diagnostics()
