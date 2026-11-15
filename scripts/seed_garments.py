"""
Create the built-in test garments and add them to the clothing library through the SAME pipeline as
an administrator upload (generate OBJ -> stage upload -> headless processing -> approve).

  python scripts/seed_garments.py            (needs Blender: BLENDER env var, PATH or E:\\Apps\\Blender)
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.garments.registry import Registry, find_blender  # noqa: E402

PROC = os.path.join(ROOT, "src", "garments", "process_garment.py")


def run_blender(blender, *args) -> str:
    out = subprocess.run([blender, "-b", "--factory-startup", "--python", PROC, "--", *args],
                         capture_output=True, text=True, timeout=600)
    return out.stdout + out.stderr


def process_upload(reg: Registry, blender: str, path: str, **opts) -> dict:
    staged = reg.stage_upload(path)
    args = ["--input", staged["original"], "--id", staged["id"], "--library", reg.dir]
    for k, v in opts.items():
        args += [f"--{k}", str(v)]
    log = run_blender(blender, *args)
    rep_path = os.path.join(reg.pending_dir(staged["id"]), "report.json")
    if not os.path.isfile(rep_path):
        raise RuntimeError("processor produced no report:\n" + log[-2000:])
    with open(rep_path, encoding="utf-8") as f:
        return json.load(f)


def main():
    blender = find_blender()
    if blender is None:
        raise SystemExit("Blender not found (set BLENDER=path\\to\\blender.exe)")
    reg = Registry()
    existing = {g["name"] for g in reg.list()}
    tmp = tempfile.mkdtemp(prefix="vfr_garments_")
    for kind, name, ease in (("tshirt", "Basic T-Shirt", 0.012), ("trousers", "Basic Trousers", 0.015)):
        if name in existing:
            print(f"{name}: already in the library")
            continue
        obj = os.path.join(tmp, f"{kind}.obj")
        print(run_blender(blender, "--generate", kind, "--out", obj, "--ease", str(ease)).split("VFR_GENERATED")[-1].splitlines()[0])
        rep = process_upload(reg, blender, obj, name=name, category=kind)
        if rep["status"] != "ok":
            raise SystemExit(f"{name}: processing failed: {rep['errors']}")
        entry = reg.approve(rep)
        print(f"{name}: added as {entry['id']} ({rep['stats']}) warnings={rep['warnings']}")


if __name__ == "__main__":
    main()
