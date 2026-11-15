"""
Clothing library registry: one JSON file + an asset directory. No Blender dependency.

  assets/garments/registry.json          approved garments (what customers can select)
  assets/garments/originals/<id>/...     uploaded files, never modified
  assets/garments/processed/<id>.blend   processed, fitted garment ready to wear
  assets/garments/previews/<id>.png      preview render
  assets/garments/pending/<id>/          processing output awaiting approval (or failed)

Garment IDs are generated (`g_<8 hex>`), so uploads can never overwrite each other's files.
"""
import json
import os
import shutil
import time
import uuid
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LIBRARY = os.path.join(ROOT, "assets", "garments")
FABRICS_PATH = os.path.join(ROOT, "config", "fabrics.json")
MESH_FORMATS = {".fbx", ".obj", ".glb", ".gltf", ".blend"}
IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def find_blender() -> Optional[str]:
    """BLENDER env var, PATH, the team's E:\\Apps\\Blender, then Program Files (newest first)."""
    import glob
    cands = [os.environ.get("BLENDER"), shutil.which("blender"), r"E:\Apps\Blender\blender.exe"]
    cands += sorted(glob.glob(os.path.join(os.environ.get("ProgramFiles", ""), "Blender Foundation",
                                           "Blender*", "blender.exe")), reverse=True)
    return next((c for c in cands if c and os.path.isfile(c)), None)


def load_fabrics(path: str = FABRICS_PATH) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


class Registry:
    def __init__(self, library_dir: str = DEFAULT_LIBRARY):
        self.dir = library_dir
        self.path = os.path.join(library_dir, "registry.json")
        for sub in ("originals", "processed", "previews", "pending"):
            os.makedirs(os.path.join(library_dir, sub), exist_ok=True)
        self.garments: Dict[str, dict] = {}
        self.load()

    # ---- persistence ----
    def load(self):
        if os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8-sig") as f:
                self.garments = {g["id"]: g for g in json.load(f).get("garments", [])}
        return self

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "garments": list(self.garments.values())}, f, indent=2)
        os.replace(tmp, self.path)   # atomic: a crash never leaves a half-written registry

    # ---- paths ----
    def rel(self, path: str) -> str:
        return os.path.relpath(path, self.dir).replace("\\", "/")

    def abs(self, rel_path: str) -> str:
        return os.path.normpath(os.path.join(self.dir, rel_path))

    @staticmethod
    def new_id() -> str:
        return "g_" + uuid.uuid4().hex[:8]

    def pending_dir(self, gid: str) -> str:
        return os.path.join(self.dir, "pending", gid)

    # ---- upload ----
    def stage_upload(self, src_path: str) -> dict:
        """Copy an uploaded file into originals/<new id>/ (the source is never touched).
        Returns {"id", "original", "kind": "mesh" | "image"}; raises ValueError if unsupported."""
        if not os.path.isfile(src_path):
            raise ValueError(f"File not found: {src_path}")
        ext = os.path.splitext(src_path)[1].lower()
        if ext not in MESH_FORMATS | IMAGE_FORMATS:
            raise ValueError(f"Unsupported format '{ext}'. 3D: {sorted(MESH_FORMATS)}; photos: {sorted(IMAGE_FORMATS)}")
        if os.path.getsize(src_path) == 0:
            raise ValueError("The file is empty (corrupted upload?)")
        gid = self.new_id()
        dst_dir = os.path.join(self.dir, "originals", gid)
        os.makedirs(dst_dir)
        dst = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst)
        if ext == ".gltf":  # .gltf references external .bin / textures next to it
            src_dir = os.path.dirname(os.path.abspath(src_path))
            try:
                for ref in json.load(open(src_path, encoding="utf-8")).get("buffers", []) + \
                           json.load(open(src_path, encoding="utf-8")).get("images", []):
                    uri = ref.get("uri", "")
                    if uri and not uri.startswith("data:") and os.path.isfile(os.path.join(src_dir, uri)):
                        os.makedirs(os.path.dirname(os.path.join(dst_dir, uri)), exist_ok=True)
                        shutil.copy2(os.path.join(src_dir, uri), os.path.join(dst_dir, uri))
            except (ValueError, OSError):
                pass  # the processor reports missing dependencies
        return {"id": gid, "original": dst, "kind": "image" if ext in IMAGE_FORMATS else "mesh"}

    # ---- approve / edit / delete ----
    def approve(self, report: dict) -> dict:
        """Register a successfully processed garment from its pending report (moves files in)."""
        if report.get("status") != "ok":
            raise ValueError("Only successfully processed garments can be added: " + "; ".join(report.get("errors", [])))
        gid = report["id"]
        pend = self.pending_dir(gid)
        blend_dst = os.path.join(self.dir, "processed", gid + ".blend")
        prev_dst = os.path.join(self.dir, "previews", gid + ".png")
        shutil.move(os.path.join(pend, "garment.blend"), blend_dst)
        if os.path.isfile(os.path.join(pend, "preview.png")):
            shutil.move(os.path.join(pend, "preview.png"), prev_dst)
        entry = {
            "id": gid, "name": report["name"], "category": report["category"], "region": report["region"],
            "original_filename": os.path.basename(report["original"]),
            "original": self.rel(report["original"]), "processed": self.rel(blend_dst),
            "preview": self.rel(prev_dst) if os.path.isfile(prev_dst) else "",
            "object": report["object"], "materials": report.get("materials", []),
            "fabric_profile": report["fabric_profile"], "fabric_overrides": {},
            "fit_style": report["fit_style"], "supported_fits": ["tight", "regular", "loose", "oversized"],
            "base_size": report.get("base_size", "M"), "sizes": report.get("sizes", ["S", "M", "L"]),
            "collision": {"capsules": True, "self_collision": False},
            "warnings": report.get("warnings", []), "stats": report.get("stats", {}),
            "uploaded": report.get("uploaded", time.strftime("%Y-%m-%d %H:%M:%S")),
            "status": "approved",
        }
        prev = self.garments.get(gid)
        if prev:  # reprocessing: keep the administrator's edits, take the new geometry + report
            for k in ("name", "fabric_profile", "fabric_overrides", "collision", "base_size", "sizes", "uploaded"):
                entry[k] = prev.get(k, entry[k])
            entry["reprocessed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.garments[gid] = entry
        self.save()
        shutil.rmtree(pend, ignore_errors=True)
        return entry

    def cancel(self, gid: str, delete_original: bool = True):
        """Discard a pending upload (and, by default, its copied original - nothing was registered)."""
        shutil.rmtree(self.pending_dir(gid), ignore_errors=True)
        if delete_original and gid not in self.garments:
            shutil.rmtree(os.path.join(self.dir, "originals", gid), ignore_errors=True)

    def update(self, gid: str, **fields) -> dict:
        allowed = {"name", "category", "region", "fit_style", "fabric_profile", "fabric_overrides",
                   "collision", "preview", "base_size", "sizes"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Not editable: {sorted(bad)}")
        self.garments[gid].update(fields)
        self.save()
        return self.garments[gid]

    def delete(self, gid: str, delete_processed: bool = True, delete_original: bool = False) -> List[str]:
        """Remove a garment from the library. Files are only deleted when asked and only if no other
        garment references them (shared textures/materials are never removed). Returns deleted paths."""
        g = self.garments.pop(gid)
        self.save()
        still_used = {self.abs(v) for o in self.garments.values() for k, v in o.items()
                      if k in ("processed", "preview", "original") and v}
        removed = []
        targets = []
        if delete_processed:
            targets += [g.get("processed"), g.get("preview")]
        if delete_original:
            targets += [g.get("original")]
        for rel_path in filter(None, targets):
            p = self.abs(rel_path)
            if p in still_used or not os.path.exists(p):
                continue
            os.remove(p)
            removed.append(p)
        if delete_original:
            d = os.path.join(self.dir, "originals", gid)
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        return removed

    def list(self, category: Optional[str] = None) -> List[dict]:
        return sorted((g for g in self.garments.values() if category in (None, "", "all", g["category"], g["region"])),
                      key=lambda g: g["name"].lower())
