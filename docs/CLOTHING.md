# Clothing library, garment fitting, Mirror Mode and cloth physics

Status 2026-11-15, Blender 5.3 on the SMPL-X twin (Windows, i7-10th gen, no GPU). Stage 1: a working
end-to-end system with two generated test garments; real garments come from the upload pipeline.

## How it fits together

```
upload (FBX/OBJ/GLB/glTF/.blend) --> originals/<id>/   (copy; the uploaded file is never touched)
        |  separate headless Blender process: src/garments/process_garment.py
        v
pending/<id>/ report.json + garment.blend + preview.png  --> Approve --> processed/<id>.blend, registry.json
                                                                 Cancel  --> discarded
Wear:  processed garment --> DRIVER copy (hidden) bound with Surface Deform to the twin
                             (template shape + garment bind pose) -> follows personalised shape + live pose
                         --> DISPLAY copy moved by the XPBD cloth solver (worker thread)
                              pins follow the driver; rest: gravity, inertia, body collision, hands
Mirror Mode: twin drawn with the camera image projected through the calibrated camera (invisible,
             but it occludes the garment where real arms/hands are in front); only the garment shows.
```

## Files

| File | Role |
|---|---|
| `src/garments/registry.py` | Library registry (JSON), staging uploads, approve/cancel/edit/delete (shared files never deleted), `find_blender()` |
| `src/garments/process_garment.py` | Headless processor + test-garment generator |
| `src/garments/cloth_sim.py` | XPBD cloth solver (numpy, no Blender) |
| `src/garments/blender_wardrobe.py` | Live wardrobe: wear/remove, simulation loop, hands, Mirror Mode, sidebar UI |
| `config/fabrics.json` | Fabric profiles, category defaults, fit-style ease, body regions |
| `scripts/seed_garments.py` | Creates "Basic T-Shirt" and "Basic Trousers" through the upload pipeline |
| `tests/test_cloth_sim.py`, `tests/blender_wardrobe_check.py` | Tests |

## Why a custom cloth solver

Measured here: Blender's built-in cloth (3.1k vertices, body collision) takes 68 ms/step while
free-falling and > 2 s/step once in contact with the 10k-vertex body; it is frame/cache based and its
colliders cannot grab. The XPBD solver: stretch + bending constraints, tethers to pinned regions (no
over-stretch), pin attachment to the driver, collision with the garment's own skin (half-space) and
11 body capsules sized from the personalised mesh, hand spheres (swept, so a hand cannot tunnel
through) and pinch-grab, teleport detection (tracking glitches snap instead of exploding), bounded
memory. Not implemented: cloth self-collision.

## Using it (3D View sidebar > VTryOn)

Mirror Mode toggle; category + garment list with preview; Wear / Remove; Edit (name, category, fit,
fabric: mass, stretch, bend, damping, friction - applied live); Reprocess (from the original, keeps
edits); Delete (dialog: confirm + separate "also delete the original upload"); Upload Clothing (file
browser + category/fit) -> processing box with warnings/errors -> Add to Clothing Library / Cancel;
Refresh. Pinch (thumb tip to index tip) grabs fabric, opening the fingers releases it.

Setup: `python scripts/seed_garments.py` once, then the dashboard's "Open Digital Twin in Blender".

## Fitting

* Units fixed by powers of 10 (cm/mm files), lying garments stood up; category from the name, else
  geometry (two leg openings = lower; sleeve span / height); always editable.
* Placement: garments authored on this body are kept as authored; others are aligned to the neck /
  waist and centred on the torso surface (warning shown).
* A-pose garments: sleeve angle from the sleeve centre line; the body is posed to match for fitting and
  binding.
* Conform (no uniform scaling): only vertices inside the body or closer than the fit-style ease (tight
  4 mm, regular 12 mm, loose 25 mm, oversized 45 mm) move; the displacement is smoothed into the 2-ring,
  so the design is otherwise untouched. Fails (not added) if > 3% stays inside the body.
* Validation: unsupported / empty / corrupt files, degenerate triangles, non-manifold edges, missing
  textures (warnings), > 6000 vertices (warning: too heavy for live cloth), missing attachment region
  (error). Photos are rejected: they need a separate 3D reconstruction step.

## Test results (actual)

| Test | Result |
|---|---|
| Cloth solver: drape / settle | 4.0% max stretch, settles to 0.001 m/s |
| Cloth solver: torso collision | no particle inside (min 12.4 cm for a 12 cm torso) |
| Cloth solver: hand push / grab / release | 3.8 cm local dent; grabbed patch +14.4 cm, far fabric 0.0 cm; back to rest |
| Cloth solver: jump / 1 m tracking glitch | hem lag 8.9 cm, settles; glitch -> 4% stretch, stable |
| Upload processing (T-shirt, trousers, cm-scaled OBJ) | 2.2-4.9 s each in a separate process |
| Worn T-shirt, standing / arms raised | 0.9% / 0.0% of vertices inside the body, 1.2 cm off the skin |
| Arms hanging against the body | 3.7% pinched in the armpit (SMPL-X skin self-intersects there) |
| Jump (25 cm) | hem lag 2.5 cm, lands with 0.9% inside |
| New body shape | garment refits, 0.2% inside |
| Hand push / pinch-grab / release in Blender | 2.4 cm local dent; grabbed 10.5 cm, rest 0.17 cm; settles back |
| Switch T-shirt -> trousers, remove | 0.7% inside; no objects left behind |
| Mirror Mode render, no garment vs camera image | median difference 0.000, p99 0.004 (avatar invisible) |
| Mirror Mode render with T-shirt | only the shirt area (4.6% of pixels) differs |
| Live viewport (Material Preview, GUI) | camera image + shirt, avatar invisible; Blender tick ~30 ms, cloth 104-160 ms/step (~7-9 Hz) |
| Library: approve / edit / delete / photo / unsupported | all behave as specified; originals kept unless asked |

## Limitations / next steps

1. Cloth runs at ~7-10 Hz on this CPU-only laptop while tracking + EEVEE share the CPU. Next: decimated
   simulation proxy driving the full mesh, or a GPU/compiled solver (numba/taichi) on the RTX machine.
2. No cloth self-collision; armpit pinch when the arms hang against the body.
3. Test garments are body copies (the trousers show the body at the crotch/inner knee); real garments
   should be authored on the base body in CLO3D. Uploaded garments with very different shapes may need
   manual category/placement correction (the preview + warnings flag it).
4. Hands interact through the twin's tracked hand joints (selected person only); force is not
   measurable from a camera - contact is position/velocity based. Pinch is the only grab gesture.
5. The customer's real clothes are not hidden yet (segmentation + inpainting is the next roadmap item).
6. MakeHuman: not in the repo; everything runs on the SMPL-X twin.
