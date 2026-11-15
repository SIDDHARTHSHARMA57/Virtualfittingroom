"""Cloth solver behaviour without Blender: gravity drape, settling, body collision, hand push,
hand grab (local pull) + release, jump/landing inertia, fast-motion stability, speed.
Run: python tests/test_cloth_sim.py"""
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.garments.cloth_sim import ClothSim  # noqa: E402


def panel(nx=24, ny=30, w=0.5, h=0.6, y0=0.15, z_top=1.4):
    """Vertical cloth panel in the x-z plane at y = y0 (in front of a body at the origin), Z up."""
    xs, zs = np.linspace(-w / 2, w / 2, nx), np.linspace(z_top, z_top - h, ny)
    X, Z = np.meshgrid(xs, zs)
    P = np.stack([X.ravel(), np.full(X.size, y0), Z.ravel()], 1)
    F = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            F += [[a, a + 1, a + nx], [a + 1, a + nx + 1, a + nx]]
    return P, np.array(F), nx, ny


def torso_capsule(y=0.0, r=0.12):
    return (np.array([[0, y, 0.9]]), np.array([[0, y, 1.45]]), np.array([r]))


log = []


def check(name, cond, info=""):
    log.append(("PASS" if cond else "FAIL") + f"  {name}  {info}")
    assert cond, name + " " + info


P, F, nx, ny = panel()
top = np.zeros(len(P))
top[:nx * 2] = 1.0                    # shoulder band pinned
dt = 1 / 15                           # live inference rate

# 1. gravity drape on a pinned band: hangs, does not stretch apart, settles
sim = ClothSim(P, F, top)
for _ in range(45):
    x = sim.step(dt, P)
stretch = np.linalg.norm(x[sim.edges[:, 1]] - x[sim.edges[:, 0]], axis=1) / sim.L0
speed = np.linalg.norm(sim.v, axis=1).max()
check("drape: no tearing/over-stretch", stretch.max() < 1.08, f"max edge stretch {100 * (stretch.max() - 1):.1f}%")
check("drape: settles after 3 s", speed < 0.05, f"max speed {speed:.3f} m/s")

# 2. body collision: panel sits in front of a torso; push the target INTO the body -> cloth stays out
cap = torso_capsule()
sim = ClothSim(P, F, np.ones(len(P)) * 0.3)
inside_target = P.copy()
inside_target[:, 1] = 0.02            # target behind the body surface (r = 0.12)
for _ in range(30):
    x = sim.step(dt, inside_target, capsules=cap, ease=0.0)
r_xy = np.linalg.norm(x[:, :2] - [0, 0], axis=1)
in_band = (x[:, 2] > 0.9) & (x[:, 2] < 1.45)
check("collision: no particle inside the torso", (r_xy[in_band] >= 0.12 - 1e-6).all(),
      f"min radius {r_xy[in_band].min():.4f} m (torso 0.12, cloth thickness 0.004)")

# 3. hand push: a sphere moving into the panel dents it locally, never passes through
sim = ClothSim(P, F, np.ones(len(P)) * 0.4)
for _ in range(10):
    sim.step(dt, P)
for k in range(10):   # hand ends ~2 cm behind the cloth's rest plane (a shirt sits ~1.2 cm off the body)
    c = np.array([[0.0, 0.25 - 0.012 * k, 1.15]])
    x = sim.step(dt, P, hands={"right": (c, np.array([0.04]))})
d_hand = np.linalg.norm(x - c[0], axis=1)
check("push: fabric never inside the hand", d_hand.min() >= 0.04 - 1e-6, f"closest {d_hand.min():.4f} m")
dent = P[:, 1] - x[:, 1]
near, far = np.linalg.norm(P[:, [0, 2]] - [0.0, 1.15], axis=1) < 0.05, np.linalg.norm(P[:, [0, 2]] - [0.0, 1.15], axis=1) > 0.2
check("push: local dent", dent[near].mean() > 0.02 and abs(dent[far]).mean() < 0.01,
      f"dent near hand {100 * dent[near].mean():.1f} cm, far {100 * abs(dent[far]).mean():.2f} cm")

# 4. grab + pull + release: local deformation, then recovery
sim = ClothSim(P, F, np.ones(len(P)) * 0.15)
for _ in range(20):
    sim.step(dt, P)
rest_x = sim.x.copy()
grab_at = np.array([0.0, 0.15, 1.05])
n = sim.grab("right", grab_at, 0.04)
for k in range(12):
    hand = grab_at + [0, 0.012 * (k + 1), 0]       # pull 14 cm towards the camera
    x = sim.step(dt, P, hands={"right": (hand[None], np.array([0.0]))})
moved = np.linalg.norm(x - rest_x, axis=1)
dist = np.linalg.norm(rest_x - grab_at, axis=1)
check("grab: grabbed patch follows the hand", moved[dist < 0.03].mean() > 0.12, f"{n} particles, moved {100 * moved[dist < 0.03].mean():.1f} cm")
check("grab: deformation is local", moved[dist > 0.25].mean() < 0.3 * moved[dist < 0.03].mean(),
      f"far region moved {100 * moved[dist > 0.25].mean():.1f} cm")
sim.release("right")
for _ in range(60):
    x = sim.step(dt, P)
back = np.linalg.norm(x - rest_x, axis=1).max()
check("release: fabric settles back", back < 0.02, f"max offset after 4 s {100 * back:.2f} cm")

# 5. jump: pinned band goes up 20 cm and lands; free hem lags (inertia) then settles, no explosion
sim = ClothSim(P, F, top)
for _ in range(30):
    sim.step(dt, P)
hem = np.arange(len(P)) >= len(P) - nx
lag, peak = [], []
for k in range(14):
    lift = 0.2 * np.sin(np.pi * k / 13)
    T = P + [0, 0, lift]
    x = sim.step(dt, T)
    lag.append((T[hem, 2] - x[hem, 2]).mean())
for _ in range(45):
    x = sim.step(dt, P)
settle = np.linalg.norm(sim.v, axis=1).max()
check("jump: hem lags the body (inertia)", max(np.abs(lag)) > 0.02, f"max hem lag {100 * max(np.abs(lag)):.1f} cm")
check("jump: lands and settles", settle < 0.05 and np.isfinite(x).all(), f"max speed 3 s after landing {settle:.3f} m/s")

# 6. tracking glitch: target teleports 1 m for one frame -> no NaN / explosion
sim = ClothSim(P, F, top)
sim.step(dt, P)
sim.step(dt, P + [1.0, 0, 0])
for _ in range(30):
    x = sim.step(dt, P)
st = np.linalg.norm(x[sim.edges[:, 1]] - x[sim.edges[:, 0]], axis=1) / sim.L0
check("glitch: stable after a 1 m tracking jump", np.isfinite(x).all() and st.max() < 1.15, f"max stretch {100 * (st.max() - 1):.1f}%")

# 7. speed on a T-shirt-sized mesh (~3.5k vertices) with 12 body capsules + 2 hands
P2, F2, *_ = panel(nx=60, ny=58, w=1.0, h=1.0)
sim = ClothSim(P2, F2, np.ones(len(P2)) * 0.3)
caps = (np.random.rand(12, 3), np.random.rand(12, 3), np.full(12, 0.05))
hands = {"left": (np.random.rand(6, 3), np.full(6, 0.02)), "right": (np.random.rand(6, 3), np.full(6, 0.02))}
sim.step(dt, P2, caps, hands)
t = time.perf_counter()
for _ in range(10):
    sim.step(dt, P2, caps, hands)
ms = (time.perf_counter() - t) / 10 * 1000
check("speed: 3.5k-vertex garment step", ms < 150, f"{ms:.0f} ms per step ({len(P2)} verts, 3 substeps x 6 iterations)")

print("\n".join(log))
print("CLOTH SIM OK")
