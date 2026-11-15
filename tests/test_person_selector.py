"""Single-person selection state machine: the spec's multi-person scenarios with synthetic
detections + frames (each person wears a differently coloured shirt). Run: python tests/test_person_selector.py"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.camera.person_tracker import (PersonSelector, Person, LOCKED, SELECTING, TEMPORARILY_LOST,  # noqa: E402
                                       REACQUIRING, LOST, DETECTING, RESET)

SHIRTS = {"A": (40, 40, 200), "B": (200, 60, 40), "C": (40, 180, 40), "D": (200, 200, 200)}  # BGR


def frame_of(people):
    """people: list of (id, shirt, bbox) -> BGR frame with each person's shirt painted in their box."""
    f = np.full((480, 640, 3), 90, np.uint8)
    for _, shirt, (x0, y0, x1, y1) in people:
        f[int(y0):int(y1), int(x0):int(x1)] = SHIRTS[shirt]
    return f


def step(sel, t, people):
    dets = [Person(i, b, 0.9, t) for i, _, b in people]
    return sel.update(dets, now=t, frame=frame_of(people))


log = []


def check(name, cond):
    log.append(("PASS" if cond else "FAIL") + "  " + name)
    assert cond, name


# 1. one person -> auto-selected (configurable)
s = PersonSelector(auto_select_single=True)
r = step(s, 0.0, [(1, "A", (250, 80, 390, 470))])
check("one person: auto-selected", r.state == LOCKED and r.target_id == 1)
s2 = PersonSelector(auto_select_single=False)
check("one person, auto off: waits for selection", step(s2, 0.0, [(1, "A", (250, 80, 390, 470))]).state == SELECTING)

# 2. three people -> never auto-select; select the middle one by tapping it
s = PersonSelector()
three = [(1, "A", (20, 80, 170, 470)), (2, "B", (245, 80, 395, 470)), (3, "C", (470, 80, 620, 470))]
r = step(s, 0.0, three)
check("three people: SELECTING, nobody chosen", r.state == SELECTING and r.target_id is None)
s.select_at(320, 300)
r = step(s, 0.1, three)
check("tap middle person -> LOCKED on ID 2", r.state == LOCKED and r.target_id == 2)

# 3. others move / walk closer (bigger box) while the target stands still -> target unchanged
t = 0.2
for k in range(10):
    t += 0.1
    moved = [(1, "A", (20 + 8 * k, 60, 200 + 12 * k, 479)), (2, "B", (245, 80, 395, 470)),
             (3, "C", (430 - 10 * k, 30, 639, 479))]
    r = step(s, t, moved)
check("others move / come closer: still ID 2", r.state == LOCKED and r.target_id == 2)
check("pose box is the target's box", r.bbox == (245, 80, 395, 470))

# 4. target moves, others still
for k in range(5):
    t += 0.1
    r = step(s, t, [(1, "A", (20, 80, 170, 470)), (2, "B", (245 + 10 * k, 80, 395 + 10 * k, 470)),
                    (3, "C", (470, 80, 620, 470))])
check("target moves: follows ID 2", r.state == LOCKED and r.bbox[0] == 285)

# 5. brief occlusion (same ID returns) -> TEMPORARILY_LOST then LOCKED, no switch
others = [(1, "A", (20, 80, 170, 470)), (3, "C", (470, 80, 620, 470))]
t += 0.3
r = step(s, t, others)
check("brief occlusion: TEMPORARILY_LOST, pose held (no box)", r.state == TEMPORARILY_LOST and r.bbox is None)
t += 0.3
r = step(s, t, others + [(2, "B", (285, 80, 435, 470))])
check("same ID back: LOCKED on ID 2", r.state == LOCKED and r.target_id == 2)

# 6. longer occlusion, ByteTrack gives the target a NEW id -> re-acquired by position + clothing
last = (285, 80, 435, 470)
t += 1.5
r = step(s, t, others)
check("long occlusion: REACQUIRING", r.state == REACQUIRING)
t += 0.2
r = step(s, t, others + [(7, "B", (290, 85, 440, 470))])
check("new ID 7 at same place + same shirt: re-acquired", r.state == LOCKED and r.target_id == 7)

# 7. impostor: target leaves; a person seen together with the target walks into its spot -> NOT taken
t += 1.5
r = step(s, t, [(1, "A", (290, 85, 440, 470)), (3, "C", (470, 80, 620, 470))])
check("co-present person (ID 1) steps into target's place: not selected", r.target_id == 7 and r.state == REACQUIRING)
# a brand-new person with different clothes in that spot -> not taken either
t += 0.2
r = step(s, t, [(9, "D", (290, 85, 440, 470)), (3, "C", (470, 80, 620, 470))])
check("new person, different shirt, same spot: not selected", r.state == REACQUIRING and r.target_id == 7)

# 8. target gone past timeout -> LOST, tracking paused; new people do NOT get auto-selected
t += 5.0
r = step(s, t, [(9, "D", (290, 85, 440, 470))])
check("timeout: LOST, needs reselection", r.state == LOST and r.bbox is None)
t += 0.5
r = step(s, t, [(9, "D", (290, 85, 440, 470))])
check("single new person while LOST: still LOST (no silent switch)", r.state == LOST)

# 9. explicit selection of another person
s.select(9)
t += 0.1
r = step(s, t, [(9, "D", (290, 85, 440, 470))])
check("explicit select ID 9: LOCKED", r.state == LOCKED and r.target_id == 9)

# 10. reset -> RESET then SELECTING with several people
s.reset()
t += 0.1
check("reset: RESET", step(s, t, three).state == RESET)
t += 0.1
check("after reset with 3 people: SELECTING", step(s, t, three).state == SELECTING)
t += 0.1
check("nobody in view: DETECTING", step(PersonSelector(), t, []).state == DETECTING)

# 11. silhouette masking (dashboard screenshot case): a bystander's raised hand INSIDE the selected
#     person's box must be blacked out; the selected person's own body must stay visible.
from src.camera.person_tracker import mask_person  # noqa: E402
img = np.full((480, 640, 3), 200, np.uint8)
target = Person(1, (125, 250, 580, 670), 0.9, 0, np.array([[250, 260], [460, 260], [580, 479], [125, 479]], float))
bystander = Person(6, (12, 330, 245, 515), 0.9, 0,
                   np.array([[12, 330], [245, 330], [245, 515], [12, 515]], float))  # arm+hand reaching in
m = mask_person(img, target, [target, bystander])
check("bystander's hand inside target box is masked", m[350, 170].sum() == 0 and m[400, 150].sum() == 0)
check("target's own torso stays visible", m[400, 350].sum() > 0)
check("everything far outside target is masked", m[50, 600].sum() == 0)

print("\n".join(log))
print(f"PERSON SELECTOR OK ({len(log)} checks)")
