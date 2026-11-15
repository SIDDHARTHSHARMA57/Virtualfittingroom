"""
Multi-person detection (YOLO11n + ByteTrack via ultralytics) and single-person selection.

Only the selected person drives the digital twin. No face recognition and nothing persistent:
IDs are ByteTrack's temporary track IDs; re-acquisition uses position + a clothing-colour histogram
held in memory for the current session only.

States
  IDLE              service not started
  DETECTING         nobody in view
  SELECTING         people in view, nobody selected (operator/customer must choose)
  LOCKED            selected person visible -> pose estimation on that person only
  TEMPORARILY_LOST  selected person missing <= lost_hold_s -> last pose is held
  REACQUIRING       missing <= reacquire_timeout_s -> waiting for that person to reappear
  LOST              timed out -> tracking paused, explicit reselection required
  RESET             selection cleared by the operator (transient)
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

IDLE, DETECTING, SELECTING, LOCKED = "IDLE", "DETECTING", "SELECTING", "LOCKED"
TEMPORARILY_LOST, REACQUIRING, LOST, RESET = "TEMPORARILY_LOST", "REACQUIRING", "LOST", "RESET"


@dataclass
class Person:
    track_id: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    conf: float
    last_seen: float = 0.0
    polygon: Optional[np.ndarray] = None     # silhouette outline (N,2) px, from the -seg model

    @property
    def center(self) -> Tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2


@dataclass
class Selection:
    state: str
    target_id: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None   # where to run pose estimation
    message: str = ""
    people: List[Person] = field(default_factory=list)


def iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def appearance(frame: Optional[np.ndarray], bbox) -> Optional[np.ndarray]:
    """Clothing-colour signature: normalised hue/saturation histogram of the torso region
    (middle of the box; excludes the head). Not biometric, kept in memory only."""
    if frame is None:
        return None
    import cv2
    x0, y0, x1, y1 = bbox
    h, w = y1 - y0, x1 - x0
    r = (int(max(0, x0 + 0.2 * w)), int(max(0, y0 + 0.25 * h)), int(min(frame.shape[1], x1 - 0.2 * w)),
         int(min(frame.shape[0], y0 + 0.65 * h)))
    if r[2] - r[0] < 4 or r[3] - r[1] < 4:
        return None
    hsv = cv2.cvtColor(frame[r[1]:r[3], r[0]:r[2]], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
    return hist / (hist.sum() + 1e-9)


def similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Histogram intersection in [0, 1]; 1.0 when either is unknown (no evidence against)."""
    if a is None or b is None:
        return 1.0
    return float(np.minimum(a, b).sum())


class PersonSelector:
    def __init__(self, auto_select_single: bool = True, lost_hold_s: float = 0.7,
                 reacquire_timeout_s: float = 5.0, reacquire_iou: float = 0.15,
                 appearance_min: float = 0.55):
        self.auto_select_single = auto_select_single
        self.lost_hold_s = lost_hold_s
        self.reacquire_timeout_s = reacquire_timeout_s
        self.reacquire_iou = reacquire_iou
        self.appearance_min = appearance_min
        self.state = IDLE
        self.target_id: Optional[int] = None
        self.last_bbox = None
        self.last_seen = 0.0
        self.signature = None
        self._co_present: set = set()   # IDs seen together with the target = definitely other people
        self._pending: Optional[int] = None
        self._pending_point: Optional[Tuple[float, float]] = None
        self._reset_requested = False
        self.people: List[Person] = []

    # ---- operator / customer actions (thread-safe enough: plain attribute writes) ----
    def select(self, track_id: int):
        self._pending = int(track_id)

    def select_at(self, x: float, y: float):
        """Touch/click at pixel (x, y): selects the (smallest) box containing it on the next update."""
        self._pending_point = (float(x), float(y))

    def reset(self):
        self._reset_requested = True

    # ---- per frame ----
    def update(self, people: List[Person], now: Optional[float] = None,
               frame: Optional[np.ndarray] = None) -> Selection:
        now = time.time() if now is None else now
        self.people = people
        by_id = {p.track_id: p for p in people}

        if self._reset_requested:
            self._reset_requested = False
            self._unlock()
            self.state = RESET
            return Selection(RESET, message="Selection cleared", people=people)

        if self._pending_point is not None:
            x, y = self._pending_point
            self._pending_point = None
            hits = [p for p in people if p.bbox[0] <= x <= p.bbox[2] and p.bbox[1] <= y <= p.bbox[3]]
            if hits:
                self._pending = min(hits, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1])).track_id
        if self._pending is not None:
            pid, self._pending = self._pending, None
            if pid in by_id:
                self._lock(by_id[pid], now, frame)

        if self.target_id is None:
            if self.state == LOST and people:
                return Selection(LOST, message="Selected person lost - tap a person to select", people=people)
            if not people:
                self.state = DETECTING if self.state != LOST else LOST
                return Selection(self.state, message="Waiting for a person", people=people)
            if len(people) == 1 and self.auto_select_single and self.state != LOST:
                self._lock(people[0], now, frame)
            else:
                self.state = SELECTING
                return Selection(SELECTING, message=f"{len(people)} people - tap one to select", people=people)

        tgt = by_id.get(self.target_id)
        if tgt is not None:
            self._on_seen(tgt, now, frame, people)
            return Selection(LOCKED, self.target_id, tgt.bbox, people=people)

        missing = now - self.last_seen
        if missing > self.reacquire_timeout_s:
            self._unlock()
            self.state = LOST
            return Selection(LOST, message="Selected person lost - tap a person to select", people=people)
        cand = self._reacquire_candidate(people, frame, missing)
        if cand is not None:
            self.target_id = cand.track_id
            self._on_seen(cand, now, frame, people)
            return Selection(LOCKED, self.target_id, cand.bbox, message="Re-acquired", people=people)
        self.state = TEMPORARILY_LOST if missing <= self.lost_hold_s else REACQUIRING
        return Selection(self.state, self.target_id, None, message="Selected person not visible", people=people)

    # ---- internals ----
    def _lock(self, p: Person, now: float, frame):
        self.target_id, self.state = p.track_id, LOCKED
        self.last_bbox, self.last_seen = p.bbox, now
        self.signature = appearance(frame, p.bbox)
        self._co_present = set()

    def _unlock(self):
        self.target_id, self.last_bbox, self.signature = None, None, None
        self._co_present = set()

    def _on_seen(self, p: Person, now: float, frame, people):
        self.state = LOCKED
        self.last_bbox, self.last_seen = p.bbox, now
        self._co_present |= {q.track_id for q in people if q.track_id != p.track_id}
        sig = appearance(frame, p.bbox)
        if sig is not None:
            self.signature = sig if self.signature is None else 0.9 * self.signature + 0.1 * sig

    def _reacquire_candidate(self, people, frame, missing: float) -> Optional[Person]:
        """A NEW track ID (ByteTrack gives a fresh ID after a long occlusion) that overlaps where the
        target was and looks like it. Never someone seen at the same time as the target; if more
        than one candidate qualifies, nobody is chosen (no silent switching)."""
        ok = []
        for p in people:
            if p.track_id in self._co_present:
                continue
            if iou(p.bbox, self.last_bbox) < self.reacquire_iou:
                continue
            if similarity(self.signature, appearance(frame, p.bbox)) < self.appearance_min:
                continue
            ok.append(p)
        return ok[0] if len(ok) == 1 else None


class PersonDetector:
    """YOLO11n person detection + ByteTrack identities (ultralytics). With a -seg model each person
    also gets a silhouette polygon (~45 ms/frame at imgsz 320 on an i7-10th-gen CPU; ~38 ms without)."""

    def __init__(self, model_path: str, imgsz: int = 320, conf: float = 0.4, tracker: str = "bytetrack.yaml"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.imgsz, self.conf, self.tracker = imgsz, conf, tracker
        self.last_ms = 0.0

    def detect(self, frame_bgr: np.ndarray, now: Optional[float] = None) -> List[Person]:
        t0 = time.perf_counter()
        res = self.model.track(frame_bgr, persist=True, classes=[0], imgsz=self.imgsz, conf=self.conf,
                               tracker=self.tracker, verbose=False)[0]
        self.last_ms = (time.perf_counter() - t0) * 1000
        now = time.time() if now is None else now
        out = []
        if res.boxes is not None and res.boxes.id is not None:
            polys = res.masks.xy if res.masks is not None else [None] * len(res.boxes)
            for box, tid, c, poly in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy(),
                                         res.boxes.conf.cpu().numpy(), polys):
                out.append(Person(int(tid), tuple(float(v) for v in box), float(c), now,
                                  poly if poly is not None and len(poly) >= 3 else None))
        return out


def mask_person(frame: np.ndarray, target: Person, others: List[Person], grow: float = 0.04) -> np.ndarray:
    """Keep only the selected person's silhouette (dilated by `grow` x frame width so limb edges and
    fast motion survive) and black out every other person's silhouette, even inside the target's
    box (a bystander's hand reaching across). Falls back to the box when there is no silhouette."""
    import cv2
    h, w = frame.shape[:2]
    if target.polygon is None:
        keep = np.zeros((h, w), np.uint8)
        x0, y0, x1, y1 = (int(v) for v in target.bbox)
        keep[max(0, y0):y1, max(0, x0):x1] = 1
        own = keep.copy()
    else:
        own = np.zeros((h, w), np.uint8)
        cv2.fillPoly(own, [target.polygon.astype(np.int32)], 1)
        k = max(3, int(grow * w) | 1)
        keep = cv2.dilate(own, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    for p in others:
        if p.track_id == target.track_id or p.polygon is None:
            continue
        theirs = np.zeros((h, w), np.uint8)
        cv2.fillPoly(theirs, [p.polygon.astype(np.int32)], 1)
        keep[(theirs == 1) & (own == 0)] = 0   # their pixels, unless the target's silhouette is there
    return frame * keep[:, :, None]
