"""
Latest-frame handoff between the tracking service (writer) and Blender (reader) via named
shared memory. No disk writes, no queue: the reader always sees the newest frame only.

Layout: 64-byte header + BGR uint8 pixels (max 1920x1080).
  header = seq (u64), width (u32), height (u32), channels (u32), pad (u32), timestamp (f64)
Seqlock: the writer makes seq odd while copying and even when done; the reader retries if seq
changed or was odd, so it never displays a half-written frame. numpy + stdlib only
(Blender imports this module too).
"""
import os
import struct
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np

NAME = "vfr_camera_frame"
HEADER = struct.Struct("<QIIIId")
HEADER_SIZE = 64
MAX_W, MAX_H, CH = 1920, 1080, 3
SIZE = HEADER_SIZE + MAX_W * MAX_H * CH


class FrameWriter:
    def __init__(self, name: str = NAME):
        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=SIZE)
        except FileExistsError:  # left over from a previous run (or a reader still holds it)
            self.shm = shared_memory.SharedMemory(name=name)
        self.seq = HEADER.unpack_from(self.shm.buf, 0)[0] & ~1

    def write(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None):
        h, w = frame_bgr.shape[:2]
        if w > MAX_W or h > MAX_H or frame_bgr.ndim != 3 or frame_bgr.shape[2] != CH:
            raise ValueError(f"frame {frame_bgr.shape} exceeds {MAX_W}x{MAX_H}x{CH}")
        buf = self.shm.buf
        self.seq += 1  # odd: writing
        HEADER.pack_into(buf, 0, self.seq, w, h, CH, 0, timestamp or time.time())
        np.ndarray((h, w, CH), np.uint8, buffer=buf, offset=HEADER_SIZE)[:] = frame_bgr
        self.seq += 1  # even: complete
        HEADER.pack_into(buf, 0, self.seq, w, h, CH, 0, timestamp or time.time())

    def close(self, unlink: bool = True):
        try:
            self.shm.close()
            if unlink:
                self.shm.unlink()  # no-op semantics on Windows; frees the name on Linux/macOS
        except Exception:
            pass


class FrameReader:
    def __init__(self, name: str = NAME):
        self.name = name
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.last_seq = -1

    def _attach(self) -> bool:
        if self.shm is not None:
            return True
        try:
            self.shm = shared_memory.SharedMemory(name=self.name)
        except (FileNotFoundError, OSError):
            return False
        if os.name == "posix":  # Python < 3.13 would unlink the writer's block when we exit
            try:
                from multiprocessing import resource_tracker
                resource_tracker.unregister(self.shm._name, "shared_memory")
            except Exception:
                pass
        return True

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """Returns (new BGR frame or None if nothing new, frame timestamp)."""
        if not self._attach():
            return None, 0.0
        buf = self.shm.buf
        for _ in range(3):
            seq, w, h, ch, _, ts = HEADER.unpack_from(buf, 0)
            if seq & 1 or seq == self.last_seq or w == 0:
                if seq == self.last_seq:
                    return None, ts
                continue
            frame = np.ndarray((h, w, ch), np.uint8, buffer=buf, offset=HEADER_SIZE).copy()
            if HEADER.unpack_from(buf, 0)[0] == seq:
                self.last_seq = seq
                return frame, ts
        return None, 0.0

    def reset(self):
        """Drop the mapping (e.g. writer restarted) so the next read re-attaches."""
        if self.shm is not None:
            try:
                self.shm.close()
            except Exception:
                pass
        self.shm, self.last_seq = None, -1


if __name__ == "__main__":
    w = FrameWriter(NAME + "_selftest")
    r = FrameReader(NAME + "_selftest")
    img = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    w.write(img, 1.0)
    f, ts = r.read()
    assert f is not None and np.array_equal(f, img) and ts == 1.0
    assert r.read()[0] is None  # nothing new
    w.write(img[::-1].copy())
    assert np.array_equal(r.read()[0], img[::-1])
    r.reset()
    w.close()
    print("frame_share OK")
