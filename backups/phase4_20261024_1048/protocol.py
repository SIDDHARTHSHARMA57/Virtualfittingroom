"""
Protocol serializer and deserializer for VirtualFittingRoom <-> Blender communication.
Preserves exact schema compatibility with blender_pose_receiver.py.
"""
from typing import Dict, Any, List, Optional
import json
import time

def serialize_packet(
    body_points: List[List[float]],
    world_points: Optional[List[List[float]]] = None,
    hands: Optional[Dict[str, List[List[float]]]] = None,
    face: Optional[List[List[float]]] = None,
    face_matrix: Optional[List[List[float]]] = None,
    blendshapes: Optional[Dict[str, float]] = None,
    timestamp: Optional[float] = None,
    schema: int = 2
) -> bytes:
    """
    Serializes tracking data into newline-delimited JSON payload expected by blender_pose_receiver.py.
    """
    packet = {
        "schema": schema,
        "timestamp": timestamp if timestamp is not None else time.time(),
        "points": body_points,
        "world_points": world_points if world_points is not None else [],
        "hands": hands if hands is not None else {},
        "face": face if face is not None else [],
        "face_matrix": face_matrix,
        "blendshapes": blendshapes if blendshapes is not None else {},
    }
    encoded = json.dumps(packet, separators=(",", ":")) + "\n"
    return encoded.encode("utf-8")

def deserialize_packet(line: str) -> Dict[str, Any]:
    """
    Deserializes a JSON line into a packet dictionary.
    """
    return json.loads(line)
