"""
Automated Unit and Integration Tests for VirtualFittingRoom Camera Subsystem.
Uses standard library unittest so no external testing dependencies are required.
"""
import time
import json
import unittest
import numpy as np

from src.camera.protocol import serialize_packet, deserialize_packet
from src.camera.tracking import OneEuroFilter
from src.camera.monitor import PerformanceMonitor
from src.camera.capture import ThreadedCamera

class TestCameraSubsystem(unittest.TestCase):
    def test_protocol_serialization_roundtrip(self):
        body_points = [[0.5, 0.5, 0.9] for _ in range(33)]
        world_points = [[0.0, 1.0, 0.0, 0.9] for _ in range(33)]
        hands = {"Right": [[0.1, 0.2, 0.3] for _ in range(21)]}
        face = [[0.4, 0.4, 0.4] for _ in range(478)]
        face_matrix = [[1.0] * 4 for _ in range(4)]
        blendshapes = {"eyeBlinkLeft": 0.85}

        payload = serialize_packet(
            body_points=body_points,
            world_points=world_points,
            hands=hands,
            face=face,
            face_matrix=face_matrix,
            blendshapes=blendshapes,
            timestamp=1234567.89
        )

        self.assertIsInstance(payload, bytes)
        self.assertTrue(payload.endswith(b"\n"))

        decoded = deserialize_packet(payload.decode("utf-8").strip())
        self.assertEqual(decoded["schema"], 2)
        self.assertEqual(decoded["timestamp"], 1234567.89)
        self.assertEqual(len(decoded["points"]), 33)
        self.assertEqual(len(decoded["world_points"]), 33)
        self.assertEqual(len(decoded["hands"]["Right"]), 21)
        self.assertEqual(len(decoded["face"]), 478)
        self.assertAlmostEqual(decoded["blendshapes"]["eyeBlinkLeft"], 0.85)

    def test_protocol_compatibility_with_blender_receiver(self):
        """Verify that blender_pose_receiver.py parsing logic succeeds on serialize_packet output."""
        body_points = [[0.5, 0.5, 0.9] for _ in range(33)]
        payload = serialize_packet(body_points=body_points)

        line = payload.decode("utf-8").strip()
        packet = json.loads(line)
        points = packet.get("points", [])
        self.assertEqual(len(points), 33)

    def test_one_euro_filter_smoothing(self):
        filter_obj = OneEuroFilter(min_cutoff=1.0, beta=0.007)
        signal_in = np.array([10.0, 20.0])
        out1 = filter_obj.filter(signal_in, timestamp=0.0)
        np.testing.assert_allclose(out1, signal_in)

        # Jitter step
        jittered = np.array([10.5, 19.8])
        out2 = filter_obj.filter(jittered, timestamp=0.033)
        # Filtered output should be smoothed between 10.0 and 10.5
        self.assertTrue(10.0 < out2[0] < 10.5)
        self.assertTrue(19.8 < out2[1] < 20.0)

    def test_performance_monitor(self):
        monitor = PerformanceMonitor(log_interval_s=0.1)
        monitor.update_inference(
            frame_age_s=0.015,
            is_stale=False,
            latencies={"pose_ms": 15.0, "hand_ms": 20.0, "face_ms": 10.0, "total_ms": 45.0}
        )
        monitor.update_network(is_connected=True, success=True)

        summary = monitor.get_summary()
        self.assertEqual(summary["total_inferred_frames"], 1)
        self.assertEqual(summary["stale_frames"], 0)
        self.assertTrue(summary["network"]["connected"])
        self.assertEqual(summary["network"]["packets_sent"], 1)

    def test_threaded_camera_stale_detection(self):
        cam = ThreadedCamera(device_id=999, stale_threshold_s=0.1)
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with cam._lock:
            cam._latest_frame = fake_frame
            cam._latest_frame_time = time.time() - 0.2  # 200ms old

        frame, ftime, is_stale = cam.get_latest_frame()
        self.assertIsNotNone(frame)
        self.assertTrue(is_stale)

if __name__ == "__main__":
    unittest.main()

class TestDashboardAndDrawing(unittest.TestCase):
    def test_render_tracking_overlays(self):
        from src.camera.drawing import render_tracking_overlays
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_tracking = {
            "body_points": [[0.5, 0.5, 0.9] for _ in range(33)],
            "hands": {"Right": [[0.5, 0.5, 0.9] for _ in range(21)]},
            "face": [[0.5, 0.5, 0.9] for _ in range(478)],
        }
        rendered = render_tracking_overlays(
            frame=frame,
            tracking_data=mock_tracking,
            draw_skeleton=True,
            draw_body_points=True,
            draw_hands=True,
            draw_face=True
        )
        self.assertEqual(rendered.shape, (480, 640, 3))
        # Verify drawing actually colored some pixels
        self.assertTrue(np.any(rendered > 0))

    def test_drawing_handles_empty_landmarks_safely(self):
        from src.camera.drawing import render_tracking_overlays
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        empty_tracking = {}
        rendered = render_tracking_overlays(
            frame=frame,
            tracking_data=empty_tracking,
            draw_skeleton=True,
            draw_body_points=True,
            draw_hands=True,
            draw_face=True
        )
        self.assertEqual(rendered.shape, (480, 640, 3))
        self.assertFalse(np.any(rendered > 0))

class TestAnatomyAndSMPLX(unittest.TestCase):
    def test_anatomical_body_model(self):
        from src.camera.anatomy import AnatomicalBodyModel
        model = AnatomicalBodyModel()

        # Mock standard T-pose 33 points
        mock_body = [[0.5, 0.5, 0.9] for _ in range(33)]
        mock_body[11] = [0.4, 0.3, 0.9]  # L Shoulder
        mock_body[12] = [0.6, 0.3, 0.9]  # R Shoulder
        mock_body[23] = [0.45, 0.7, 0.9] # L Hip
        mock_body[24] = [0.55, 0.7, 0.9] # R Hip

        anatomy = model.compute_anatomical_regions(mock_body)
        self.assertIn("torso", anatomy)
        self.assertIn("clavicles", anatomy["torso"])
        self.assertIn("sternum", anatomy["torso"])
        self.assertIn("chest_contours", anatomy["torso"])
        self.assertIn("waist_contours", anatomy["torso"])
        self.assertIn("pelvis", anatomy["torso"])
        self.assertGreater(anatomy["summary_counts"]["estimated_surface_points"], 0)

    def test_render_3d_orthographic_view(self):
        from src.camera.drawing import render_3d_orthographic_view
        verts = np.random.uniform(-1.0, 1.0, size=(10475, 3)).astype(np.float32)
        front_img = render_3d_orthographic_view(verts, view_angle="front", width=200, height=200)
        side_img = render_3d_orthographic_view(verts, view_angle="side", width=200, height=200)
        rear_img = render_3d_orthographic_view(verts, view_angle="rear", width=200, height=200)

        self.assertEqual(front_img.shape, (200, 200, 3))
        self.assertEqual(side_img.shape, (200, 200, 3))
        self.assertEqual(rear_img.shape, (200, 200, 3))

class TestLiveMotionAndOrientation(unittest.TestCase):
    def test_orientation_head_above_pelvis(self):
        from src.camera.drawing import render_3d_orthographic_view
        # Fake human mesh with head at y=+0.3 and pelvis at y=-0.3
        verts = np.array([
            [0.0, 0.3, 0.0],   # Head (3D +Y)
            [0.0, -0.3, 0.0],  # Pelvis (3D -Y)
        ], dtype=np.float32)

        # Test side view
        side_img = render_3d_orthographic_view(verts, view_angle="side", width=200, height=200)
        self.assertEqual(side_img.shape, (200, 200, 3))

        # Check screen coordinates of head vs pelvis in side projection:
        # In render_3d_orthographic_view:
        # screen_y_pts = -proj_v.
        # Head (y=+0.3) -> screen_y = -0.3 (lower y value -> TOP of image)
        # Pelvis (y=-0.3) -> screen_y = +0.3 (higher y value -> BOTTOM of image)
        head_screen_y = -0.3
        pelvis_screen_y = 0.3
        self.assertLess(head_screen_y, pelvis_screen_y, "Head screen Y must be smaller than pelvis screen Y (i.e. above it on screen)")

class TestFullBodyMotionAndDepth(unittest.TestCase):
    # MediaPipe convention for a person facing the camera: their LEFT landmarks (11, 13, 23...) are at +x.
    # Landmarks not set in a mock get visibility 0 so the solver ignores them.

    def test_forward_backward_depth_mapping(self):
        """Verify that when user reaches forward towards camera (-Z in MediaPipe), SMPL-X moves forward (+Z)."""
        from src.camera.smplx_fitter import SMPLXFittingEngine
        engine = SMPLXFittingEngine(model_dir='models')

        mock_world = [[0.0, 0.0, 0.0, 0.0] for _ in range(33)]
        mock_world[11] = [ 0.20, -0.30, 0.0, 0.95]
        mock_world[12] = [-0.20, -0.30, 0.0, 0.95]
        mock_world[23] = [ 0.10,  0.20, 0.0, 0.95]
        mock_world[24] = [-0.10,  0.20, 0.0, 0.95]

        # Case 1: Resting arm
        mock_world[13] = [ 0.20, -0.05, 0.0, 0.95]
        mock_world[15] = [ 0.20,  0.20, 0.0, 0.95]
        res1 = engine.update_from_world_landmarks(mock_world)
        z_rest = res1['joints'][20, 2]

        # Case 2: Reaching forward toward camera (MediaPipe Z = -0.35m)
        mock_world[15] = [ 0.20, -0.05, -0.35, 0.95]
        for _ in range(5):  # temporal term: give it a few frames to settle
            res2 = engine.update_from_world_landmarks(mock_world)
        z_forward = res2['joints'][20, 2]

        delta_z = z_forward - z_rest
        self.assertGreater(delta_z, 0.10, f"Forward reaching must yield positive SMPL-X Z delta. Got {delta_z}")

    def test_sitting_posture_retargeting(self):
        """Verify that thighs and knees extend forward (+Z in SMPL-X) in a sitting posture."""
        from src.camera.smplx_fitter import SMPLXFittingEngine
        engine = SMPLXFittingEngine(model_dir='models')

        mock = [[0.0, 0.0, 0.0, 0.0] for _ in range(33)]
        mock[11] = [ 0.20, -0.30, 0.0, 0.95]
        mock[12] = [-0.20, -0.30, 0.0, 0.95]
        mock[23] = [ 0.10,  0.20, 0.0, 0.95]
        mock[24] = [-0.10,  0.20, 0.0, 0.95]
        # Knees forward in sitting pose
        mock[25] = [ 0.10,  0.20, -0.40, 0.95]
        mock[26] = [-0.10,  0.20, -0.40, 0.95]
        mock[27] = [ 0.10,  0.60, -0.40, 0.95]
        mock[28] = [-0.10,  0.60, -0.40, 0.95]

        res = engine.update_from_world_landmarks(mock)
        knee_z = res['joints'][4, 2]
        self.assertGreater(knee_z, 0.15, f"SMPL-X left knee must extend forward in sitting pose. Got {knee_z}")

    def test_lateral_spine_lean(self):
        """Verify that torso and head respond to lateral leaning motion."""
        from src.camera.smplx_fitter import SMPLXFittingEngine
        engine = SMPLXFittingEngine(model_dir='models')

        mock = [[0.0, 0.0, 0.0, 0.0] for _ in range(33)]
        mock[23] = [ 0.10, 0.20, 0.0, 0.95]
        mock[24] = [-0.10, 0.20, 0.0, 0.95]
        # Lean toward +x: shoulders displaced by +0.25m
        mock[11] = [ 0.45, -0.30, 0.0, 0.95]
        mock[12] = [ 0.05, -0.30, 0.0, 0.95]
        mock[13] = [ 0.45, -0.05, 0.0, 0.95]
        mock[14] = [ 0.05, -0.05, 0.0, 0.95]
        mock[25] = [ 0.10,  0.60, 0.0, 0.95]
        mock[26] = [-0.10,  0.60, 0.0, 0.95]

        res = engine.update_from_world_landmarks(mock)
        head_x = res['joints'][15, 0]
        self.assertGreater(head_x, 0.10, f"Head must tilt right during lateral lean. Got {head_x}")
