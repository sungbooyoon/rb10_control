import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "rb10_control"
    / "joint_trajectory_controller.py"
)


def _quat_xyzw_to_rotmat(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("invalid quaternion")
    x, y, z, w = q / norm
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _install_fake_modules():
    class FakeLogger:
        def __init__(self):
            self.messages = []

        def info(self, msg):
            self.messages.append(("info", msg))

        def warn(self, msg):
            self.messages.append(("warn", msg))

        def error(self, msg):
            self.messages.append(("error", msg))

    class FakeClock:
        def now(self):
            return types.SimpleNamespace(nanoseconds=0)

    class FakePublisher:
        def __init__(self):
            self.messages = []

        def publish(self, msg):
            self.messages.append(msg)

    class FakeClient:
        def wait_for_service(self, timeout_sec=0.0):
            return True

        def call_async(self, req):
            future = types.SimpleNamespace()
            future.done = lambda: True
            future.exception = lambda: None
            future.result = lambda: types.SimpleNamespace(success=True)
            return future

    class FakeNode:
        def __init__(self, name):
            self._name = name
            self._logger = FakeLogger()
            self._clock = FakeClock()

        def create_subscription(self, *args, **kwargs):
            return object()

        def create_publisher(self, *args, **kwargs):
            return FakePublisher()

        def create_client(self, *args, **kwargs):
            return FakeClient()

        def get_logger(self):
            return self._logger

        def get_clock(self):
            return self._clock

        def destroy_node(self):
            return None

    class FakeDuration:
        def __init__(self, seconds=0.0):
            self.seconds = float(seconds)

        def to_msg(self):
            return types.SimpleNamespace(seconds=self.seconds)

    class FakeTracIK:
        instances = []
        next_result = np.zeros(6, dtype=float)
        next_exception = None
        next_fk_pos = np.zeros(3, dtype=float)
        next_fk_rot = np.eye(3, dtype=float)
        result_fn = None
        fk_fn = None

        def __init__(self, base_link_name, tip_link_name, urdf_path):
            self.base_link_name = base_link_name
            self.tip_link_name = tip_link_name
            self.urdf_path = urdf_path
            self.ik_calls = []
            self.__class__.instances.append(self)

        def ik(self, target_pos, target_rotmat, seed_jnt_values=None):
            self.ik_calls.append(
                {
                    "target_pos": np.asarray(target_pos, dtype=float).copy(),
                    "target_rotmat": np.asarray(target_rotmat, dtype=float).copy(),
                    "seed_jnt_values": np.asarray(seed_jnt_values, dtype=float).copy(),
                }
            )
            if self.__class__.next_exception is not None:
                raise self.__class__.next_exception
            if callable(self.__class__.result_fn):
                result = self.__class__.result_fn(target_pos, target_rotmat, seed_jnt_values)
                if result is None:
                    return None
                return np.asarray(result, dtype=float).copy()
            if self.__class__.next_result is None:
                return None
            return np.asarray(self.__class__.next_result, dtype=float).copy()

        def fk(self, joint_values):
            if callable(self.__class__.fk_fn):
                pos, rot = self.__class__.fk_fn(joint_values)
                return (
                    np.asarray(pos, dtype=float).copy(),
                    np.asarray(rot, dtype=float).copy(),
                )
            return (
                np.asarray(self.__class__.next_fk_pos, dtype=float).copy(),
                np.asarray(self.__class__.next_fk_rot, dtype=float).copy(),
            )

    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy.ok = lambda: False
    fake_rclpy.init = lambda *args, **kwargs: None
    fake_rclpy.shutdown = lambda *args, **kwargs: None
    fake_rclpy.spin_once = lambda *args, **kwargs: None
    fake_rclpy.spin_until_future_complete = lambda *args, **kwargs: None
    fake_rclpy.time = types.SimpleNamespace(Time=lambda: None)

    fake_rclpy_duration = types.ModuleType("rclpy.duration")
    fake_rclpy_duration.Duration = FakeDuration

    fake_rclpy_node = types.ModuleType("rclpy.node")
    fake_rclpy_node.Node = FakeNode

    fake_trac_ik = types.ModuleType("trac_ik")
    fake_trac_ik.TracIK = FakeTracIK

    fake_rbpodo_msgs = types.ModuleType("rbpodo_msgs")
    fake_rbpodo_msgs_srv = types.ModuleType("rbpodo_msgs.srv")
    fake_rbpodo_msgs_srv.TaskStop = types.SimpleNamespace(Request=type("Request", (), {}))

    fake_sensor_msgs = types.ModuleType("sensor_msgs")
    fake_sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    fake_sensor_msgs_msg.JointState = type("JointState", (), {})

    fake_tf2_ros = types.ModuleType("tf2_ros")
    fake_tf2_ros.Buffer = type("Buffer", (), {})
    fake_tf2_ros.TransformListener = type("TransformListener", (), {"__init__": lambda self, *args, **kwargs: None})

    fake_tf_transformations = types.ModuleType("tf_transformations")
    fake_tf_transformations.quaternion_from_matrix = lambda T: np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    fake_tf_transformations.quaternion_matrix = lambda q: np.block(
        [
            [_quat_xyzw_to_rotmat(q), np.zeros((3, 1), dtype=float)],
            [np.zeros((1, 3), dtype=float), np.ones((1, 1), dtype=float)],
        ]
    )

    fake_trajectory_msgs = types.ModuleType("trajectory_msgs")
    fake_trajectory_msgs_msg = types.ModuleType("trajectory_msgs.msg")
    fake_trajectory_msgs_msg.JointTrajectory = type("JointTrajectory", (), {"__init__": lambda self: setattr(self, "points", []) or setattr(self, "joint_names", [])})
    fake_trajectory_msgs_msg.JointTrajectoryPoint = type("JointTrajectoryPoint", (), {"__init__": lambda self: None})

    modules = {
        "rclpy": fake_rclpy,
        "rclpy.duration": fake_rclpy_duration,
        "rclpy.node": fake_rclpy_node,
        "trac_ik": fake_trac_ik,
        "rbpodo_msgs": fake_rbpodo_msgs,
        "rbpodo_msgs.srv": fake_rbpodo_msgs_srv,
        "sensor_msgs": fake_sensor_msgs,
        "sensor_msgs.msg": fake_sensor_msgs_msg,
        "tf2_ros": fake_tf2_ros,
        "tf_transformations": fake_tf_transformations,
        "trajectory_msgs": fake_trajectory_msgs,
        "trajectory_msgs.msg": fake_trajectory_msgs_msg,
    }
    return modules, {"FakeTracIK": FakeTracIK}


def _load_module():
    modules, fake_classes = _install_fake_modules()
    module_name = "rb10_control_joint_trajectory_controller_test"
    sys.modules.pop(module_name, None)
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module, fake_classes


class RB10ControllerTRACIKTests(unittest.TestCase):
    def setUp(self):
        self.module, self.fake_classes = _load_module()
        self.fake_classes["FakeTracIK"].instances.clear()
        self.fake_classes["FakeTracIK"].next_result = np.zeros(6, dtype=float)
        self.fake_classes["FakeTracIK"].next_exception = None
        self.fake_classes["FakeTracIK"].next_fk_pos = np.zeros(3, dtype=float)
        self.fake_classes["FakeTracIK"].next_fk_rot = np.eye(3, dtype=float)
        self.fake_classes["FakeTracIK"].result_fn = None
        self.fake_classes["FakeTracIK"].fk_fn = None

    def test_tracik_seed_and_rotation_matrix_are_used(self):
        self.fake_classes["FakeTracIK"].next_result = np.array([0.3, -0.1, 0.2, 0.4, -0.5, 0.6], dtype=float)
        self.fake_classes["FakeTracIK"].next_fk_pos = np.array([0.3, -0.2, 0.4], dtype=float)
        self.fake_classes["FakeTracIK"].next_fk_rot = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        controller = self.module.RB10Controller(urdf_path="/tmp/rb10.urdf", wait_for_joint_state_sec=0.0)
        controller._latest_positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=float)
        quat_y_90 = np.array([0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0)], dtype=float)

        q_goal = controller.compute_target_qpos_from_pose(
            np.array([0.3, -0.2, 0.4], dtype=float),
            quat_y_90,
            enforce_guard=False,
        )

        np.testing.assert_allclose(q_goal, self.fake_classes["FakeTracIK"].next_result)
        solver = self.fake_classes["FakeTracIK"].instances[-1]
        self.assertEqual(solver.base_link_name, "link0")
        self.assertEqual(solver.tip_link_name, "tcp")
        self.assertEqual(solver.urdf_path, "/tmp/rb10.urdf")
        self.assertGreaterEqual(len(solver.ik_calls), 1)
        ik_call = solver.ik_calls[0]
        np.testing.assert_allclose(ik_call["target_pos"], np.array([0.3, -0.2, 0.4], dtype=float))
        np.testing.assert_allclose(ik_call["seed_jnt_values"], np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=float))
        np.testing.assert_allclose(
            ik_call["target_rotmat"],
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            atol=1e-8,
        )

    def test_tracik_failure_sets_last_ik_fail(self):
        self.fake_classes["FakeTracIK"].next_result = None

        controller = self.module.RB10Controller(urdf_path="/tmp/rb10.urdf", wait_for_joint_state_sec=0.0)
        controller._latest_positions = np.zeros(6, dtype=float)

        q_goal = controller.compute_target_qpos_from_pose(
            np.array([0.6, -0.1, 0.2], dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            enforce_guard=False,
        )

        self.assertIsNone(q_goal)
        self.assertIsNotNone(controller.last_ik_fail)
        self.assertIn("TRAC-IK returned no solution", controller.last_ik_fail)

    def test_rejects_approximate_solution(self):
        self.fake_classes["FakeTracIK"].next_result = np.array([0.3, -0.1, 0.2, 0.4, -0.5, 0.6], dtype=float)
        self.fake_classes["FakeTracIK"].next_fk_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self.fake_classes["FakeTracIK"].next_fk_rot = np.eye(3, dtype=float)

        controller = self.module.RB10Controller(urdf_path="/tmp/rb10.urdf", wait_for_joint_state_sec=0.0)
        controller._latest_positions = np.zeros(6, dtype=float)
        quat_y_90 = np.array([0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0)], dtype=float)

        q_goal = controller.compute_target_qpos_from_pose(
            np.array([0.3, -0.2, 0.4], dtype=float),
            quat_y_90,
            enforce_guard=False,
        )

        self.assertIsNone(q_goal)
        self.assertIsNotNone(controller.last_ik_fail)
        self.assertIn("TRAC-IK rejected approximate solution", controller.last_ik_fail)

    def test_rejects_shoulder_soft_limit(self):
        self.fake_classes["FakeTracIK"].next_result = np.array([0.3, 1.7, 0.2, 0.4, -0.5, 0.6], dtype=float)

        controller = self.module.RB10Controller(urdf_path="/tmp/rb10.urdf", wait_for_joint_state_sec=0.0)
        controller._latest_positions = np.zeros(6, dtype=float)

        q_goal = controller.compute_target_qpos_from_pose(
            np.array([0.3, -0.2, 0.4], dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            enforce_guard=False,
        )

        self.assertIsNone(q_goal)
        self.assertIsNotNone(controller.last_ik_fail)
        self.assertIn("Soft joint limit reject", controller.last_ik_fail)
        self.assertIn("shoulder=", controller.last_ik_fail)

    def test_rejects_wrist2_soft_limit(self):
        self.fake_classes["FakeTracIK"].next_result = np.array([0.3, 0.2, 0.1, 0.4, math.pi / 2.0, 0.6], dtype=float)

        controller = self.module.RB10Controller(urdf_path="/tmp/rb10.urdf", wait_for_joint_state_sec=0.0)
        controller._latest_positions = np.zeros(6, dtype=float)

        q_goal = controller.compute_target_qpos_from_pose(
            np.array([0.3, -0.2, 0.4], dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            enforce_guard=False,
        )

        self.assertIsNone(q_goal)
        self.assertIsNotNone(controller.last_ik_fail)
        self.assertIn("Soft joint limit reject", controller.last_ik_fail)
        self.assertIn("wrist2=", controller.last_ik_fail)

    def test_single_seed_ik_tries_once(self):
        self.fake_classes["FakeTracIK"].next_result = np.array([0.3, -0.1, 0.2, 0.4, -0.5, 0.6], dtype=float)
        self.fake_classes["FakeTracIK"].next_fk_pos = np.array([0.8, 0.3, 0.4], dtype=float)
        self.fake_classes["FakeTracIK"].next_fk_rot = np.eye(3, dtype=float)
        controller = self.module.RB10Controller(urdf_path="/tmp/rb10.urdf", wait_for_joint_state_sec=0.0)
        controller._latest_positions = np.array([3.0, -0.2, -1.4, -1.7, 0.0, 1.4], dtype=float)

        q_goal = controller.compute_target_qpos_from_pose(
            np.array([0.8, 0.3, 0.4], dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            enforce_guard=False,
        )

        self.assertIsNotNone(q_goal)
        solver = self.fake_classes["FakeTracIK"].instances[-1]
        self.assertEqual(len(solver.ik_calls), 1)
        np.testing.assert_allclose(
            solver.ik_calls[0]["seed_jnt_values"],
            np.array([3.0, -0.2, -1.4, -1.7, 0.0, 1.4], dtype=float),
        )

    def test_ik_solution_is_aligned_to_seed_branch_before_publish(self):
        self.fake_classes["FakeTracIK"].next_result = np.array([-3.10, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.fake_classes["FakeTracIK"].next_fk_pos = np.array([0.8, 0.3, 0.4], dtype=float)
        self.fake_classes["FakeTracIK"].next_fk_rot = np.eye(3, dtype=float)

        controller = self.module.RB10Controller(urdf_path="/tmp/rb10.urdf", wait_for_joint_state_sec=0.0)
        controller._latest_positions = np.array([3.12, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

        q_goal = controller.compute_target_qpos_from_pose(
            np.array([0.8, 0.3, 0.4], dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            enforce_guard=True,
        )

        self.assertIsNotNone(q_goal)
        self.assertGreater(float(q_goal[0]), 3.0)
        self.assertLess(abs(float(q_goal[0] - controller._latest_positions[0])), 0.1)

    def test_guard_rejects_large_single_seed_step(self):
        self.fake_classes["FakeTracIK"].next_result = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

        controller = self.module.RB10Controller(urdf_path="/tmp/rb10.urdf", wait_for_joint_state_sec=0.0)
        controller._latest_positions = np.zeros(6, dtype=float)

        q_goal = controller.compute_target_qpos_from_pose(
            np.array([0.8, 0.3, 0.4], dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            enforce_guard=True,
        )

        self.assertIsNone(q_goal)
        self.assertIsNotNone(controller.last_ik_fail)
        self.assertIn("Guard reject", controller.last_ik_fail)

    def test_execute_pose_sequence_fails_without_segment_refine(self):
        controller = self.module.RB10Controller(urdf_path="/tmp/rb10.urdf", wait_for_joint_state_sec=0.0)
        controller._latest_positions = np.zeros(6, dtype=float)

        published = {}

        def _fake_publish(q_goals, durations, min_point_duration=0.0):
            published["q_goals"] = [np.asarray(q, dtype=float).copy() for q in q_goals]
            published["durations"] = list(durations)
            published["min_point_duration"] = float(min_point_duration)
            return True

        def _fake_compute(target_ee_pos, target_ee_rot, enforce_guard=True, seed_q=None):
            x = float(np.asarray(target_ee_pos, dtype=float)[0])
            seed0 = float(np.asarray(seed_q, dtype=float)[0])
            if abs(x - 1.0) < 1e-9 and abs(seed0) < 1e-9:
                controller.last_ik_fail = "TRAC-IK guard reject | synthetic full-step failure"
                return None
            q = np.zeros(6, dtype=float)
            q[0] = x * 0.2
            controller.last_ik_fail = None
            return q

        controller.publish_joint_trajectory = _fake_publish
        controller.compute_target_qpos_from_pose = _fake_compute

        q_path = controller.execute_pose_sequence(
            target_ee_positions=np.array([[1.0, 0.0, 0.0]], dtype=float),
            target_ee_rots=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float),
            point_durations=[0.8],
            enforce_guard=True,
            seed_q=np.zeros(6, dtype=float),
            min_point_duration=0.0,
            start_ee_position=np.array([0.0, 0.0, 0.0], dtype=float),
            start_ee_rot=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        )

        self.assertIsNone(q_path)
        self.assertNotIn("q_goals", published)
        self.assertIsNotNone(controller.last_ik_fail)
        self.assertIn("Sequence IK failed at index 0", controller.last_ik_fail)

    def test_execute_pose_sequence_can_skip_intermediate_waypoints(self):
        controller = self.module.RB10Controller(urdf_path="/tmp/rb10.urdf", wait_for_joint_state_sec=0.0)
        controller._latest_positions = np.zeros(6, dtype=float)

        published = {}

        def _fake_publish(q_goals, durations, min_point_duration=0.0):
            published["q_goals"] = [np.asarray(q, dtype=float).copy() for q in q_goals]
            published["durations"] = list(durations)
            published["min_point_duration"] = float(min_point_duration)
            return True

        def _fake_compute(target_ee_pos, target_ee_rot, enforce_guard=True, seed_q=None):
            del target_ee_rot, enforce_guard
            x = float(np.asarray(target_ee_pos, dtype=float)[0])
            if abs(x - 0.2) < 1e-9:
                controller.last_ik_fail = "synthetic skip candidate failure"
                return None
            q = np.zeros(6, dtype=float)
            q[0] = x
            controller.last_ik_fail = None
            return q

        controller.publish_joint_trajectory = _fake_publish
        controller.compute_target_qpos_from_pose = _fake_compute

        q_path = controller.execute_pose_sequence(
            target_ee_positions=np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=float),
            target_ee_rots=np.array(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
            point_durations=[0.1, 0.2, 0.3],
            enforce_guard=True,
            seed_q=np.zeros(6, dtype=float),
            min_point_duration=0.0,
            max_waypoint_skip=2,
        )

        self.assertIsNotNone(q_path)
        self.assertEqual(len(q_path), 1)
        self.assertAlmostEqual(float(q_path[0][0]), 0.3, places=9)
        self.assertIn("q_goals", published)
        self.assertEqual(len(published["q_goals"]), 1)
        self.assertEqual(len(published["durations"]), 1)
        self.assertAlmostEqual(float(published["durations"][0]), 0.6, places=9)

    def test_fk_uses_tracik_fk(self):
        self.fake_classes["FakeTracIK"].next_fk_pos = np.array([0.4, 0.1, 0.2], dtype=float)
        self.fake_classes["FakeTracIK"].next_fk_rot = np.eye(3, dtype=float)

        controller = self.module.RB10Controller(urdf_path="/tmp/rb10.urdf", wait_for_joint_state_sec=0.0)
        controller._latest_positions = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=float)

        T = controller._fk_current_T()

        self.assertIsNotNone(T)
        np.testing.assert_allclose(T[:3, 3], np.array([0.4, 0.1, 0.2], dtype=float))
        np.testing.assert_allclose(T[:3, :3], np.eye(3, dtype=float))

if __name__ == "__main__":
    unittest.main()
