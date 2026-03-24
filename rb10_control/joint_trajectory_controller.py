#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RB10 joint_trajectory_controller helpers.

This module is intended to be imported from other local projects as:

    from rb10_control import RB10Controller

while still supporting the legacy script entrypoint in
``scripts/rb10_controller.py``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from ikpy.chain import Chain
from rclpy.duration import Duration
from rclpy.node import Node
from rbpodo_msgs.srv import TaskStop
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_from_matrix, quaternion_matrix
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ================= Config =================
JTC_TOPIC = "/joint_trajectory_controller/joint_trajectory"
JOINT_STATES_TOPIC = "/joint_states"
TASK_STOP_SERVICE = "/rbpodo_hardware/task_stop"

BASE_LINK = "link0"
EE_LINK = "tcp"

# ROS 퍼블리시/구독 기준 6개 조인트 이름 (순서 중요)
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

URDF_PATH = "/home/sungboo/ros2_ws/src/rbpodo_ros2/rbpodo_description/robots/rb10_1300e_u.urdf"

# IKPy active mask (fixed=False, 가동조인트=True, 최종 tcp는 프레임만 포함하므로 False)
ACTIVE_LINKS_MASK = [False, True, True, True, True, True, True, False]

# 안전가드
MAX_STEP_PER_JOINT_RAD = 0.25
MAX_STEP_L2_RAD = 0.45
IK_MAX_ITER = 80

# 디버그 토글
DEBUG = False


class RB10Controller(Node):
    """BASE(frame) 기준 pos(3) + quat(xyzw) -> IK -> JointTrajectory publish."""

    def __init__(
        self,
        base_link: str = BASE_LINK,
        ee_link: str = EE_LINK,
        urdf_path: str = URDF_PATH,
        joint_states_topic: str = JOINT_STATES_TOPIC,
        jtc_topic: str = JTC_TOPIC,
        task_stop_service: str = TASK_STOP_SERVICE,
        wait_for_joint_state_sec: float = 5.0,
    ):
        super().__init__("rb10_controller")

        self.base_link = str(base_link)
        self.ee_link = str(ee_link)
        self.urdf_path = str(urdf_path)
        self.joint_states_topic = str(joint_states_topic)
        self.jtc_topic = str(jtc_topic)
        self.task_stop_service = str(task_stop_service)

        # 외부에서 확인 가능한 최근 IK 실패 사유
        self.last_ik_fail: Optional[str] = None

        # pubs/subs
        self.joint_sub = self.create_subscription(JointState, self.joint_states_topic, self._joint_cb, 10)
        self.traj_pub = self.create_publisher(JointTrajectory, self.jtc_topic, 10)
        self.task_stop_client = self.create_client(TaskStop, self.task_stop_service)

        # TF
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)

        # IKPy 체인 (full: fixed 포함, active mask 적용)
        self.chain = Chain.from_urdf_file(
            self.urdf_path,
            base_elements=[self.base_link],
            active_links_mask=ACTIVE_LINKS_MASK,
        )

        # IKPy 링크 이름/인덱스
        self._link_names: List[str] = [lk.name for lk in self.chain.links]
        self._name2idx = {n: i for i, n in enumerate(self._link_names)}

        # JOINT_NAMES -> IKPy 인덱스 매핑
        try:
            self._idx6 = np.array([self._name2idx[n] for n in JOINT_NAMES], dtype=int)
        except KeyError as e:
            raise RuntimeError(f"URDF/IKPy 체인에 '{e.args[0]}' 조인트가 없습니다. JOINT_NAMES/URDF를 맞춰주세요.")

        tip_name = self.get_chain_tip_link_name()
        if tip_name != self.ee_link:
            self.get_logger().warn(
                f"IKPy chain tip is '{tip_name}', expected EE_LINK '{self.ee_link}'. "
                "URDF/EE_LINK 설정을 확인하세요."
            )
        elif DEBUG:
            self.get_logger().info(f"IKPy chain tip verified: {tip_name}")

        # 최신 JointState(6개, JOINT_NAMES 순)
        self._latest_positions: Optional[np.ndarray] = None
        self._joint_index_map = None

        # 첫 JointState 대기
        self.get_logger().info("Waiting for first JointState...")
        end = self.get_clock().now().nanoseconds + int(max(0.0, float(wait_for_joint_state_sec)) * 1e9)
        while rclpy.ok() and (self.get_clock().now().nanoseconds < end) and (self._latest_positions is None):
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._latest_positions is None:
            self.get_logger().warn("No joint_states received. joint_states를 확인하세요.")
        else:
            self.get_logger().info("Got initial joint_states.")

    # ---------- 공용 진단 헬퍼 ----------
    def _ik_fail(self, reason: str, extra: Optional[str] = None) -> None:
        self.last_ik_fail = reason if extra is None else f"{reason} | {extra}"
        self.get_logger().warn(self.last_ik_fail)

    # ---------- JointState 콜백 ----------
    def _joint_cb(self, msg: JointState):
        if self._joint_index_map is None:
            self._joint_index_map = {name: i for i, name in enumerate(msg.name)}
            missing = [n for n in JOINT_NAMES if n not in self._joint_index_map]
            if missing:
                self.get_logger().warn(f"JointState에 없는 조인트: {missing}")

        q = np.zeros(len(JOINT_NAMES), dtype=float)
        for k, name in enumerate(JOINT_NAMES):
            idx = self._joint_index_map.get(name)
            if idx is not None and idx < len(msg.position):
                q[k] = msg.position[idx]
        self._latest_positions = q

    # ---------- q6 <-> q_full ----------
    def _q_full_from_q6(self, q6: np.ndarray) -> np.ndarray:
        q_full = np.zeros(len(self._link_names), dtype=float)
        q_full[self._idx6] = q6
        return q_full

    def _q6_from_q_full(self, q_full: np.ndarray) -> np.ndarray:
        return q_full[self._idx6].astype(float, copy=False)

    def _coerce_q6(self, q6: Sequence[float], name: str = "q6") -> np.ndarray:
        q = np.asarray(q6, dtype=float).reshape(-1)
        if q.shape[0] != len(JOINT_NAMES):
            raise ValueError(f"{name} must have {len(JOINT_NAMES)} elements, got shape {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError(f"{name} contains NaN/Inf")
        return q.copy()

    def _resolve_seed_q6(self, seed_q: Optional[Sequence[float]]) -> Optional[np.ndarray]:
        if seed_q is not None:
            try:
                return self._coerce_q6(seed_q, name="seed_q")
            except Exception as e:
                self._ik_fail("Invalid seed_q", str(e))
                return None
        if self._latest_positions is None:
            self._ik_fail("No joint_states — IK seed unavailable")
            return None
        return np.asarray(self._latest_positions, dtype=float).copy()

    # ---------- FK ----------
    def _fk_current_T(self) -> Optional[np.ndarray]:
        if self._latest_positions is None:
            return None
        try:
            q_full = self._q_full_from_q6(self._latest_positions)
            return np.asarray(self.chain.forward_kinematics(q_full), dtype=float)
        except Exception as e:
            if DEBUG:
                self.get_logger().warn(f"FK 실패: {e}")
            return None

    def _fk_current_T_of(self, q6: np.ndarray) -> Optional[np.ndarray]:
        try:
            q_full = self._q_full_from_q6(q6)
            return np.asarray(self.chain.forward_kinematics(q_full), dtype=float)
        except Exception:
            return None

    # ---------- 안전가드 ----------
    @staticmethod
    def _wrap_pi(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2 * np.pi) - np.pi

    def _guard_ok(self, q6: np.ndarray, seed6: np.ndarray) -> bool:
        diffs = self._wrap_pi(q6 - seed6)
        max_abs = float(np.max(np.abs(diffs)))
        l2 = float(np.linalg.norm(diffs))
        if (max_abs > MAX_STEP_PER_JOINT_RAD) or (l2 > MAX_STEP_L2_RAD):
            self._ik_fail(f"Guard reject: Δq too large (max={max_abs:.3f} rad, L2={l2:.3f} rad)")
            return False
        return True

    # ---------- IK ----------
    def compute_target_qpos_from_pose(
        self,
        target_ee_pos: np.ndarray,
        target_ee_rot: np.ndarray,
        enforce_guard: bool = True,
        seed_q: Optional[Sequence[float]] = None,
    ) -> Optional[np.ndarray]:
        """Compute joint angles for a BASE-frame TCP pose target."""
        seed6 = self._resolve_seed_q6(seed_q)
        if seed6 is None:
            return None

        p = np.asarray(target_ee_pos, dtype=float).reshape(3,)
        q = np.asarray(target_ee_rot, dtype=float).reshape(4,)
        n = float(np.linalg.norm(q))
        if not np.isfinite(n) or n <= 0:
            self._ik_fail("Invalid target quaternion (norm <= 0 or NaN)")
            return None
        q /= n
        r_tgt = quaternion_matrix(q)[:3, :3]

        seed_full = self._q_full_from_q6(seed6)
        try:
            q_full = self.chain.inverse_kinematics(
                target_position=p.tolist(),
                target_orientation=r_tgt.tolist(),
                initial_position=seed_full,
                max_iter=IK_MAX_ITER,
                orientation_mode="all",
            )
        except Exception as e:
            self._ik_fail("IKPy inverse exception", str(e))
            return None

        if q_full is None:
            self._ik_fail("IKPy returned None")
            return None

        q6 = self._q6_from_q_full(np.asarray(q_full, dtype=float))
        if enforce_guard and not self._guard_ok(q6, seed6):
            return None
        return q6

    def compute_joint_path_from_pose_sequence(
        self,
        target_ee_positions: np.ndarray,
        target_ee_rots: np.ndarray,
        enforce_guard: bool = True,
        seed_q: Optional[Sequence[float]] = None,
    ) -> Optional[List[np.ndarray]]:
        pos = np.asarray(target_ee_positions, dtype=float)
        rot = np.asarray(target_ee_rots, dtype=float)

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"target_ee_positions must be (N,3), got {pos.shape}")
        if rot.ndim != 2 or rot.shape[1] != 4:
            raise ValueError(f"target_ee_rots must be (N,4) [xyzw], got {rot.shape}")
        if pos.shape[0] != rot.shape[0]:
            raise ValueError("target_ee_positions and target_ee_rots length mismatch")

        q_seed = self._resolve_seed_q6(seed_q)
        if q_seed is None:
            return None

        q_path: List[np.ndarray] = []
        for i in range(pos.shape[0]):
            q_goal = self.compute_target_qpos_from_pose(
                pos[i],
                rot[i],
                enforce_guard=enforce_guard,
                seed_q=q_seed,
            )
            if q_goal is None:
                prev_reason = self.last_ik_fail or "unknown IK failure"
                self._ik_fail(f"Sequence IK failed at index {i}", prev_reason)
                return None
            q_goal = np.asarray(q_goal, dtype=float)
            q_path.append(q_goal)
            q_seed = q_goal

        return q_path

    # ---------- Trajectory publish ----------
    def publish_qpos(
        self,
        q_goal: List[float],
        duration: float = 0.3,
        min_point_duration: float = 0.20,
    ) -> bool:
        # Backward-compatible convenience wrapper for single-point commands.
        return self.publish_joint_trajectory([q_goal], [duration], min_point_duration=min_point_duration)

    def publish_joint_trajectory(
        self,
        q_goals: Sequence[Sequence[float]],
        durations: Sequence[float],
        min_point_duration: float = 0.20,
    ) -> bool:
        if len(q_goals) == 0:
            raise ValueError("q_goals must not be empty")
        if len(q_goals) != len(durations):
            raise ValueError("q_goals and durations length mismatch")

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        traj.points = []

        elapsed = 0.0
        min_dt = max(0.0, float(min_point_duration))
        for i, (q_goal, duration) in enumerate(zip(q_goals, durations)):
            q = self._coerce_q6(q_goal, name=f"q_goal[{i}]")
            dt = float(duration)
            if not np.isfinite(dt):
                raise ValueError(f"duration[{i}] is NaN/Inf")
            elapsed += max(min_dt, dt)

            p = JointTrajectoryPoint()
            p.positions = q.tolist()
            p.time_from_start = Duration(seconds=elapsed).to_msg()
            traj.points.append(p)

        self.traj_pub.publish(traj)
        if DEBUG:
            self.get_logger().info(
                f"Trajectory published (points={len(traj.points)}, total_duration={elapsed:.2f}s)"
            )
        return True

    def execute_pose_sequence(
        self,
        target_ee_positions: np.ndarray,
        target_ee_rots: np.ndarray,
        point_durations: Sequence[float],
        enforce_guard: bool = True,
        seed_q: Optional[Sequence[float]] = None,
        min_point_duration: float = 0.20,
    ) -> Optional[List[np.ndarray]]:
        if len(point_durations) != int(np.asarray(target_ee_positions).shape[0]):
            raise ValueError("point_durations length must match the number of target poses")

        q_path = self.compute_joint_path_from_pose_sequence(
            target_ee_positions=target_ee_positions,
            target_ee_rots=target_ee_rots,
            enforce_guard=enforce_guard,
            seed_q=seed_q,
        )
        if q_path is None:
            return None

        self.publish_joint_trajectory(q_path, point_durations, min_point_duration=min_point_duration)
        return q_path

    def emergency_stop(self, timeout: float = 1.0, wait_for_service_sec: float = 0.5) -> bool:
        req = TaskStop.Request()
        req.timeout = float(timeout)

        if not self.task_stop_client.wait_for_service(timeout_sec=max(0.0, float(wait_for_service_sec))):
            self.get_logger().warn(f"TaskStop service unavailable: {self.task_stop_service}")
            return False

        future = self.task_stop_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=max(0.0, float(timeout)))

        if not future.done():
            self.get_logger().warn("TaskStop service call timed out")
            return False

        exc = future.exception()
        if exc is not None:
            self.get_logger().warn(f"TaskStop service call failed: {exc}")
            return False

        result = future.result()
        if result is None:
            self.get_logger().warn("TaskStop service returned no response")
            return False

        ok = bool(result.success)
        if ok:
            self.get_logger().warn("Emergency stop requested via TaskStop")
        else:
            self.get_logger().warn("TaskStop service responded with success=False")
        return ok

    # ---------- 상태 조회 ----------
    def get_chain_tip_link_name(self) -> str:
        if len(self._link_names) <= 0:
            return ""
        return str(self._link_names[-1])

    def get_current_joint_positions(self) -> Optional[np.ndarray]:
        if self._latest_positions is None:
            return None
        return np.asarray(self._latest_positions, dtype=float).copy()

    def get_current_ee_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(pos[3], quat[4]) — FK 우선, 실패 시 TF 조회."""
        t = self._fk_current_T()
        if t is not None:
            return t[:3, 3].astype(float), np.asarray(quaternion_from_matrix(t), dtype=float)
        try:
            tf = self.tf_buf.lookup_transform(self.base_link, self.ee_link, rclpy.time.Time())
            pos = np.array(
                [
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                ],
                dtype=float,
            )
            quat = np.array(
                [
                    tf.transform.rotation.x,
                    tf.transform.rotation.y,
                    tf.transform.rotation.z,
                    tf.transform.rotation.w,
                ],
                dtype=float,
            )
            return pos, quat
        except Exception:
            return None


def main() -> None:
    rclpy.init()
    node = RB10Controller()
    try:
        if DEBUG:
            node.get_logger().info(f"Current EE pose: {node.get_current_ee_pose()}")

        target_ee_pos = [0.7, 0.1, 0.3]
        target_ee_rot_xyzw = [0, -0.7071068, -0.7071068, 0]

        q = node.compute_target_qpos_from_pose(target_ee_pos, target_ee_rot_xyzw, enforce_guard=False)
        if q is not None:
            node.publish_joint_trajectory([q.tolist()], [10.0])
            if DEBUG:
                node.get_logger().info("Moved to target pose.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


__all__ = [
    "ACTIVE_LINKS_MASK",
    "BASE_LINK",
    "DEBUG",
    "EE_LINK",
    "IK_MAX_ITER",
    "JOINT_NAMES",
    "JOINT_STATES_TOPIC",
    "JTC_TOPIC",
    "MAX_STEP_L2_RAD",
    "MAX_STEP_PER_JOINT_RAD",
    "RB10Controller",
    "TASK_STOP_SERVICE",
    "URDF_PATH",
    "main",
]
