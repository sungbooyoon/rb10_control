#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import numpy as np
from tf2_ros import Buffer, TransformListener
from tf_transformations import (
    quaternion_matrix, quaternion_from_matrix
)
from ikpy.chain import Chain

# ---------------- Config ----------------
JTC_TOPIC = "/joint_trajectory_controller/joint_trajectory"
JOINT_STATES_TOPIC = "/joint_states"
PUBLISH_RATE_HZ = 30

BASE_LINK = "link0"
EE_LINK   = "tcp"

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
URDF_PATH  = "/home/sungboo/ros2_ws/src/rbpodo_ros2/rbpodo_description/robots/rb10_1300e_u.urdf"

# ikpy 체인 구성용
ACTIVE_LINKS_MASK = [False, True, True, True, True, True, True, False]
BASE_ELEMENTS     = [BASE_LINK]

# 안전가드
MAX_STEP_PER_JOINT_RAD = 0.15
MAX_STEP_L2_RAD        = 0.40
IK_MAX_ITER            = 50


class RB10Controller(Node):
    """ 포즈(np) -> IK -> JointTrajectory publish """
    def __init__(self):
        super().__init__("rb10_controller")
        self.joint_sub = self.create_subscription(JointState, JOINT_STATES_TOPIC, self._joint_cb, 10)
        self.traj_pub  = self.create_publisher(JointTrajectory, JTC_TOPIC, 10)

        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)

        self.chain_full = Chain.from_urdf_file(URDF_PATH, base_elements=BASE_ELEMENTS)
        self.chain = Chain(name="rb10_reduced", links=[
            l for (l, active) in zip(self.chain_full.links, ACTIVE_LINKS_MASK) if active
        ])

        self._latest_positions: Optional[List[float]] = None
        self._joint_index_map = None

        self.get_logger().info("Waiting for first JointState...")
        end = time.time() + 5.0
        while rclpy.ok() and time.time() < end and self._latest_positions is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._latest_positions is None:
            self.get_logger().warn("No joint_states received. joint_states를 확인하세요.")
        else:
            self.get_logger().info(f"Got initial joint_states: {[f'{v:.3f}' for v in self._latest_positions]}")

    def _joint_cb(self, msg: JointState):
        if self._joint_index_map is None:
            self._joint_index_map = {name: i for i, name in enumerate(msg.name)}
            missing = [n for n in JOINT_NAMES if n not in self._joint_index_map]
            if missing:
                self.get_logger().warn(f"JointState에 없는 조인트: {missing}")

        positions = [0.0] * len(JOINT_NAMES)
        for i, name in enumerate(JOINT_NAMES):
            idx = self._joint_index_map.get(name, None)
            if idx is None or idx >= len(msg.position):
                positions[i] = positions[i] if self._latest_positions else 0.0
            else:
                positions[i] = msg.position[idx]
        self._latest_positions = positions

    # ===== FK / IK =====
    def _fk_current_T(self) -> Optional[np.ndarray]:
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신 — FK 불가")
            return None
        try:
            T = self.chain.forward_kinematics(self._latest_positions)
            return np.asarray(T, dtype=float)
        except Exception as e:
            self.get_logger().warn(f"IKPy forward 실패: {e}")
            return None

    def fk_from_joints(self, q: List[float]) -> Optional[np.ndarray]:
        try:
            T = self.chain.forward_kinematics(q)
            return np.asarray(T, dtype=float)
        except Exception as e:
            self.get_logger().warn(f"IKPy forward 실패: {e}")
            return None

    def _angle_diff_matrix(self, sols: np.ndarray, seed: np.ndarray) -> np.ndarray:
        diffs = sols - seed
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
        return diffs

    def _guard_ok(self, q: np.ndarray, seed: np.ndarray) -> bool:
        diffs = self._angle_diff_matrix(q.reshape(1, -1), seed).reshape(-1)
        l2 = float(np.linalg.norm(diffs))
        if np.any(np.abs(diffs) > MAX_STEP_PER_JOINT_RAD) or l2 > MAX_STEP_L2_RAD:
            self.get_logger().warn(
                f"IK Δ 과다: max|Δ|={np.max(np.abs(diffs)):.3f} rad, ||Δ||2={l2:.3f} rad → 취소"
            )
            return False
        return True

    def compute_target_qpos_from_pose(
        self,
        target_ee_pos: np.ndarray,       # (3,)
        target_ee_rot_xyzw: np.ndarray,  # (4,)
        enforce_guard: bool = True
    ) -> Optional[np.ndarray]:
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신 — IK seed 불가")
            return None
        try:
            pos  = np.asarray(target_ee_pos, dtype=float).reshape(3,)
            quat = np.asarray(target_ee_rot_xyzw, dtype=float).reshape(4,)
        except Exception:
            self.get_logger().warn(f"target shape error: pos={np.shape(target_ee_pos)}, quat={np.shape(target_ee_rot_xyzw)}")
            return None

        T = quaternion_matrix(quat)
        T[0, 3], T[1, 3], T[2, 3] = pos.tolist()

        try:
            q = self.chain.inverse_kinematics_frame(
                T,
                initial_position=np.asarray(self._latest_positions, dtype=float),
                max_iter=IK_MAX_ITER
            )
        except Exception as e:
            self.get_logger().warn(f"IKPy inverse 실패: {e}")
            return None

        if q is None or len(q) != len(JOINT_NAMES):
            return None

        q = np.asarray(q, dtype=float)
        if enforce_guard and not self._guard_ok(q, np.asarray(self._latest_positions, dtype=float)):
            return None
        return q

    def publish_qpos(self, q_goal: List[float], duration: float = 0.3) -> bool:
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        p1 = JointTrajectoryPoint()
        p1.positions = list(q_goal)
        p1.time_from_start = Duration(seconds=float(max(0.2, duration))).to_msg()

        traj.points = [p1]
        self.traj_pub.publish(traj)
        self.get_logger().info(f"Trajectory published (duration={duration:.2f}s)")
        return True

    def get_current_ee_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """ (pos[3], quat[4]) — FK 우선, 실패 시 TF """
        T = self._fk_current_T()
        if T is not None:
            pos = T[:3, 3].astype(float)
            quat = np.asarray(quaternion_from_matrix(T), dtype=float)
            return pos, quat
        try:
            t = self.tf_buf.lookup_transform(BASE_LINK, EE_LINK, rclpy.time.Time())
            pos = np.array([
                float(t.transform.translation.x),
                float(t.transform.translation.y),
                float(t.transform.translation.z),
            ], dtype=float)
            quat = np.array([
                float(t.transform.rotation.x),
                float(t.transform.rotation.y),
                float(t.transform.rotation.z),
                float(t.transform.rotation.w),
            ], dtype=float)
            return pos, quat
        except Exception as e:
            self.get_logger().warn(f"현재 EE pose 조회 실패(FK/TF): {e}")
            return None


def main():
    rclpy.init()
    node = RB10Controller()
    try:
        target_ee_pos = [0.5, -0.11, 0.35]
        target_ee_rot_xyzw = [0.707, 0.0, 0.707, 0.0]

        q = node.compute_target_qpos_from_pose(target_ee_pos, target_ee_rot_xyzw, enforce_guard=False)
        if q is not None:
            node.publish_qpos(q.tolist(), duration=10.0)
            node.get_logger().info("Moved to initial target pose.")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
