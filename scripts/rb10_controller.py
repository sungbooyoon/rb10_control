#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RB10 Controller (ROS 2 / rclpy)
- BASE(frame) 기준 EE pose (pos[3] + quat[xyzw]) -> IK -> JointTrajectory publish
- IK 실패 시 항상 이유를 WARN 로그로 남기고, self.last_ik_fail 에 저장
"""

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import numpy as np
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_matrix, quaternion_from_matrix
from ikpy.chain import Chain

# ================= Config =================
JTC_TOPIC = "/joint_trajectory_controller/joint_trajectory"
JOINT_STATES_TOPIC = "/joint_states"

BASE_LINK = "link0"
EE_LINK   = "tcp"

# ROS 퍼블리시/구독 기준 6개 조인트 이름 (순서 중요)
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

URDF_PATH  = "/home/sungboo/ros2_ws/src/rbpodo_ros2/rbpodo_description/robots/rb10_1300e_u.urdf"

# IKPy active mask (fixed=False, 가동조인트=True, 최종 tcp는 프레임만 포함하므로 False)
ACTIVE_LINKS_MASK = [False, True, True, True, True, True, True, False]

# 안전가드
MAX_STEP_PER_JOINT_RAD = 0.20
MAX_STEP_L2_RAD        = 0.40
IK_MAX_ITER            = 80

# 디버그 토글
DEBUG = False


class RB10Controller(Node):
    """ BASE(frame) 기준 pos(3) + quat(xyzw) -> IK -> JointTrajectory publish """
    def __init__(self):
        super().__init__("rb10_controller")

        # 외부에서 확인 가능한 최근 IK 실패 사유
        self.last_ik_fail: Optional[str] = None

        # pubs/subs
        self.joint_sub = self.create_subscription(JointState, JOINT_STATES_TOPIC, self._joint_cb, 10)
        self.traj_pub  = self.create_publisher(JointTrajectory, JTC_TOPIC, 10)

        # TF
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)

        # IKPy 체인 (full: fixed 포함, active mask 적용)
        self.chain = Chain.from_urdf_file(
            URDF_PATH,
            base_elements=[BASE_LINK],
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

        # 최신 JointState(6개, JOINT_NAMES 순)
        self._latest_positions: Optional[np.ndarray] = None
        self._joint_index_map = None

        # 첫 JointState 대기
        self.get_logger().info("Waiting for first JointState...")
        end = self.get_clock().now().nanoseconds + int(5e9)
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
        target_ee_pos: np.ndarray,        # (3,)
        target_ee_rot_xyzw: np.ndarray,   # (4,) xyzw (BASE 기준)
        enforce_guard: bool = True,
    ) -> Optional[np.ndarray]:
        if self._latest_positions is None:
            self._ik_fail("No joint_states — IK seed unavailable")
            return None

        # 입력 정규화
        p = np.asarray(target_ee_pos, dtype=float).reshape(3,)
        q = np.asarray(target_ee_rot_xyzw, dtype=float).reshape(4,)
        n = float(np.linalg.norm(q))
        if not np.isfinite(n) or n <= 0:
            self._ik_fail("Invalid target quaternion (norm <= 0 or NaN)")
            return None
        q /= n

        # IKPy 요구 포맷: 절대 회전 3x3
        R_tgt = quaternion_matrix(q)[:3, :3]

        # seed = 현재 상태
        seed_full = self._q_full_from_q6(self._latest_positions)

        # IK
        try:
            q_full = self.chain.inverse_kinematics(
                target_position=p.tolist(),
                target_orientation=R_tgt.tolist(),
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
        if len(q_full) != len(self._link_names):
            self._ik_fail(f"IKPy length mismatch (got {len(q_full)}, expect {len(self._link_names)})")
            return None

        q6 = self._q6_from_q_full(np.asarray(q_full, dtype=float))

        if enforce_guard and not self._guard_ok(q6, self._latest_positions):
            return None

        if DEBUG:
            # orientation error 체크
            T_sol = self._fk_current_T_of(q6)
            if T_sol is not None:
                R_sol = T_sol[:3, :3]
                theta = math.acos(max(-1.0, min(1.0, (np.trace(R_sol.T @ R_tgt) - 1) / 2)))
                self.get_logger().info(f"ori_err_rad={theta:.6f}")

        return q6

    # ---------- Trajectory publish ----------
    def publish_qpos(self, q_goal: List[float], duration: float = 0.3) -> bool:
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        p1 = JointTrajectoryPoint()
        p1.positions = list(map(float, q_goal))
        p1.time_from_start = Duration(seconds=float(max(0.2, duration))).to_msg()

        traj.points = [p1]
        self.traj_pub.publish(traj)
        if DEBUG:
            self.get_logger().info(f"Trajectory published (duration={duration:.2f}s)")
        return True

    # ---------- 현재 EE pose ----------
    def get_current_ee_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """ (pos[3], quat[4]) — FK 우선, 실패 시 TF 조회 """
        T = self._fk_current_T()
        if T is not None:
            return T[:3, 3].astype(float), np.asarray(quaternion_from_matrix(T), dtype=float)
        try:
            t = self.tf_buf.lookup_transform(BASE_LINK, EE_LINK, rclpy.time.Time())
            pos = np.array([t.transform.translation.x,
                            t.transform.translation.y,
                            t.transform.translation.z], dtype=float)
            quat = np.array([t.transform.rotation.x,
                             t.transform.rotation.y,
                             t.transform.rotation.z,
                             t.transform.rotation.w], dtype=float)
            return pos, quat
        except Exception:
            return None


def main():
    rclpy.init()
    node = RB10Controller()
    try:
        if DEBUG:
            node.get_logger().info(f"Current EE pose: {node.get_current_ee_pose()}")
        
        # 예시: BASE 기준 목표
        target_ee_pos = [0.7, 0.1, 0.3]
        target_ee_rot_xyzw = [0, -0.7071068, -0.7071068, 0]

        q = node.compute_target_qpos_from_pose(target_ee_pos, target_ee_rot_xyzw, enforce_guard=False)
        if q is not None:
            node.publish_qpos(q.tolist(), duration=10.0)
            if DEBUG:
                node.get_logger().info("Moved to target pose.")
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":
    main()
