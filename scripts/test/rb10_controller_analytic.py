#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# ---------------- Quaternion to ZYX Euler Helper ----------------
def _quat_xyzw_to_zyx_rad(q_xyzw: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion [x, y, z, w] to ZYX Euler angles [rx, ry, rz] in radians,
    consistent with R = Rz * Ry * Rx.
    """
    R = quaternion_matrix(q_xyzw)[:3, :3]
    # Clamp to avoid numerical issues
    def clamp(x, lo=-1.0, hi=1.0):
        return max(lo, min(hi, x))
    # For R = Rz(ψ) * Ry(θ) * Rx(φ):
    # θ = asin(-R[2,0]), φ = atan2(R[2,1], R[2,2]), ψ = atan2(R[1,0], R[0,0])
    sy = -R[2, 0]
    sy = clamp(sy)
    ry = float(np.arcsin(sy))
    # Check for gimbal lock (|cos(ry)| ~ 0)
    if abs(np.cos(ry)) > 1e-8:
        rx = float(np.arctan2(R[2, 1], R[2, 2]))
        rz = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        # Gimbal lock fallback: set rx = 0 and compute rz from remaining terms
        rx = 0.0
        rz = float(np.arctan2(-R[0, 1], R[1, 1]))
    return rx, ry, rz

# ---------------- Config ----------------
JTC_TOPIC = "/joint_trajectory_controller/joint_trajectory"
JOINT_STATES_TOPIC = "/joint_states"

BASE_LINK = "link0"
EE_LINK   = "tcp"

# ROS 퍼블리시/구독 기준 6개 조인트 이름 (순서 중요)
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

URDF_PATH  = "/home/sungboo/ros2_ws/src/rbpodo_ros2/rbpodo_description/robots/rb10_1300e_u.urdf"

# IKPy에 넘길 active mask (fixed=False, 가동조인트=True, 끝 tcp는 프레임만 포함하므로 False)
ACTIVE_LINKS_MASK = [False, True, True, True, True, True, True, False]

# 안전가드
MAX_STEP_PER_JOINT_RAD = 0.15
MAX_STEP_L2_RAD        = 0.40
IK_MAX_ITER            = 80


# 로그 토글
DEBUG = False


class RB10Controller(Node):
    """ BASE(frame) 기준 pos(3) + quat(xyzw) -> IK -> JointTrajectory publish """
    def __init__(self):
        super().__init__("rb10_controller")

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

        # IKPy 최종 링크(=조인트) 이름 목록 및 인덱스 맵
        self._link_names: List[str] = [lk.name for lk in self.chain.links]
        self._name2idx = {n: i for i, n in enumerate(self._link_names)}

        # JOINT_NAMES를 IKPy 인덱스로 벡터화해두면 q6<->q_full 변환이 매우 빠름
        try:
            self._idx6 = np.array([self._name2idx[n] for n in JOINT_NAMES], dtype=int)
        except KeyError as e:
            raise RuntimeError(f"URDF/IKPy 체인에 '{e.args[0]}' 조인트가 없습니다. JOINT_NAMES/URDF를 맞춰주세요.")

        # 디버그: EE가 tcp인지 1회만 확인
        if DEBUG:
            self.get_logger().info(f"EE (last link) = {self.chain.links[-1].name}")

        # 최신 JointState(6개, JOINT_NAMES 순서 저장)
        self._latest_positions: Optional[np.ndarray] = None
        self._joint_index_map = None

        # JointState 첫 수신까지 잠깐 대기 (필요 최소만)
        self.get_logger().info("Waiting for first JointState...")
        end = self.get_clock().now().nanoseconds + int(5e9)
        while rclpy.ok() and (self.get_clock().now().nanoseconds < end) and (self._latest_positions is None):
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._latest_positions is None:
            self.get_logger().warn("No joint_states received. joint_states를 확인하세요.")
        else:
            self.get_logger().info("Got initial joint_states.")

    # ---------------- JointState 콜백 ----------------
    def _joint_cb(self, msg: JointState):
        if self._joint_index_map is None:
            self._joint_index_map = {name: i for i, name in enumerate(msg.name)}
            missing = [n for n in JOINT_NAMES if n not in self._joint_index_map]
            if missing:
                self.get_logger().warn(f"JointState에 없는 조인트: {missing}")

        # JOINT_NAMES 순서로 뽑아서 저장 (누락 시 0 유지)
        q = np.zeros(len(JOINT_NAMES), dtype=float)
        for k, name in enumerate(JOINT_NAMES):
            idx = self._joint_index_map.get(name)
            if idx is not None and idx < len(msg.position):
                q[k] = msg.position[idx]
        self._latest_positions = q

    # ---------------- q6 <-> q_full 변환 (벡터화, 할당 최소화) ----------------
    def _q_full_from_q6(self, q6: np.ndarray) -> np.ndarray:
        q_full = np.zeros(len(self._link_names), dtype=float)
        q_full[self._idx6] = q6
        return q_full

    def _q6_from_q_full(self, q_full: np.ndarray) -> np.ndarray:
        return q_full[self._idx6].astype(float, copy=False)

    # ---------------- FK ----------------
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

    # ---------------- 안전가드 ----------------
    @staticmethod
    def _wrap_pi(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2 * np.pi) - np.pi

    def _guard_ok(self, q6: np.ndarray, seed6: np.ndarray) -> bool:
        diffs = self._wrap_pi(q6 - seed6)
        if np.any(np.abs(diffs) > MAX_STEP_PER_JOINT_RAD) or np.linalg.norm(diffs) > MAX_STEP_L2_RAD:
            if DEBUG:
                self.get_logger().warn(
                    f"Δq 과다: max={np.max(np.abs(diffs)):.3f} rad, L2={np.linalg.norm(diffs):.3f} rad"
                )
            return False
        return True

    # ---------------- IK ----------------
    def compute_target_qpos_from_pose(
        self,
        target_ee_pos: np.ndarray,        # (3,)
        target_ee_rot_xyzw: np.ndarray,   # (4,) xyzw (BASE 기준)
        enforce_guard: bool = True,
    ) -> Optional[np.ndarray]:
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신 — IK seed 불가")
            return None

        # 입력 파싱/정규화
        p = np.asarray(target_ee_pos, dtype=float).reshape(3,)
        q = np.asarray(target_ee_rot_xyzw, dtype=float).reshape(4,)
        n = float(np.linalg.norm(q))
        if not np.isfinite(n) or n <= 0:
            self.get_logger().warn("목표 쿼터니언이 유효하지 않음")
            return None
        q /= n

        # IKPy가 요구하는 절대 referential 3x3 회전 행렬
        R_tgt = quaternion_matrix(q)[:3, :3]

        # --- Analytic IK (RB10-1300) ---
        try:
            from analytic_ik import ik_rb10_1300  # expects pos[m], rot[rad] (ZYX)
        except ImportError:
            self.get_logger().error("analytic_ik.py not found or import failed — analytic IK is required.")
            return None
        rx, ry, rz = _quat_xyzw_to_zyx_rad(q)
        # Analytic solver returns degrees; convert to radians for controllers
        th_deg = ik_rb10_1300(p.tolist(), [rx, ry, rz])
        q6 = np.deg2rad(np.asarray(th_deg, dtype=float))
        if enforce_guard and not self._guard_ok(q6, self._latest_positions):
            return None
        return q6

    # FK 검증용(선택적) — DEBUG 시에만 사용
    def _fk_current_T_of(self, q6: np.ndarray) -> Optional[np.ndarray]:
        try:
            q_full = self._q_full_from_q6(q6)
            return np.asarray(self.chain.forward_kinematics(q_full), dtype=float)
        except Exception:
            return None

    # ---------------- Trajectory publish ----------------
    def publish_joint_trajectory(
        self,
        q_goals: List[List[float]],
        durations: List[float],
    ) -> bool:
        if len(q_goals) != len(durations):
            raise ValueError("q_goals and durations length mismatch")

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        traj.points = []

        elapsed = 0.0
        for q_goal, duration in zip(q_goals, durations):
            elapsed += float(max(0.2, duration))
            p = JointTrajectoryPoint()
            p.positions = list(map(float, q_goal))
            p.time_from_start = Duration(seconds=elapsed).to_msg()
            traj.points.append(p)

        self.traj_pub.publish(traj)
        if DEBUG:
            self.get_logger().info(f"Trajectory published (points={len(traj.points)}, total={elapsed:.2f}s)")
        return True

    def publish_qpos(self, q_goal: List[float], duration: float = 0.3) -> bool:
        return self.publish_joint_trajectory([q_goal], [duration])

    # ---------------- 현재 EE pose ----------------
    def get_current_ee_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """ (pos[3], quat[4]) — FK 우선, 실패 시 TF """
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
        # 현재 포즈 출력(테스트)
        if DEBUG:
            print(node.get_current_ee_pose())

        # 예시: BASE 기준 목표
        target_ee_pos = [0.6, 0.1, 0.4]
        target_ee_rot_xyzw = [0.7071068, 0, 0, 0.7071068]

        q = node.compute_target_qpos_from_pose(target_ee_pos, target_ee_rot_xyzw, enforce_guard=False)
        if q is not None:
            node.publish_joint_trajectory([q.tolist()], [10.0])
            if DEBUG:
                node.get_logger().info("Moved to target pose.")

    finally:
        if DEBUG:
            print(node.get_current_ee_pose())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
