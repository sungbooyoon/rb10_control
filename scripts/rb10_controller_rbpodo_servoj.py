#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RB10 Controller (ROS 2 / rclpy)
- BASE(frame) 기준 EE pose (pos[3] + quat[xyzw]) -> IK -> rbpodo move_servo_j publish
- IK 실패 시 항상 이유를 WARN 로그로 남기고, self.last_ik_fail 에 저장
"""

import math
import time
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

import numpy as np
from tf_transformations import quaternion_matrix, quaternion_from_euler
from ikpy.chain import Chain

try:
    import rbpodo as rb
except Exception:
    rb = None

# ================= Config =================
BASE_LINK = "link0"
EE_LINK   = "tcp"

# ROS 퍼블리시/구독 기준 6개 조인트 이름 (순서 중요)
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

URDF_PATH  = "/home/sungboo/ros2_ws/src/rbpodo_ros2/rbpodo_description/robots/rb10_1300e_u.urdf"

ROBOT_IP = "10.0.2.7"
RBPODO_SIMULATION = False
RBPODO_SPEED_BAR = 0.5
RBPODO_DATA_TIMEOUT = 0.05
RBPODO_SERVO_T1 = 0.01   # rbpodo_ros2 default
RBPODO_SERVO_T2 = 0.10   # rbpodo_ros2 default
RBPODO_SERVO_GAIN = 0.5  # rbpodo_ros2 default
RBPODO_SERVO_ALPHA = 0.5 # rbpodo_ros2 default

# IKPy active mask (fixed=False, 가동조인트=True, 최종 tcp는 프레임만 포함하므로 False)
ACTIVE_LINKS_MASK = [False, True, True, True, True, True, True, False]

# 안전가드
MAX_STEP_PER_JOINT_RAD = 0.20
MAX_STEP_L2_RAD        = 0.40
IK_MAX_ITER            = 80

# 디버그 토글
DEBUG = False


class RB10Controller(Node):
    """ BASE(frame) 기준 pos(3) + quat(xyzw) -> IK -> rbpodo move_servo_j publish """
    def __init__(
        self,
        robot_ip: str = ROBOT_IP,
        simulation: bool = RBPODO_SIMULATION,
        speed_bar: float = RBPODO_SPEED_BAR,
        data_timeout: float = RBPODO_DATA_TIMEOUT,
        wait_for_initial_data: bool = True,
        initial_data_timeout_sec: float = 5.0,
    ):
        super().__init__("rb10_controller")

        # 외부에서 확인 가능한 최근 IK 실패 사유
        self.last_ik_fail: Optional[str] = None
        self._robot_ip = str(robot_ip)
        self._data_timeout = float(data_timeout)

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

        # 최신 관절각(6개, JOINT_NAMES 순, rad)
        self._latest_positions: Optional[np.ndarray] = None

        # rbpodo direct command channel only (no fallback)
        if rb is None:
            raise RuntimeError("rbpodo import failed. Install rbpodo to use rb10_controller.")
        self._rb_robot = rb.Cobot(self._robot_ip)
        self._rb_rc = rb.ResponseCollector()
        self._rb_data = rb.CobotData(self._robot_ip)
        if simulation:
            self._rb_robot.set_operation_mode(self._rb_rc, rb.OperationMode.Simulation)
        self._rb_robot.set_speed_bar(self._rb_rc, float(speed_bar))
        self._rb_robot.flush(self._rb_rc)
        self.get_logger().info(f"rbpodo direct enabled (ip={self._robot_ip}, sim={simulation})")

        if wait_for_initial_data:
            # 첫 rbpodo 데이터 대기
            self.get_logger().info("Waiting for first rbpodo joint data...")
            end = self.get_clock().now().nanoseconds + int(float(initial_data_timeout_sec) * 1e9)
            while rclpy.ok() and (self.get_clock().now().nanoseconds < end) and (self._latest_positions is None):
                self.get_current_joint_states()
                time.sleep(0.02)

            if self._latest_positions is None:
                self.get_logger().warn("No rbpodo joint data received.")
            else:
                self.get_logger().info("Got initial rbpodo joint data.")

    # ---------- 공용 진단 헬퍼 ----------
    def _ik_fail(self, reason: str, extra: Optional[str] = None) -> None:
        self.last_ik_fail = reason if extra is None else f"{reason} | {extra}"
        self.get_logger().warn(self.last_ik_fail)

    def _request_rbpodo_sdata(self):
        try:
            data = self._rb_data.request_data(self._data_timeout)
            if data is None or getattr(data, "sdata", None) is None:
                return None
            return data.sdata
        except Exception as e:
            if DEBUG:
                self.get_logger().warn(f"rbpodo request_data failed: {e}")
            return None

    # ---------- q6 <-> q_full ----------
    def _q_full_from_q6(self, q6: np.ndarray) -> np.ndarray:
        q_full = np.zeros(len(self._link_names), dtype=float)
        q_full[self._idx6] = q6
        return q_full

    def _q6_from_q_full(self, q_full: np.ndarray) -> np.ndarray:
        return q_full[self._idx6].astype(float, copy=False)

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
        target_ee_pos: np.ndarray,               # (3,)
        target_ee_rot: np.ndarray,               # (4,) [xyzw] or (3,) [ypr]
        enforce_guard: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Compute joint angles for desired EE pose.

        Args:
            target_ee_pos: [x, y, z] in BASE frame
            target_ee_rot: orientation quaternion [x, y, z, w]
            enforce_guard: limit joint step size

        Returns:
            q6: np.ndarray(6,) or None
        """
        seed6 = self.get_current_joint_states()
        if seed6 is None:
            seed6 = self._latest_positions
        if seed6 is None:
            self._ik_fail("No joint state seed available (rbpodo/ROS)")
            return None

        # --- normalize input ---
        p = np.asarray(target_ee_pos, dtype=float).reshape(3,)

        q = np.asarray(target_ee_rot, dtype=float).reshape(4,)
        n = float(np.linalg.norm(q))
        if not np.isfinite(n) or n <= 0:
            self._ik_fail("Invalid target quaternion (norm <= 0 or NaN)")
            return None
        q /= n
        R_tgt = quaternion_matrix(q)[:3, :3]
        
        # --- seed & IK ---
        seed_full = self._q_full_from_q6(seed6)
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

        q6 = self._q6_from_q_full(np.asarray(q_full, dtype=float))
        if enforce_guard and not self._guard_ok(q6, seed6):
            return None

        return q6


    # ---------- rbpodo publish ----------
    def publish_qpos(
        self,
        q_goal: List[float],
        t1: Optional[float] = None,
        t2: Optional[float] = None,
        gain: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> bool:
        try:
            q_goal = list(map(float, q_goal))
            if len(q_goal) != 6:
                self.get_logger().warn(f"publish_qpos expects 6 joints, got {len(q_goal)}")
                return False

            q_goal_rad = np.asarray(q_goal, dtype=float)
            q_goal_deg = np.rad2deg(q_goal_rad)

            # Servo-j command policy by default.
            # Keep rbpodo_ros2 semantics:
            # use controller config defaults unless explicit t1/t2 are provided.
            cmd_t1 = RBPODO_SERVO_T1 if t1 is None else float(t1)
            cmd_t2 = RBPODO_SERVO_T2 if t2 is None else float(t2)
            cmd_gain = RBPODO_SERVO_GAIN if gain is None else float(gain)
            cmd_alpha = RBPODO_SERVO_ALPHA if alpha is None else float(alpha)

            # API constraints:
            # t1 >= 0.002, 0.02 < t2 < 0.2, gain > 0, 0 < alpha < 1
            cmd_t1 = max(0.002, cmd_t1)
            cmd_t2 = float(np.clip(cmd_t2, 0.021, 0.199))
            cmd_gain = max(1e-6, cmd_gain)
            cmd_alpha = float(np.clip(cmd_alpha, 0.001, 0.999))

            self._rb_robot.move_servo_j(self._rb_rc, q_goal_deg, cmd_t1, cmd_t2, cmd_gain, cmd_alpha)
            return True
        except Exception as e:
            self.get_logger().warn(f"rbpodo move_servo_j failed: {e}")
            return False

    # ---------- 현재 EE pose ----------
    def get_current_joint_states(self) -> Optional[np.ndarray]:
        """ Current joints from rbpodo sdata.jnt_ang (rad, JOINT_NAMES order) """
        s = self._request_rbpodo_sdata()
        if s is None:
            return None
        try:
            q_deg = np.array([float(s.jnt_ang[i]) for i in range(6)], dtype=float)
            q_rad = np.deg2rad(q_deg)
            self._latest_positions = q_rad.copy()
            return q_rad
        except Exception as e:
            if DEBUG:
                self.get_logger().warn(f"Failed to parse jnt_ang: {e}")
            return None

    def get_current_ee_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """ Current EE pose from rbpodo sdata.tcp_pos -> (pos[m], quat[xyzw]) """
        s = self._request_rbpodo_sdata()
        if s is None:
            return None
        try:
            tcp = [float(s.tcp_pos[i]) for i in range(6)]  # [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            pos = np.array([tcp[0], tcp[1], tcp[2]], dtype=float) / 1000.0
            rx = math.radians(tcp[3])
            ry = math.radians(tcp[4])
            rz = math.radians(tcp[5])
            qx, qy, qz, qw = quaternion_from_euler(rz, ry, rx, axes="rzyx")
            quat = np.array([qx, qy, qz, qw], dtype=float)
            return pos, quat
        except Exception as e:
            if DEBUG:
                self.get_logger().warn(f"Failed to parse tcp_pos: {e}")
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
            node.publish_qpos(q.tolist())
            if DEBUG:
                node.get_logger().info("Moved to target pose.")
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":
    main()
