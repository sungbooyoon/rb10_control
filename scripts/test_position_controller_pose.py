#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
import numpy as np
from tf_transformations import quaternion_matrix

# ===== IKPy =====
from ikpy.chain import Chain

# === 사용 환경에 맞게 필요하면 수정 ===
CONTROLLER_CMD_TOPIC = "/position_controllers/commands"   # JointGroupPositionController 입력
JOINT_STATES_TOPIC   = "/rbpodo/joint_states"             # launch에서 remap한 joint_states
PUBLISH_RATE_HZ      = 30
BASE_LINK  = "link0"  # URDF상의 base 링크명
EE_LINK    = "tcp"    # URDF상의 EE 링크명 (tool0, tcp 등 프로젝트에 맞게)
DELTA_XYZ  = (0.01, 0.01, 0.01)  # EE를 소폭 이동할 거리(m): (dx, dy, dz)
HOLD_SEC   = 1.0
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]  # 컨트롤러가 기대하는 순서

URDF_PATH = "/home/sungboo/ros2_ws/src/rbpodo_ros2/rbpodo_description/robots/rb10_1300e_u.urdf"

# IKPy 체인 구성 옵션
#   - IKPy는 종종 base용 가상 조인트/링크(고정)가 포함됩니다.
#   - active_links_mask 길이는 체인의 링크 수와 같아야 합니다.
#     아래는 예시(mask를 모르면 None으로 두세요 → 모든 구동 조인트 사용)
ACTIVE_LINKS_MASK = [False, True, True, True, True, True, True, False]
BASE_ELEMENTS = [BASE_LINK]  # 이걸로 base를 지정하면 체인 구성 안정적

# 안전 가드: IK 해가 seed(현재 관절)에서 너무 멀리 튀면 실행하지 않음
MAX_STEP_PER_JOINT_RAD = 0.15   # 각 조인트 당 허용 최대 변화량 (rad) ≈ 8.6°
MAX_STEP_L2_RAD        = 0.40   # 전체 변화량의 L2 노름 상한 (rad)

# IKPy 튜닝(필요 시)
IK_MAX_ITER = 50


class PositionControllerTester(Node):
    def __init__(self):
        super().__init__("position_controller_tester")

        self.cmd_pub = self.create_publisher(Float64MultiArray, CONTROLLER_CMD_TOPIC, 10)
        self.joint_sub = self.create_subscription(JointState, JOINT_STATES_TOPIC, self._joint_cb, 10)

        # TF 초기화
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)

        # === IKPy 체인 구성 ===
        self.chain_full = Chain.from_urdf_file(
            URDF_PATH,
            base_elements=BASE_ELEMENTS
        )
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
            self.get_logger().warn("No joint_states received. "
                                   "실로봇/시뮬레이터가 joint_states를 내보내고 있는지 확인하세요.")
        else:
            self.get_logger().info(f"Got initial joint_states: {self._latest_positions}")

    def _joint_cb(self, msg: JointState):
        # name->index 맵핑
        if self._joint_index_map is None:
            self._joint_index_map = {name: i for i, name in enumerate(msg.name)}
            missing = [n for n in JOINT_NAMES if n not in self._joint_index_map]
            if missing:
                self.get_logger().warn(f"JointState에 없는 조인트가 있습니다: {missing}")

        positions = [0.0] * len(JOINT_NAMES)
        for i, name in enumerate(JOINT_NAMES):
            idx = self._joint_index_map.get(name, None)
            if idx is None or idx >= len(msg.position):
                positions[i] = positions[i] if self._latest_positions else 0.0
            else:
                positions[i] = msg.position[idx]
        self._latest_positions = positions

    # ===== IKPy FK/IK =====
    def _fk_current_T(self) -> Optional[np.ndarray]:
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신 — FK 불가")
            return None
        try:
            # IKPy는 4x4 homogeneous matrix 반환
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

    def ik_from_T(self, T_target: np.ndarray, seed: np.ndarray | List[float],
                   enforce_guard: bool = True) -> Optional[np.ndarray]:
        """IKPy inverse — 4x4 변환행렬 입력"""
        try:
            # orientation_weight로 회전 중요도 조절
            q = self.chain.inverse_kinematics_frame(
                T_target,
                initial_position=np.asarray(seed, dtype=float),
                max_iter=IK_MAX_ITER
            )
        except Exception as e:
            self.get_logger().warn(f"IKPy inverse 실패: {e}")
            return None

        if q is None or len(q) != len(JOINT_NAMES):
            return None

        q = np.asarray(q, dtype=float)

        if enforce_guard:
            diffs = self._angle_diff_matrix(q.reshape(1, -1), np.asarray(seed, dtype=float)).reshape(-1)
            l2 = float(np.linalg.norm(diffs))
            if np.any(np.abs(diffs) > MAX_STEP_PER_JOINT_RAD) or l2 > MAX_STEP_L2_RAD:
                self.get_logger().warn(
                    "IK 해 변화량 과다로 명령 취소: "
                    f"max|Δ|={np.max(np.abs(diffs)):.3f} rad, ||Δ||2={l2:.3f} rad"
                )
                return None

        return q

    def ik_from_pose(self, pose: PoseStamped, seed: np.ndarray | List[float],
                     enforce_guard: bool = True) -> Optional[np.ndarray]:
        T = self._pose_to_T(pose)
        return self.ik_from_T(T, seed, enforce_guard=enforce_guard)

    # ===== 유틸 =====
    def _log_fk_pose(self, q: List[float], label: str = "state"):
        T = self.fk_from_joints(q)
        if T is None:
            self.get_logger().warn(f"[{label}] FK 실패")
            return
        pos = T[:3, 3]
        self.get_logger().info(f"[{label}] joints={['%.3f'%v for v in q]} "
                               f"pose=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    @staticmethod
    def _pose_to_T(ps: PoseStamped) -> np.ndarray:
        q = ps.pose.orientation
        T = quaternion_matrix([q.x, q.y, q.z, q.w])  # 4x4
        T[0, 3] = ps.pose.position.x
        T[1, 3] = ps.pose.position.y
        T[2, 3] = ps.pose.position.z
        return T

    @staticmethod
    def _apply_delta(T: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        Td = np.eye(4)
        Td[0, 3] = dx; Td[1, 3] = dy; Td[2, 3] = dz
        return T @ Td

    def _angle_diff_matrix(self, sols: np.ndarray, seed: np.ndarray) -> np.ndarray:
        diffs = sols - seed
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
        return diffs

    # ===== 재사용 API =====
    def set_safety_limits(self, max_per_joint_rad: Optional[float] = None, max_l2_rad: Optional[float] = None):
        global MAX_STEP_PER_JOINT_RAD, MAX_STEP_L2_RAD
        if max_per_joint_rad is not None:
            MAX_STEP_PER_JOINT_RAD = float(max_per_joint_rad)
        if max_l2_rad is not None:
            MAX_STEP_L2_RAD = float(max_l2_rad)
        self.get_logger().info(f"Safety limits -> per_joint={MAX_STEP_PER_JOINT_RAD:.3f} rad, L2={MAX_STEP_L2_RAD:.3f} rad")

    def command_joints(self, q_cmd: List[float], hold_sec: float = 0.2, rate_hz: int = PUBLISH_RATE_HZ):
        """관절각 명령을 일정 시간 동안 주기적으로 송신 (wall-clock 기반, sim_time 영향 없음)"""
        msg = Float64MultiArray(); msg.data = list(q_cmd)
        period = 1.0 / float(rate_hz)
        t_end  = time.monotonic() + hold_sec
        next_t = time.monotonic()

        while rclpy.ok() and time.monotonic() < t_end:
            self.cmd_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)

            next_t += period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)


    def move_to_pose(self, target_pose: PoseStamped, hold_sec: float = 0.3, enforce_guard: bool = True) -> bool:
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신")
            return False
        q = self.ik_from_pose(target_pose, seed=self._latest_positions, enforce_guard=enforce_guard)
        if q is None:
            return False
        self.command_joints(q.tolist(), hold_sec=hold_sec)
        return True

    def move_to_T(self, T_target: np.ndarray, hold_sec: float = 0.3, enforce_guard: bool = True) -> bool:
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신")
            return False
        q = self.ik_from_T(T_target, seed=self._latest_positions, enforce_guard=enforce_guard)
        if q is None:
            return False
        self.command_joints(q.tolist(), hold_sec=hold_sec)
        return True

    def move_ee_small_delta_with_ikpy(self, dx: float, dy: float, dz: float, hold_sec: float):
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신 — 현재 관절각을 알 수 없습니다.")
            return

        T_cur = self._fk_current_T()
        if T_cur is None:
            self.get_logger().warn("FK로 현재 EE Pose를 구하지 못했습니다 — 동작 생략")
            return
        T_tgt = self._apply_delta(T_cur, dx, dy, dz)

        seed = np.asarray(self._latest_positions, dtype=float)
        q_sel = self.ik_from_T(T_tgt, seed, enforce_guard=True)
        if q_sel is None:
            self.get_logger().warn("IK 실패 또는 가드 위반으로 명령 취소")
            return

        # 디버깅용 로그: 현재 상태 vs IK 해
        self._log_fk_pose(self._latest_positions, label="current")
        self._log_fk_pose(q_sel.tolist(), label="ik_solution")

        self.get_logger().info(f"EE Δ=({dx:.3f},{dy:.3f},{dz:.3f}) → q_cmd={[f'{v:.3f}' for v in q_sel.tolist()]}")
        self.command_joints(q_sel.tolist(), hold_sec=hold_sec)


    def move_ee_to_position(self, x: float, y: float, z: float,
                            hold_sec: float = 0.3,
                            keep_current_orientation: bool = True,
                            quat_xyzw: Optional[List[float]] = None,
                            enforce_guard: bool = True) -> bool:
        """
        BASE_LINK 좌표계에서 EE를 (x,y,z) 절대 위치로 이동.
        - keep_current_orientation=True 이면 현재 EE의 자세(orientation)를 유지
        - quat_xyzw가 주어지면 그 자세로 세팅 (예: [qx,qy,qz,qw])
        - 둘 다 False/None이면 단위 자세(회전 없음)를 사용
        """
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신")
            return False

        # 목표 4x4 T 구성
        T = np.eye(4)
        T[0, 3] = float(x); T[1, 3] = float(y); T[2, 3] = float(z)

        if quat_xyzw is not None:
            # 사용자가 지정한 자세 사용
            from tf_transformations import quaternion_matrix
            T[:4, :4] = quaternion_matrix(quat_xyzw)
            T[0, 3] = float(x); T[1, 3] = float(y); T[2, 3] = float(z)
        elif keep_current_orientation:
            # 현재 EE의 자세 유지 (FK가 있으면 FK, 없으면 TF 시도)
            T_cur = self._fk_current_T()
            if T_cur is None:
                # FK가 실패하면 TF에서 현재 EE pose를 시도
                try:
                    t = self.tf_buf.lookup_transform(BASE_LINK, EE_LINK, rclpy.time.Time())
                    from tf_transformations import quaternion_matrix
                    T_ori = quaternion_matrix([t.transform.rotation.x,
                                            t.transform.rotation.y,
                                            t.transform.rotation.z,
                                            t.transform.rotation.w])
                    T[:3, :3] = T_ori[:3, :3]
                except Exception as e:
                    self.get_logger().warn(f"현재 자세 조회 실패(FK/TF) → 단위자세 사용: {e}")
            else:
                T[:3, :3] = T_cur[:3, :3]
        # else: 단위자세(기본값)

        # IK + 명령 송신
        q = self.ik_from_T(T, seed=self._latest_positions, enforce_guard=enforce_guard)
        if q is None:
            self.get_logger().warn("IK 실패 또는 가드 위반")
            return False

        # 디버깅 로그
        self._log_fk_pose(self._latest_positions, label="current")
        self._log_fk_pose(q.tolist(), label="ik_solution(target_abs)")

        self.command_joints(q.tolist(), hold_sec=hold_sec)
        return True


def main():
    rclpy.init()
    node = PositionControllerTester()
    try:
        node.move_ee_small_delta_with_ikpy(*DELTA_XYZ, hold_sec=HOLD_SEC)
        node.get_logger().info("Done. You should have seen simple step motions via position controller (IKPy).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
