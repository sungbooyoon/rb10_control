#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import List, Iterable, Callable, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
import numpy as np
import ikfastpy
from tf_transformations import quaternion_matrix, quaternion_from_matrix

# === 사용 환경에 맞게 필요하면 수정 ===
CONTROLLER_CMD_TOPIC = "/position_controllers/commands"   # JointGroupPositionController 입력
JOINT_STATES_TOPIC   = "/rbpodo/joint_states"              # launch에서 remap한 joint_states
PUBLISH_RATE_HZ      = 30
BASE_LINK  = "link0"   # 베이스 프레임 (너의 URDF/SRDF에 맞게 수정)
EE_LINK    = "tcp"       # EE 링크명 (너의 URDF/SRDF에 맞게 수정)
DELTA_XYZ  = (0.02, 0.02, 0.02)  # EE를 소폭 이동할 거리(m): (dx, dy, dz)
HOLD_SEC   = 1.0
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

# 안전 가드: IK 해가 seed(현재 관절)에서 너무 멀리 튀면 실행하지 않음
MAX_STEP_PER_JOINT_RAD = 0.15   # 각 조인트 당 허용 최대 변화량 (rad) ≈ 8.6°
MAX_STEP_L2_RAD        = 0.40   # 전체 변화량의 L2 노름 상한 (rad)

class PositionControllerTester(Node):
    def __init__(self):
        super().__init__("position_controller_tester")

        self.cmd_pub = self.create_publisher(Float64MultiArray, CONTROLLER_CMD_TOPIC, 10)
        self.joint_sub = self.create_subscription(JointState, JOINT_STATES_TOPIC, self._joint_cb, 10)

        # TF와 IKFast 초기화
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.ik = ikfastpy.PyKinematics()  # RB10용으로 빌드된 ikfastpy 바인딩이 로드되어야 함
        self.n_joints = self.ik.getDOF()
        if self.n_joints != len(JOINT_NAMES):
            self.get_logger().warn(f"IKFast DOF({self.n_joints})와 JOINT_NAMES({len(JOINT_NAMES)}) 길이가 다릅니다. 확인 필요!")

        self._latest_positions = None  # type: List[float]
        self._joint_index_map = None   # name -> idx (from first received JointState)

        self.get_logger().info("Waiting for first JointState...")
        # 잠깐 기다리며 첫 joint_states 수신
        end = time.time() + 5.0
        while rclpy.ok() and time.time() < end and self._latest_positions is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._latest_positions is None:
            self.get_logger().warn("No joint_states received. "
                                   "실로봇/시뮬레이터가 joint_states를 내보내고 있는지 확인하세요.")
        else:
            self.get_logger().info(f"Got initial joint_states: {self._latest_positions}")

    def _joint_cb(self, msg: JointState):
        # 첫 수신에서 name->index 맵핑 생성
        if self._joint_index_map is None:
            self._joint_index_map = {name: i for i, name in enumerate(msg.name)}
            missing = [n for n in JOINT_NAMES if n not in self._joint_index_map]
            if missing:
                self.get_logger().warn(f"JointState에 없는 조인트가 있습니다: {missing}")

        # 컨트롤러 순서(JOINT_NAMES)에 맞춰 포지션 배열 재정렬
        positions = [0.0] * len(JOINT_NAMES)
        for i, name in enumerate(JOINT_NAMES):
            idx = self._joint_index_map.get(name, None)
            if idx is None or idx >= len(msg.position):
                # 데이터가 없으면 이전값 유지 or 0.0
                positions[i] = positions[i] if self._latest_positions else 0.0
            else:
                positions[i] = msg.position[idx]
        self._latest_positions = positions

    def _fk_current_T(self) -> np.ndarray | None:
        """현재 joint_states(self._latest_positions)를 사용해 IKFast FK로 EE 4x4 T를 계산"""
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신 — FK 불가")
            return None
        try:
            ee_pose = self.ik.forward(self._latest_positions)
            T34 = np.asarray(ee_pose, dtype=float).reshape(3, 4)
            T = np.eye(4)
            T[:3, :4] = T34
            return T
        except Exception as e:
            self.get_logger().warn(f"IKFast forward 실패: {e}")
            return None

    def fk_from_joints(self, q: List[float]) -> np.ndarray | None:
        """주어진 관절각 q로부터 IKFast FK로 4x4 EE 변환행렬 T를 계산"""
        try:
            ee_pose = self.ik.forward(q)
            T34 = np.asarray(ee_pose, dtype=float).reshape(3, 4)
            T = np.eye(4)
            T[:3, :4] = T34
            return T
        except Exception as e:
            self.get_logger().warn(f"IKFast forward 실패: {e}")
            return None

    def ik_from_T(self, T_target: np.ndarray, seed: np.ndarray | List[float],
                   enforce_guard: bool = True) -> np.ndarray | None:
        """목표 4x4 변환행렬에 대해 IKFast inverse를 호출하고 seed와 가장 가까운 해를 선택.
        enforce_guard=True면 과도한 점프를 가드한다."""
        # IKFast는 3x4 행렬을 1D list로 받음
        T34 = T_target[:3, :4].reshape(-1).tolist()
        try:
            joint_configs = self.ik.inverse(T34)
        except Exception as e:
            self.get_logger().warn(f"IKFast inverse 호출 실패: {e}")
            return None

        if joint_configs is None or len(joint_configs) == 0:
            return None

        n = self.n_joints
        sols = np.asarray(joint_configs, dtype=float).reshape(-1, n)
        seed = np.asarray(seed, dtype=float)
        q_sel = self._closest_solution(sols, seed)
        if q_sel.size == 0:
            return None

        if enforce_guard:
            diffs = self._angle_diff_matrix(q_sel.reshape(1, -1), seed).reshape(-1)
            l2 = float(np.linalg.norm(diffs))
            too_big_joint = bool(np.any(np.abs(diffs) > MAX_STEP_PER_JOINT_RAD))
            if too_big_joint or l2 > MAX_STEP_L2_RAD:
                self.get_logger().warn(
                    "IK 해 변화량 과다로 명령 취소: "
                    f"max|Δ|={np.max(np.abs(diffs)):.3f} rad, ||Δ||2={l2:.3f} rad"
                )
                return None

        return q_sel

    def ik_from_pose(self, pose: PoseStamped, seed: np.ndarray | List[float],
                     enforce_guard: bool = True) -> np.ndarray | None:
        """PoseStamped 입력으로 IK 계산 (내부에서 4x4 T로 변환)"""
        T = self._pose_to_T(pose)
        return self.ik_from_T(T, seed, enforce_guard=enforce_guard)

    def _log_fk_pose(self, q: List[float], label: str = "state"):
        """주어진 q에 대한 FK pose를 계산해 로그로 출력"""
        T = self.fk_from_joints(q)
        if T is None:
            self.get_logger().warn(f"[{label}] FK 실패")
            return
        pos = T[:3, 3]
        self.get_logger().info(f"[{label}] joints={['%.3f'%v for v in q]} "
                               f"pose=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    def _lookup_ee_pose(self) -> PoseStamped | None:
        """TF에서 BASE_LINK 기준 EE_LINK Pose를 얻는다"""
        try:
            t = self.tf_buf.lookup_transform(BASE_LINK, EE_LINK, rclpy.time.Time())
            ps = PoseStamped()
            ps.header.frame_id = BASE_LINK
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = t.transform.translation.x
            ps.pose.position.y = t.transform.translation.y
            ps.pose.position.z = t.transform.translation.z
            ps.pose.orientation = t.transform.rotation
            return ps
        except Exception as e:
            self.get_logger().warn(f"TF lookup 실패: {e}")
            return None

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

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        a = (angle + np.pi) % (2 * np.pi) - np.pi
        return a

    def _closest_solution(self, sols: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """여러 해 중 seed와 가장 가까운 해 선택(각도 래핑 고려)"""
        if sols.size == 0:
            return np.array([])
        diffs = self._angle_diff_matrix(sols, seed)
        idx = np.argmin(np.sum(diffs * diffs, axis=1))
        return sols[idx]

    def _angle_diff_matrix(self, sols: np.ndarray, seed: np.ndarray) -> np.ndarray:
        diffs = sols - seed
        # 각도 래핑 처리
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
        return diffs

    # ===== Reusable Public APIs (for teleop / arbitrary pose commands) =====
    def set_safety_limits(self, max_per_joint_rad: Optional[float] = None, max_l2_rad: Optional[float] = None):
        """런타임에 안전 가드 상한을 조정"""
        global MAX_STEP_PER_JOINT_RAD, MAX_STEP_L2_RAD
        if max_per_joint_rad is not None:
            MAX_STEP_PER_JOINT_RAD = float(max_per_joint_rad)
        if max_l2_rad is not None:
            MAX_STEP_L2_RAD = float(max_l2_rad)
        self.get_logger().info(f"Safety limits -> per_joint={MAX_STEP_PER_JOINT_RAD:.3f} rad, L2={MAX_STEP_L2_RAD:.3f} rad")

    def command_joints(self, q_cmd: List[float], hold_sec: float = 0.2):
        """관절각 명령을 position_controller로 일정 시간 유지 송신"""
        rate = self.create_rate(PUBLISH_RATE_HZ)
        msg = Float64MultiArray(); msg.data = list(q_cmd)
        t_end = time.time() + hold_sec
        while rclpy.ok() and time.time() < t_end:
            self.cmd_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            rate.sleep()

    def move_to_pose(self, target_pose: PoseStamped, hold_sec: float = 0.3, enforce_guard: bool = True) -> bool:
        """목표 PoseStamped로 이동(IKFast), 성공 시 True 반환"""
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신")
            return False
        q = self.ik_from_pose(target_pose, seed=self._latest_positions, enforce_guard=enforce_guard)
        if q is None:
            return False
        self.command_joints(q.tolist(), hold_sec=hold_sec)
        return True

    def move_to_T(self, T_target: np.ndarray, hold_sec: float = 0.3, enforce_guard: bool = True) -> bool:
        """목표 4x4 행렬로 이동(IKFast), 성공 시 True 반환"""
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신")
            return False
        q = self.ik_from_T(T_target, seed=self._latest_positions, enforce_guard=enforce_guard)
        if q is None:
            return False
        self.command_joints(q.tolist(), hold_sec=hold_sec)
        return True

    def move_ee_small_delta_with_ikfast(self, dx: float, dy: float, dz: float, hold_sec: float):
        """[예시] 현재 EE Pose에서 (dx,dy,dz)만큼 이동 — 내부적으로 위의 재사용 API를 활용"""
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

    def send_positions(self, positions: List[float]):
        msg = Float64MultiArray()
        msg.data = positions
        self.cmd_pub.publish(msg)

def main():
    rclpy.init()
    node = PositionControllerTester()

    try:
        node.move_ee_small_delta_with_ikfast(*DELTA_XYZ, hold_sec=HOLD_SEC)

        node.get_logger().info("Done. You should have seen simple step motions via position controller.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()