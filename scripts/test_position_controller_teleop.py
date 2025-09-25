#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
teleop_ikfast_stream — RB10 텔레옵용 IKFast 스트리밍 노드 (ROS 2 Humble)

목표:
- joint_states(FK) → 현재 EE 자세 계산
- 외부 텔레옵 입력을 받아 (증분 ΔTwist 또는 절대 Pose)
- IKFast(ikfastpy)로 관절해 in-process 계산
- position_controllers 로 30 Hz 이상 연속 publish (오픈루프 안정)

입력 토픽(선택):
- /teleop/ee_delta  (geometry_msgs/TwistStamped): base 프레임 기준 EE 선형/각속도 명령
- /teleop/ee_pose   (geometry_msgs/PoseStamped):   base 프레임 기준 EE 절대 목표 포즈

출력 토픽:
- /position_controllers/commands (std_msgs/Float64MultiArray): JointGroupPositionController

주의:
- 오픈루프 컨트롤러이므로 스트리밍(연속 publish)이 기본입니다.
- 안전 가드(MAX_STEP_*)를 넘는 IK 해는 버립니다.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped, PoseStamped

import ikfastpy
from tf_transformations import quaternion_matrix

# ===================== 사용자 환경에 맞게 수정 =====================
CONTROLLER_CMD_TOPIC = "/position_controllers/commands"
JOINT_STATES_TOPIC   = "/rbpodo/joint_states"
DELTA_TOPIC          = "/teleop/ee_delta"   # 증분 입력 (TwistStamped)
POSE_TOPIC           = "/teleop/ee_pose"    # 절대 Pose 입력 (PoseStamped)

PUBLISH_RATE_HZ      = 30.0  # 스트리밍 publish 주기
BASE_LINK            = "link0"
EE_LINK              = "tcp"

# JointGroupPositionController 의 조인트 순서와 동일해야 함
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

# 안전 가드: seed 대비 과도한 점프 방지
MAX_STEP_PER_JOINT_RAD = 0.15   # ≈ 8.6°
MAX_STEP_L2_RAD        = 0.40

# 증분 명령에 대한 게인 (Twist → Δx 적분)
LIN_GAIN = 0.2   # m/s → m (dt로 스케일됨)
ANG_GAIN = 0.0   # rad/s → rad (회전 증분 비활성: 0.0)

# 포즈 명령 유지 시간(hold) — 오픈루프 안정화 보조
HOLD_SEC = 0.05
# ===============================================================


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


@dataclass
class DeltaCmd:
    lin: np.ndarray  # shape (3,) in base frame
    ang: np.ndarray  # shape (3,) in base frame
    stamp_ns: int


class TeleopIKFastStreamer(Node):
    def __init__(self):
        super().__init__("teleop_ikfast_stream")

        # Publishers / Subscribers
        self.cmd_pub = self.create_publisher(Float64MultiArray, CONTROLLER_CMD_TOPIC, 10)
        self.joint_sub = self.create_subscription(JointState, JOINT_STATES_TOPIC, self._on_joint, 50)
        self.delta_sub = self.create_subscription(TwistStamped, DELTA_TOPIC, self._on_delta, 10)
        self.pose_sub  = self.create_subscription(PoseStamped,  POSE_TOPIC,  self._on_pose,  10)

        # IKFast
        self.ik = ikfastpy.PyKinematics()  # RB10용으로 빌드된 바인딩이어야 함
        self.n_joints = self.ik.getDOF()
        if self.n_joints != len(JOINT_NAMES):
            self.get_logger().warn(f"IKFast DOF({self.n_joints}) != JOINT_NAMES({len(JOINT_NAMES)}) — 확인 필요")

        # State
        self._name_to_idx = None
        self._q: Optional[np.ndarray] = None           # 최신 관절각 (JOINT_NAMES 순서)
        self._delta_cmd: Optional[DeltaCmd] = None     # 최신 Δ 명령
        self._abs_pose: Optional[PoseStamped] = None   # 최신 절대 Pose 명령

        # Timer for streaming
        self._dt = 1.0 / PUBLISH_RATE_HZ
        self._timer = self.create_timer(self._dt, self._tick)

        self.get_logger().info("Waiting for joint_states...")
        end = time.time() + 5.0
        while rclpy.ok() and self._q is None and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._q is None:
            self.get_logger().warn("No joint_states received. 실로봇/시뮬레이터 상태 확인.")
        else:
            self.get_logger().info(f"Got initial joints: {[f'{v:.3f}' for v in self._q.tolist()]}")

    # --------------------- Callbacks ---------------------
    def _on_joint(self, msg: JointState):
        if self._name_to_idx is None:
            self._name_to_idx = {n: i for i, n in enumerate(msg.name)}
            missing = [n for n in JOINT_NAMES if n not in self._name_to_idx]
            if missing:
                self.get_logger().warn(f"JointState에 없는 조인트: {missing}")
        q = []
        for n in JOINT_NAMES:
            idx = self._name_to_idx.get(n)
            q.append(msg.position[idx] if idx is not None and idx < len(msg.position) else 0.0)
        self._q = np.asarray(q, dtype=float)

    def _on_delta(self, msg: TwistStamped):
        lin = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], dtype=float)
        ang = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z], dtype=float)
        self._delta_cmd = DeltaCmd(lin=lin, ang=ang, stamp_ns=msg.header.stamp.nanosec)
        # 절대 포즈 명령이 들어와 있다면 증분 명령이 덮어쓰도록(더 최신 입력 우선)
        self._abs_pose = None

    def _on_pose(self, msg: PoseStamped):
        # base 프레임 기준 Pose만 허용 (필요 시 변환 추가)
        if msg.header.frame_id and msg.header.frame_id != BASE_LINK:
            self.get_logger().warn(f"POSE frame '{msg.header.frame_id}' != BASE_LINK '{BASE_LINK}' — 무시")
            return
        self._abs_pose = msg
        self._delta_cmd = None

    # --------------------- Core Ops ---------------------
    def _fk_T(self, q: np.ndarray) -> Optional[np.ndarray]:
        try:
            ee_pose = self.ik.forward(q.tolist())
            T = np.eye(4)
            T[:3, :4] = np.asarray(ee_pose, dtype=float).reshape(3, 4)
            return T
        except Exception as e:
            self.get_logger().warn(f"FK 실패: {e}")
            return None

    def _ik_from_T(self, T_target: np.ndarray, seed: np.ndarray) -> Optional[np.ndarray]:
        try:
            raw = self.ik.inverse(T_target[:3, :4].reshape(-1).tolist())
        except Exception as e:
            self.get_logger().warn(f"IK 실패: {e}")
            return None
        if raw is None or len(raw) == 0:
            return None
        sols = np.asarray(raw, dtype=float).reshape(-1, self.n_joints)
        # seed와 가장 가까운 해 선택
        diffs = angle_diff(sols, seed)
        idx = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        q_sel = sols[idx]
        # 안전 가드
        step = angle_diff(q_sel, seed)
        if np.any(np.abs(step) > MAX_STEP_PER_JOINT_RAD) or np.linalg.norm(step) > MAX_STEP_L2_RAD:
            self.get_logger().warn(
                "IK 점프 과다 — 취소: max|Δ|=%.3f, ||Δ||2=%.3f" % (np.max(np.abs(step)), np.linalg.norm(step))
            )
            return None
        return q_sel

    def _apply_delta_cart(self, T: np.ndarray, dt: float, lin: np.ndarray, ang: np.ndarray) -> np.ndarray:
        # 단순 평행이동만 적용 (기본): 회전 증분은 ANG_GAIN=0.0 로 비활성
        Td = np.eye(4)
        Td[0, 3] = LIN_GAIN * lin[0] * dt
        Td[1, 3] = LIN_GAIN * lin[1] * dt
        Td[2, 3] = LIN_GAIN * lin[2] * dt
        # 회전 증분을 쓰고 싶다면 여기에서 소각 회전 행렬을 곱해줌
        return T @ Td

    def _publish_once(self, q_cmd: np.ndarray):
        msg = Float64MultiArray(); msg.data = q_cmd.tolist()
        self.cmd_pub.publish(msg)

    # --------------------- Main Tick ---------------------
    def _tick(self):
        if self._q is None:
            return

        seed = self._q.copy()
        T_cur = self._fk_T(seed)
        if T_cur is None:
            return

        # 1) 절대 포즈 명령 우선
        if self._abs_pose is not None:
            T_tgt = self._pose_to_T(self._abs_pose)
            q_sel = self._ik_from_T(T_tgt, seed)
            if q_sel is not None:
                self._publish_once(q_sel)
                self._q = q_sel  # 내부 시드 갱신
            return

        # 2) 증분 명령
        if self._delta_cmd is not None:
            T_tgt = self._apply_delta_cart(T_cur, self._dt, self._delta_cmd.lin, self._delta_cmd.ang)
            q_sel = self._ik_from_T(T_tgt, seed)
            if q_sel is not None:
                self._publish_once(q_sel)
                self._q = q_sel
            return

        # 3) 입력 없으면 아무 것도 하지 않음 (마지막 자세 유지)
        return

    # --------------------- Utils ---------------------
    @staticmethod
    def _pose_to_T(ps: PoseStamped) -> np.ndarray:
        q = ps.pose.orientation
        T = quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0, 3] = ps.pose.position.x
        T[1, 3] = ps.pose.position.y
        T[2, 3] = ps.pose.position.z
        return T


def main():
    rclpy.init()
    node = TeleopIKFastStreamer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()