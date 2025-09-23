#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # 1) 컨트롤러 구동(이미 MoveIt으로 켜져 있다면 생략 가능)
# # ros2 launch <bringup> ...

# # 2) 노드 실행
# python3 ik_pose_streamer.py

# # 3) 테스트: target pose 퍼블리시(예: x로 5cm 전방)
# ros2 topic pub -r 10 /target_pose geometry_msgs/msg/PoseStamped \
# "{header: {frame_id: world}, pose: {position: {x: 0.8, y: 0.0, z: 0.55}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"

import time
import numpy as np
from typing import Optional, List

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient

import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_matrix, translation_matrix

import ikfastpy


# ================== 사용자 환경 파라미터 ==================
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
BASE_FRAME = "base_link"     # 로봇 기준 프레임
WORLD_FRAME = "world"        # 목표 Pose의 기본 프레임(입력)
IK_RATE_HZ = 30.0            # 30 Hz
POINT_TIME = 0.08            # 목표점 도달 시간 [s] (0.05~0.15 권장)

# 점프 억제 한계 (너무 튀는 해는 스킵)
MAX_PER_JOINT_STEP = 0.08    # [rad/step] 한 주기당 허용 최대 변화량
MAX_STEP_NORM = 0.20         # [rad] 전체 L2 노름 한계

# IK/선택 가중치 (각 관절 가중치; 필요시 튠)
JOINT_WEIGHTS = np.array([1.0, 1.0, 0.8, 0.6, 0.6, 0.5], dtype=float)

# ==========================================================


def angle_wrap(a: np.ndarray) -> np.ndarray:
    """[-pi, pi]로 wrap"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def angular_distance(q1: np.ndarray, q2: np.ndarray, w: np.ndarray) -> float:
    """가중 각거리 (조인트 wrap 고려)"""
    dq = angle_wrap(q1 - q2)
    return float(np.linalg.norm(w * dq))


def pose_to_hmat(pose: PoseStamped) -> np.ndarray:
    """PoseStamped → 4x4 homogeneous matrix (row-major)"""
    t = translation_matrix([pose.pose.position.x,
                            pose.pose.position.y,
                            pose.pose.position.z])
    q = [pose.pose.orientation.x,
         pose.pose.orientation.y,
         pose.pose.orientation.z,
         pose.pose.orientation.w]
    R = quaternion_matrix(q)
    H = np.matmul(t, R)
    return H


class IKPoseStreamer(Node):
    def __init__(self):
        super().__init__("ik_pose_streamer")

        # IKFast
        self.kin = ikfastpy.PyKinematics()
        self.nj = self.kin.getDOF()
        assert self.nj == len(JOINT_NAMES), "IK DOF와 조인트명 개수가 다릅니다."

        # 상태
        self.current_q: Optional[np.ndarray] = None
        self.target_pose: Optional[PoseStamped] = None

        # 구독
        self.create_subscription(JointState, "/joint_states", self.cb_joint_states, 10)
        self.create_subscription(PoseStamped, "/target_pose", self.cb_target_pose, 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=3.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # 액션 클라이언트 (FollowJointTrajectory)
        self.jtc_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory"
        )

        # 타이머 루프 (30 Hz)
        self.timer = self.create_timer(1.0 / IK_RATE_HZ, self.control_cycle)

        self.get_logger().info("IK Pose Streamer started (30 Hz).")

    # --------- 콜백 ----------
    def cb_joint_states(self, msg: JointState):
        # 필요한 조인트만 추출 (순서에 맞춰)
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        q = []
        try:
            for jn in JOINT_NAMES:
                q.append(msg.position[name_to_idx[jn]])
        except KeyError:
            # 아직 해당 조인트가 안 들어왔을 수 있음
            return
        self.current_q = np.array(q, dtype=float)

    def cb_target_pose(self, msg: PoseStamped):
        self.target_pose = msg

    # --------- 제어 루프 ----------
    def control_cycle(self):
        if self.current_q is None or self.target_pose is None:
            return
        if not self.jtc_client.server_is_ready():
            # 액션 서버 준비 기다리는 중
            return

        # 1) Pose를 BASE_FRAME으로 변환
        try:
            pose_base = self.ensure_pose_in_base(self.target_pose)
        except TransformException as e:
            self.get_logger().warn(f"TF transform 실패: {e}")
            return

        # 2) Pose → Homogeneous Matrix → IK
        H = pose_to_hmat(pose_base)  # 4x4
        ee_pose_flat = H.reshape(-1).tolist()
        sols = self.kin.inverse(ee_pose_flat)  # flat list
        if sols is None or len(sols) == 0:
            # IK 실패
            return

        sols = np.asarray(sols, dtype=float).reshape(-1, self.nj)
        q_now = self.current_q.copy()

        # 3) 현재 q와 가장 가까운 해 선택 (가중 거리 최솟값)
        best_q = None
        best_cost = 1e9
        for q_sol in sols:
            cost = angular_distance(q_sol, q_now, JOINT_WEIGHTS)
            if cost < best_cost:
                best_cost = cost
                best_q = q_sol

        if best_q is None:
            return

        # 4) 점프 억제: per-joint / 전체 노름 검사
        dq = angle_wrap(best_q - q_now)
        if np.any(np.abs(dq) > MAX_PER_JOINT_STEP) or np.linalg.norm(dq) > MAX_STEP_NORM:
            # 너무 튀는 해 → 이번 주기 스킵 (홀드)
            self.get_logger().debug("IK 해 점프 과다 → 스킵")
            return

        q_cmd = angle_wrap(q_now + dq)  # 안전상 wrap

        # 5) Trajectory point 구성 (짧은 1-point, preempt 동작)
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        pt = JointTrajectoryPoint()
        pt.positions = q_cmd.tolist()
        pt.time_from_start = Duration(seconds=POINT_TIME).to_msg()
        traj.points = [pt]

        # 6) 액션 goal 전송 (연속 전송 시 이전 goal 자동 preempt)
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj
        goal_msg.goal_time_tolerance = Duration(seconds=0.05).to_msg()

        # fire-and-forget (future 콜백 생략; 주기적으로 계속 보냄)
        self.jtc_client.send_goal_async(goal_msg)

    # --------- 유틸 ----------
    def ensure_pose_in_base(self, pose: PoseStamped) -> PoseStamped:
        """pose를 BASE_FRAME으로 변환 (같으면 그대로 반환)"""
        if pose.header.frame_id == "" or pose.header.frame_id is None:
            # frame_id가 비었으면 world로 가정
            pose.header.frame_id = WORLD_FRAME

        if pose.header.frame_id == BASE_FRAME:
            return pose

        # 최신 TF로 변환
        tf: TransformStamped = self.tf_buffer.lookup_transform(
            BASE_FRAME, pose.header.frame_id, rclpy.time.Time()
        )
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = BASE_FRAME

        # 변환: out = T * pose
        H_src = pose_to_hmat(pose)
        T = self.tf_to_hmat(tf)
        H_out = T @ H_src

        # H→PoseStamped
        out.pose.position.x = H_out[0, 3]
        out.pose.position.y = H_out[1, 3]
        out.pose.position.z = H_out[2, 3]
        # 회전은 행렬→쿼터니언
        import math
        qw = math.sqrt(max(0.0, 1.0 + H_out[0,0] + H_out[1,1] + H_out[2,2])) / 2.0
        qx = (H_out[2,1] - H_out[1,2]) / (4.0 * max(qw, 1e-9))
        qy = (H_out[0,2] - H_out[2,0]) / (4.0 * max(qw, 1e-9))
        qz = (H_out[1,0] - H_out[0,1]) / (4.0 * max(qw, 1e-9))
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw
        return out

    @staticmethod
    def tf_to_hmat(tf: TransformStamped) -> np.ndarray:
        t = translation_matrix([tf.transform.translation.x,
                                tf.transform.translation.y,
                                tf.transform.translation.z])
        q = [tf.transform.rotation.x,
             tf.transform.rotation.y,
             tf.transform.rotation.z,
             tf.transform.rotation.w]
        R = quaternion_matrix(q)
        return np.matmul(t, R)


def main():
    rclpy.init()
    node = IKPoseStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()