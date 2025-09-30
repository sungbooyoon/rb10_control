#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from rclpy.action import ActionClient
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
import numpy as np
from tf_transformations import quaternion_matrix

# ===== IKPy =====
from ikpy.chain import Chain

JOINT_TRAJ_ACTION = "/joint_trajectory_controller/follow_joint_trajectory"
JOINT_STATES_TOPIC = "/rbpodo/joint_states"
PUBLISH_RATE_HZ = 30
BASE_LINK = "link0"
EE_LINK = "tcp"
DELTA_XYZ = (0.02, 0.02, 0.02)
HOLD_SEC = 1.0
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

URDF_PATH = "/home/sungboo/ros2_ws/src/rbpodo_ros2/rbpodo_description/robots/rb10_1300e_u.urdf"

ACTIVE_LINKS_MASK = [False, True, True, True, True, True, True, False]
BASE_ELEMENTS = [BASE_LINK]

MAX_STEP_PER_JOINT_RAD = 0.15
MAX_STEP_L2_RAD = 0.40

IK_MAX_ITER = 50

# ===== Controller I/O mode =====
# If True, use FollowJointTrajectory Action (current default)
# If False, publish directly to the controller's topic (no feedback/result)
USE_JTC_ACTION = False
JTC_TOPIC = "/joint_trajectory_controller/joint_trajectory"
# Single-point trajectory in topic mode: controller will start from current state
USE_SINGLE_POINT_TRAJ = True


class JointTrajectoryControllerTester(Node):
    def __init__(self):
        super().__init__("joint_trajectory_controller_tester")
        self.joint_sub = self.create_subscription(JointState, JOINT_STATES_TOPIC, self._joint_cb, 10)
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        # I/O to JointTrajectoryController
        self.traj_pub = self.create_publisher(JointTrajectory, JTC_TOPIC, 10)
        if USE_JTC_ACTION:
            self.traj_action = ActionClient(self, FollowJointTrajectory, JOINT_TRAJ_ACTION)
            self.get_logger().info(f"Waiting for JointTrajectory action server at {JOINT_TRAJ_ACTION} ...")
            self.traj_action.wait_for_server(timeout_sec=3.0)
        else:
            self.traj_action = None
            self.get_logger().info(f"Using topic-based control: publishing JointTrajectory to {JTC_TOPIC}")
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
        try:
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

    def send_trajectory(self, q_start: List[float], q_goal: List[float], duration: float = 0.6) -> bool:
        """
        Send a simple 2-point trajectory (current -> goal) to JointTrajectoryController.
        If USE_JTC_ACTION is True, send via FollowJointTrajectory action and wait for the result.
        Otherwise, publish trajectory to the topic JTC_TOPIC (fire-and-forget, no feedback).
        duration: seconds from start for the goal point.
        """
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        p0 = JointTrajectoryPoint()
        p0.positions = list(q_start)
        p0.time_from_start = Duration(seconds=0.0).to_msg()

        p1 = JointTrajectoryPoint()
        p1.positions = list(q_goal)
        p1.time_from_start = Duration(seconds=float(max(0.2, duration))).to_msg()

        # Choose fastest representation in topic mode: single-point message (controller starts from current state)
        if not USE_JTC_ACTION and USE_SINGLE_POINT_TRAJ:
            traj.points = [p1]
        else:
            traj.points = [p0, p1]

        if USE_JTC_ACTION:
            if self.traj_action is None or not self.traj_action.server_is_ready():
                self.get_logger().warn("JointTrajectory action server not ready. Is joint_trajectory_controller active?")
                return False
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj
            send_future = self.traj_action.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future)
            goal_handle = send_future.result()
            if goal_handle is None or not goal_handle.accepted:
                self.get_logger().warn("Trajectory goal rejected by controller.")
                return False
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result()
            ok = (result is not None and getattr(result.result, "error_code", 0) == 0)
            if ok:
                self.get_logger().info("Trajectory execution succeeded (action).")
            else:
                self.get_logger().warn(f"Trajectory finished with error_code={getattr(result.result, 'error_code', None)}")
            return ok
        else:
            # Topic-based: publish once (controller will consume the latest message)
            self.traj_pub.publish(traj)
            pts = len(traj.points)
            self.get_logger().info(f"Trajectory published to topic (points={pts}, no action feedback).")
            return True

    def command_joints(self, q_cmd: List[float], hold_sec: float = 0.2, rate_hz: int = PUBLISH_RATE_HZ):
        """
        Use JointTrajectoryController to move to q_cmd over hold_sec seconds.
        In topic mode (USE_JTC_ACTION=False) this is fire-and-forget (no feedback).
        rate_hz is ignored (kept for API compatibility).
        """
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신 — 현재 관절각을 알 수 없습니다.")
            return
        self.send_trajectory(self._latest_positions, q_cmd, duration=max(hold_sec, 0.3))

    def move_to_pose(self, target_pose: PoseStamped, hold_sec: float = 0.3, enforce_guard: bool = True) -> bool:
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신")
            return False
        q = self.ik_from_pose(target_pose, seed=self._latest_positions, enforce_guard=enforce_guard)
        if q is None:
            return False
        self.send_trajectory(self._latest_positions, q.tolist(), duration=max(hold_sec, 0.3))
        return True

    def move_to_T(self, T_target: np.ndarray, hold_sec: float = 0.3, enforce_guard: bool = True) -> bool:
        if self._latest_positions is None:
            self.get_logger().warn("joint_states 미수신")
            return False
        q = self.ik_from_T(T_target, seed=self._latest_positions, enforce_guard=enforce_guard)
        if q is None:
            return False
        self.send_trajectory(self._latest_positions, q.tolist(), duration=max(hold_sec, 0.3))
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
        self.send_trajectory(self._latest_positions, q_sel.tolist(), duration=max(hold_sec, 0.3))


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

        self.send_trajectory(self._latest_positions, q.tolist(), duration=max(hold_sec, 0.3))
        return True


def main():
    rclpy.init()
    node = JointTrajectoryControllerTester()
    try:
        node.move_ee_to_position(0.85, 0.20, 0.55, hold_sec=HOLD_SEC, keep_current_orientation=True, enforce_guard=False)
        node.get_logger().info("Done. A 2-point trajectory was sent to JointTrajectoryController (IKPy).")
        
        # rate_hz = 30.0
        # period = 1.0 / rate_hz
        # node.get_logger().info(f"Starting main loop at {rate_hz:.1f} Hz (period={period*1000:.1f} ms)")
        # start_time = time.time()
        # loop_count = 0
        # while rclpy.ok() and loop_count < 100:
        #     t0 = time.time()
        #     node.move_ee_to_position(0.85, 0.20, 0.55, hold_sec=HOLD_SEC, keep_current_orientation=True, enforce_guard=False)
        #     t1 = time.time()
        #     loop_dt = t1 - t0
        #     node.get_logger().info(f"Loop {loop_count}: dt={loop_dt*1000:.1f} ms")
        #     sleep_time = period - (time.time() - t0)
        #     if sleep_time > 0:
        #         time.sleep(sleep_time)
        #     loop_count += 1

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
