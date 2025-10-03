#!/usr/bin/env python3
"""
ex_cartesian_waypoints.py — Generate circular Cartesian waypoints and execute with user approval.
"""

from threading import Thread
from typing import List
import sys

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from geometry_msgs.msg import Pose

from pymoveit2 import MoveIt2
from pymoveit2.robots import rb10 as robot  # 환경에 맞게 수정

from collision_utils import apply_collision_from_params
from waypoint_generator import circle_waypoints, line_waypoints  # ✅ 외부 모듈 사용

# ---------------- User Settings ----------------
LINE_START = (0.8, 0.0, 0.75) 
LINE_END = (0.8, 0.0, 0.45)

CENTER = (0.8, 0.0, 0.6)     # 중심 (m)
RADIUS = 0.15                 # 반지름 (m)
NUM_WAYPOINTS = 100
PLANE = "xy"                  # "xy" | "yz" | "xz"
# ORIENTATION = (-0.7071, 0.0, 0.0, 0.7071)
ORIENTATION = (0.5, -0.5, 0.5, 0.5)

CARTESIAN_MAX_STEP = 0.005
VEL_SCALE = 0.2
ACC_SCALE = 0.2
# ------------------------------------------------


def _pose_to_tuples(p: Pose):
    pos = (p.position.x, p.position.y, p.position.z)
    quat = (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
    return pos, quat


def main():
    rclpy.init()
    node = Node("ex_cartesian_waypoints")

    # 충돌객체 파라미터 (옵션)
    node.declare_parameter("shape", "box")
    node.declare_parameter("action", "add")
    node.declare_parameter("position", [0.0, 0.0, 0.0])
    node.declare_parameter("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    node.declare_parameter("dimensions", [2.0, 2.0, 0.1])

    cbg = ReentrantCallbackGroup()
    moveit2 = MoveIt2(
        node=node,
        joint_names=robot.joint_names(),
        base_link_name=robot.base_link_name(),
        end_effector_name=robot.end_effector_name(),
        group_name=robot.MOVE_GROUP_ARM,
        callback_group=cbg,
    )

    # Spin 스레드
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    th = Thread(target=executor.spin, daemon=True)
    th.start()
    node.create_rate(1.0).sleep()

    # 속도/가속/충돌회피 설정
    moveit2.max_velocity = VEL_SCALE
    moveit2.max_acceleration = ACC_SCALE
    moveit2.cartesian_avoid_collisions = True

    # ✅ 충돌 객체 반영
    _ = apply_collision_from_params(node=node, moveit2=moveit2)

    # ✅ 원 경로 생성
    # waypoints: List[Pose] = circle_waypoints(CENTER, RADIUS, NUM_WAYPOINTS, PLANE, ORIENTATION)
    waypoints: List[Pose] = line_waypoints(LINE_START, LINE_END, NUM_WAYPOINTS, ORIENTATION)
    node.get_logger().info(f"Generated {NUM_WAYPOINTS} waypoints on {PLANE}-plane")

    # ✅ 첫 번째 waypoint로 선이동
    first_wp = waypoints[0]
    pos0, quat0 = _pose_to_tuples(first_wp)
    node.get_logger().info(f"Move to first waypoint: pos={pos0}, quat={quat0}")

    try:
        moveit2.move_to_pose(position=pos0, quat_xyzw=quat0)
        moveit2.wait_until_executed()
    except Exception as e:
        node.get_logger().error(f"Failed to move to first waypoint: {e}")
        rclpy.shutdown()
        th.join()
        sys.exit(1)

    # ✅ 사용자 승인 대기
    try:
        ans = input("\n[ENTER] 실행 / [n] 취소 >> ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        ans = "n"
    if ans == "n":
        node.get_logger().warn("Execution canceled by user.")
        rclpy.shutdown()
        th.join()
        return

    # ✅ Cartesian traj 실행
    traj = moveit2.plan_cartesian_waypoints(
        waypoints=waypoints,
        max_step=CARTESIAN_MAX_STEP,
        cartesian_fraction_threshold=0.0,
    )

    if traj:
        node.get_logger().info(f"Execute Cartesian trajectory with {len(traj.points)} points")
        moveit2.execute(traj)
        moveit2.wait_until_executed()
    else:
        node.get_logger().error("Cartesian planning failed (fraction == 0?)")

    rclpy.shutdown()
    th.join()


if __name__ == "__main__":
    main()
