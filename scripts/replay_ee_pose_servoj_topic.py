#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replay EE pose array in policy-rate topic publishing while servo loop runs at higher fixed rate.

Goal:
- Policy loop: publish target pose at low rate (e.g., 30 Hz)
- Servo loop: run controller timer at high rate (e.g., 100 Hz)
- Verify decoupling with recv_count vs ctrl_count stats

Usage:
python3 scripts/replay_ee_pose_servoj_topic.py --policy-hz 30 --control-hz 100 --duration 8

정상이라면 마지막에 대략 이런 형태로 나옵니다:

controller_ctrl가 controller_recv보다 유의미하게 큼
measured_ctrl_hz가 measured_policy_hz보다 큼
rate_split_ok=True
"""

import argparse
import math
import time
from typing import List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf_transformations import quaternion_from_euler, quaternion_multiply

from scripts.rbpodo_controller_servoj_topic import RB10ServoJPoseTopicController


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4,)
    n = float(np.linalg.norm(q))
    if n <= 0.0 or not np.isfinite(n):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def make_test_ee_pose_array(
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    hz: float,
    duration_sec: float,
    radius_xy_m: float,
    z_amp_m: float,
    yaw_amp_rad: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = max(2, int(float(hz) * float(duration_sec)))
    dt = 1.0 / float(hz)
    poses: List[Tuple[np.ndarray, np.ndarray]] = []

    base_quat = _normalize_quat(start_quat)
    cx, cy, cz = map(float, start_pos)
    duration_sec = max(1e-6, float(duration_sec))

    for i in range(n):
        t = i * dt
        phase = 2.0 * math.pi * (t / duration_sec)

        x = cx + float(radius_xy_m) * math.cos(phase)
        y = cy + float(radius_xy_m) * math.sin(phase)
        z = cz + float(z_amp_m) * math.sin(2.0 * phase)

        yaw = float(yaw_amp_rad) * math.sin(phase)
        dq = quaternion_from_euler(0.0, 0.0, yaw, axes="sxyz")
        quat = _normalize_quat(quaternion_multiply(base_quat, dq))
        poses.append((np.array([x, y, z], dtype=float), quat))

    return poses


class PolicyPosePublisher(Node):
    def __init__(self, poses: List[Tuple[np.ndarray, np.ndarray]], policy_hz: float) -> None:
        super().__init__("policy_pose_publisher")
        self._poses = poses
        self._idx = 0
        self._sent = 0
        self._pub = self.create_publisher(PoseStamped, "/rb10_controller/target_pose", 10)
        self._timer = self.create_timer(1.0 / float(policy_hz), self._on_timer)

    @property
    def sent_count(self) -> int:
        return self._sent

    @property
    def done(self) -> bool:
        return self._idx >= len(self._poses)

    def _on_timer(self) -> None:
        if self._idx >= len(self._poses):
            self._timer.cancel()
            return

        p, q = self._poses[self._idx]
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "link0"
        msg.pose.position.x = float(p[0])
        msg.pose.position.y = float(p[1])
        msg.pose.position.z = float(p[2])
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self._pub.publish(msg)

        self._idx += 1
        self._sent += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Topic-based EE replay with policy/servo rate split verification")
    p.add_argument("--policy-hz", type=float, default=30.0, help="policy publish rate (Hz)")
    p.add_argument("--control-hz", type=float, default=100.0, help="servo controller loop rate (Hz)")
    p.add_argument("--duration", type=float, default=8.0, help="replay duration (sec)")
    p.add_argument("--radius", type=float, default=0.005, help="xy circle radius (m)")
    p.add_argument("--z-amp", type=float, default=0.003, help="z oscillation amplitude (m)")
    p.add_argument("--yaw-amp", type=float, default=0.05, help="yaw oscillation amplitude (rad)")
    p.add_argument("--tail-sec", type=float, default=0.5, help="extra spin time after last policy sample")
    p.add_argument("--no-guard", action="store_true", help="disable IK safety guard")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()

    controller = RB10ServoJPoseTopicController(
        control_hz=float(args.control_hz),
        enforce_guard=(not args.no_guard),
    )
    publisher: PolicyPosePublisher = None
    executor = MultiThreadedExecutor(num_threads=2)
    t0 = time.perf_counter()

    try:
        current = controller.get_current_ee_pose()
        if current is None:
            controller.get_logger().error("Cannot read current EE pose from rbpodo.")
            return
        start_pos, start_quat = current
        poses = make_test_ee_pose_array(
            start_pos=np.asarray(start_pos, dtype=float),
            start_quat=np.asarray(start_quat, dtype=float),
            hz=float(args.policy_hz),
            duration_sec=float(args.duration),
            radius_xy_m=float(args.radius),
            z_amp_m=float(args.z_amp),
            yaw_amp_rad=float(args.yaw_amp),
        )

        publisher = PolicyPosePublisher(poses, float(args.policy_hz))
        executor.add_node(controller)
        executor.add_node(publisher)

        while rclpy.ok() and not publisher.done:
            executor.spin_once(timeout_sec=0.1)

        tail_end = time.perf_counter() + max(0.0, float(args.tail_sec))
        while rclpy.ok() and (time.perf_counter() < tail_end):
            executor.spin_once(timeout_sec=0.1)

        elapsed = max(1e-6, time.perf_counter() - t0)
        stats = controller.get_loop_stats()
        sent = publisher.sent_count
        recv_count = int(stats["recv_count"])
        ctrl_count = int(stats["ctrl_count"])
        policy_rate_meas = sent / elapsed
        ctrl_rate_meas = ctrl_count / elapsed
        ratio = (ctrl_count / max(1, recv_count))

        print("\n=== Topic Replay Verification ===")
        print(f"policy_hz_target={float(args.policy_hz):.2f}, control_hz_target={float(args.control_hz):.2f}")
        print(f"elapsed={elapsed:.3f}s, policy_sent={sent}, controller_recv={recv_count}, controller_ctrl={ctrl_count}")
        print(f"measured_policy_hz={policy_rate_meas:.2f}, measured_ctrl_hz={ctrl_rate_meas:.2f}, ctrl_per_recv={ratio:.2f}")

        split_ok = (ctrl_rate_meas > (policy_rate_meas * 1.5)) and (ctrl_count > recv_count)
        print(f"rate_split_ok={split_ok}")
        if not split_ok:
            print("WARN: control loop did not run significantly faster than policy publish rate.")
    finally:
        if publisher is not None:
            publisher.destroy_node()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
