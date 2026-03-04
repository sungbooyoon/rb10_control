#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replay EE pose array at fixed 30 Hz using rbpodo move_servo_j controller.

Pose format:
- position: [x, y, z] in meters (BASE frame)
- quaternion: [x, y, z, w] (BASE frame)
"""

import argparse
import math
import time
from typing import List, Tuple

import numpy as np
import rclpy
from tf_transformations import quaternion_from_euler, quaternion_multiply

from scripts.rb10_controller_rbpodo_servoj import RB10Controller


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4,)
    n = float(np.linalg.norm(q))
    if n <= 0.0 or not np.isfinite(n):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def make_test_ee_pose_array(
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    hz: float = 30.0,
    duration_sec: float = 8.0,
    radius_xy_m: float = 0.01,
    z_amp_m: float = 0.005,
    yaw_amp_rad: float = 0.08,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Conservative test trajectory around current pose.
    """
    n = max(2, int(hz * duration_sec))
    dt = 1.0 / hz
    poses: List[Tuple[np.ndarray, np.ndarray]] = []
    base_quat = _normalize_quat(start_quat)
    cx, cy, cz = map(float, start_pos)

    for i in range(n):
        t = i * dt
        phase = 2.0 * math.pi * (t / duration_sec)

        x = cx + radius_xy_m * math.cos(phase)
        y = cy + radius_xy_m * math.sin(phase)
        z = cz + z_amp_m * math.sin(2.0 * phase)

        yaw = yaw_amp_rad * math.sin(phase)
        dq = quaternion_from_euler(0.0, 0.0, yaw, axes="sxyz")
        quat = _normalize_quat(quaternion_multiply(base_quat, dq))

        poses.append((np.array([x, y, z], dtype=float), quat))

    return poses


def replay_ee_pose_array(
    controller: RB10Controller,
    ee_poses: List[Tuple[np.ndarray, np.ndarray]],
    hz: float = 30.0,
    enforce_guard: bool = True,
) -> bool:
    dt = 1.0 / hz
    t0 = time.perf_counter()
    sent = 0
    ik_fail = 0
    cmd_fail = 0

    for i, (pos, quat) in enumerate(ee_poses):
        q = controller.compute_target_qpos_from_pose(pos, quat, enforce_guard=enforce_guard)
        if q is None:
            ik_fail += 1
        else:
            ok = controller.publish_qpos(q.tolist())
            if ok:
                sent += 1
            else:
                cmd_fail += 1

        next_tick = t0 + (i + 1) * dt
        sleep_sec = next_tick - time.perf_counter()
        if sleep_sec > 0.0:
            time.sleep(sleep_sec)

    controller.get_logger().info(
        f"Replay done: total={len(ee_poses)}, sent={sent}, ik_fail={ik_fail}, cmd_fail={cmd_fail}"
    )
    return (ik_fail == 0) and (cmd_fail == 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay generated EE pose array via move_servo_j")
    p.add_argument("--hz", type=float, default=30.0, help="replay rate (Hz)")
    p.add_argument("--duration", type=float, default=8.0, help="trajectory duration (sec)")
    p.add_argument("--radius", type=float, default=0.01, help="xy circle radius (m)")
    p.add_argument("--z-amp", type=float, default=0.005, help="z oscillation amplitude (m)")
    p.add_argument("--yaw-amp", type=float, default=0.08, help="yaw oscillation amplitude (rad)")
    p.add_argument("--no-guard", action="store_true", help="disable IK safety guard")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = RB10Controller()
    try:
        current = node.get_current_ee_pose()
        if current is None:
            node.get_logger().error("Cannot read current EE pose from rbpodo.")
            return

        start_pos, start_quat = current
        ee_poses = make_test_ee_pose_array(
            start_pos=start_pos,
            start_quat=start_quat,
            hz=float(args.hz),
            duration_sec=float(args.duration),
            radius_xy_m=float(args.radius),
            z_amp_m=float(args.z_amp),
            yaw_amp_rad=float(args.yaw_amp),
        )
        node.get_logger().info(
            f"Generated test EE pose array: N={len(ee_poses)}, hz={args.hz:.1f}, duration={args.duration:.2f}s"
        )
        replay_ee_pose_array(
            controller=node,
            ee_poses=ee_poses,
            hz=float(args.hz),
            enforce_guard=(not args.no_guard),
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
