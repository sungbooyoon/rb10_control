#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Topic-driven wrapper for RB10Controller (servo-j path).

Input:
- ~/target_pose (geometry_msgs/PoseStamped), BASE frame xyz + quaternion

Behavior:
- Run control loop at fixed rate
- IK (target pose -> qpos)
- publish_qpos() -> move_servo_j
"""

from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger

from scripts.rbpodo_controller_servoj import RB10Controller


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4,)
    n = float(np.linalg.norm(q))
    if n <= 0.0 or not np.isfinite(n):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


class RB10ServoJPoseTopicController(RB10Controller):
    def __init__(self, control_hz: float = 30.0, enforce_guard: bool = True) -> None:
        super().__init__()

        self.declare_parameter("control_hz", float(control_hz))
        self.declare_parameter("enforce_guard", bool(enforce_guard))

        self._hz = float(self.get_parameter("control_hz").value)
        self._enforce_guard = bool(self.get_parameter("enforce_guard").value)

        self._target_pos: Optional[np.ndarray] = None
        self._target_quat: Optional[np.ndarray] = None
        self._recv_count = 0
        self._ctrl_count = 0

        self.create_subscription(PoseStamped, "~/target_pose", self._target_pose_cb, 10)
        self.create_service(Trigger, "~/hold_current", self._hold_current_cb)
        self._timer = self.create_timer(1.0 / self._hz, self._on_timer)
        self.get_logger().info(f"Pose-topic servo-j controller started (hz={self._hz:.1f})")

    def get_loop_stats(self) -> dict:
        return {
            "control_hz": float(self._hz),
            "recv_count": int(self._recv_count),
            "ctrl_count": int(self._ctrl_count),
            "has_target": bool(self._target_pos is not None and self._target_quat is not None),
        }

    def _target_pose_cb(self, msg: PoseStamped) -> None:
        self._target_pos = np.array(
            [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ],
            dtype=float,
        )
        self._target_quat = _normalize_quat(
            np.array(
                [
                    float(msg.pose.orientation.x),
                    float(msg.pose.orientation.y),
                    float(msg.pose.orientation.z),
                    float(msg.pose.orientation.w),
                ],
                dtype=float,
            )
        )
        self._recv_count += 1

    def _hold_current_cb(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        del req
        pose = self.get_current_ee_pose()
        if pose is None:
            res.success = False
            res.message = "No current EE pose from rbpodo."
            return res
        pos, quat = pose
        self._target_pos = np.asarray(pos, dtype=float)
        self._target_quat = _normalize_quat(np.asarray(quat, dtype=float))
        res.success = True
        res.message = "Target set to current EE pose."
        return res

    def _on_timer(self) -> None:
        if self._target_pos is None or self._target_quat is None:
            return
        q = self.compute_target_qpos_from_pose(
            self._target_pos, self._target_quat, enforce_guard=self._enforce_guard
        )
        if q is None:
            return
        self.publish_qpos(q.tolist())
        self._ctrl_count += 1

        if (self._ctrl_count % int(max(1.0, self._hz))) == 0:
            self.get_logger().info(
                f"loop={self._ctrl_count}, recv={self._recv_count}, target_pos={self._target_pos.round(4).tolist()}"
            )


def main() -> None:
    rclpy.init()
    node = RB10ServoJPoseTopicController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
