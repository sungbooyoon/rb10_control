#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-space admittance + IK + joint servo-j controller.

Block diagram style:
  x_r (xyz+quat) -> Admittance -> x_d -> IK -> q_d -> move_servo_j

This script is intentionally separated from existing controllers.
"""

import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from tf_transformations import quaternion_inverse, quaternion_multiply, quaternion_from_euler

from scripts.rbpodo_controller_servoj import RB10Controller


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4,)
    n = float(np.linalg.norm(q))
    if n <= 0.0 or not np.isfinite(n):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    qn = _normalize_quat(q)
    if qn[3] < 0.0:
        qn = -qn
    w = float(np.clip(qn[3], -1.0, 1.0))
    v = qn[:3]
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(nv, w)
    axis = v / nv
    return axis * angle


class RBPodoAdmittanceJointController(RB10Controller):
    def __init__(self) -> None:
        super().__init__()

        # Loop / limits
        self.declare_parameter("control_hz", 30.0)
        self.declare_parameter("max_linear_vel", 0.08)    # m/s
        self.declare_parameter("max_angular_vel", 0.8)    # rad/s

        # Admittance virtual model (translation)
        self.declare_parameter("m_trans", [4.0, 4.0, 4.0])
        self.declare_parameter("d_trans", [80.0, 80.0, 80.0])
        self.declare_parameter("k_trans", [120.0, 120.0, 120.0])

        # Admittance virtual model (rotation)
        self.declare_parameter("m_rot", [0.25, 0.25, 0.25])
        self.declare_parameter("d_rot", [8.0, 8.0, 8.0])
        self.declare_parameter("k_rot", [20.0, 20.0, 20.0])

        # Wrench preprocessing
        self.declare_parameter("deadband_force", [2.0, 2.0, 2.0])   # N
        self.declare_parameter("deadband_torque", [0.2, 0.2, 0.2])  # Nm
        self.declare_parameter("wrench_lpf_alpha", 0.2)
        self.declare_parameter("auto_bias_samples", 100)

        self._hz = float(self.get_parameter("control_hz").value)
        self._dt = 1.0 / self._hz
        self._max_v = float(self.get_parameter("max_linear_vel").value)
        self._max_w = float(self.get_parameter("max_angular_vel").value)

        self._m_t = np.asarray(self.get_parameter("m_trans").value, dtype=float)
        self._d_t = np.asarray(self.get_parameter("d_trans").value, dtype=float)
        self._k_t = np.asarray(self.get_parameter("k_trans").value, dtype=float)
        self._m_r = np.asarray(self.get_parameter("m_rot").value, dtype=float)
        self._d_r = np.asarray(self.get_parameter("d_rot").value, dtype=float)
        self._k_r = np.asarray(self.get_parameter("k_rot").value, dtype=float)

        self._db_f = np.asarray(self.get_parameter("deadband_force").value, dtype=float)
        self._db_m = np.asarray(self.get_parameter("deadband_torque").value, dtype=float)
        self._w_alpha = float(self.get_parameter("wrench_lpf_alpha").value)
        self._auto_bias_samples = int(self.get_parameter("auto_bias_samples").value)

        # Desired (admittance output) states
        pose = self.get_current_ee_pose()
        if pose is None:
            raise RuntimeError("Cannot initialize admittance state: no current EE pose.")
        pos0, quat0 = pose
        self._x_d = np.asarray(pos0, dtype=float).copy()
        self._q_d = _normalize_quat(np.asarray(quat0, dtype=float))
        self._v_d = np.zeros(3, dtype=float)
        self._w_d = np.zeros(3, dtype=float)

        # Reference pose (input trajectory)
        self._x_r = self._x_d.copy()
        self._q_r = self._q_d.copy()

        # Wrench filtering / bias
        self._wrench_bias = np.zeros(6, dtype=float)
        self._wrench_lpf = np.zeros(6, dtype=float)
        self._calibrate_bias(self._auto_bias_samples)

        self.create_subscription(PoseStamped, "~/target_pose", self._target_pose_cb, 10)
        self.create_service(Trigger, "~/zero_bias", self._zero_bias_cb)
        self.create_service(Trigger, "~/hold_current", self._hold_current_cb)
        self._timer = self.create_timer(self._dt, self._on_timer)
        self.get_logger().info(f"Admittance+IK controller started (hz={self._hz:.1f})")

    @staticmethod
    def _extract_wrench(s) -> Optional[np.ndarray]:
        fx = getattr(s, "eft_fx", None)
        fy = getattr(s, "eft_fy", None)
        fz = getattr(s, "eft_fz", None)
        mx = getattr(s, "eft_mx", None)
        my = getattr(s, "eft_my", None)
        mz = getattr(s, "eft_mz", None)
        if None in (fx, fy, fz, mx, my, mz):
            return None
        w = np.asarray([fx, fy, fz, mx, my, mz], dtype=float)
        if not np.all(np.isfinite(w)):
            return None
        return w

    @staticmethod
    def _apply_deadband(x: np.ndarray, db: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[np.abs(y) < db] = 0.0
        return y

    def _target_pose_cb(self, msg: PoseStamped) -> None:
        self._x_r = np.array(
            [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ],
            dtype=float,
        )
        self._q_r = _normalize_quat(
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

    def _calibrate_bias(self, samples: int) -> None:
        acc = np.zeros(6, dtype=float)
        n = 0
        for _ in range(max(0, samples)):
            s = self._request_rbpodo_sdata()
            if s is None:
                continue
            w = self._extract_wrench(s)
            if w is None:
                continue
            acc += w
            n += 1
        if n > 0:
            self._wrench_bias = acc / float(n)
            self.get_logger().info(f"Wrench bias calibrated with {n} samples.")
        else:
            self.get_logger().warn("Wrench bias calibration skipped (no valid samples).")

    def _zero_bias_cb(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        del req
        self._calibrate_bias(max(20, self._auto_bias_samples))
        res.success = True
        res.message = "Wrench bias recalibrated."
        return res

    def _hold_current_cb(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        del req
        pose = self.get_current_ee_pose()
        if pose is None:
            res.success = False
            res.message = "No current pose."
            return res
        p, q = pose
        self._x_r = np.asarray(p, dtype=float)
        self._q_r = _normalize_quat(np.asarray(q, dtype=float))
        self._x_d = self._x_r.copy()
        self._q_d = self._q_r.copy()
        self._v_d[:] = 0.0
        self._w_d[:] = 0.0
        res.success = True
        res.message = "Reference and desired states reset to current pose."
        return res

    def _get_filtered_wrench(self) -> Optional[np.ndarray]:
        s = self._request_rbpodo_sdata()
        if s is None:
            return None
        w = self._extract_wrench(s)
        if w is None:
            return None
        w = w - self._wrench_bias
        a = float(np.clip(self._w_alpha, 0.0, 1.0))
        self._wrench_lpf = (1.0 - a) * self._wrench_lpf + a * w
        return self._wrench_lpf

    def _on_timer(self) -> None:
        wrench = self._get_filtered_wrench()
        if wrench is None:
            return

        f_ext = self._apply_deadband(wrench[:3], self._db_f)
        m_ext = self._apply_deadband(wrench[3:], self._db_m)

        # Translation admittance:
        # M * x_ddot + D * x_dot + K * (x_d - x_r) = F_ext
        a_t = (f_ext - self._d_t * self._v_d - self._k_t * (self._x_d - self._x_r)) / np.maximum(self._m_t, 1e-6)
        self._v_d += a_t * self._dt
        self._v_d = np.clip(self._v_d, -self._max_v, self._max_v)
        self._x_d += self._v_d * self._dt

        # Rotation admittance on orientation error vector:
        q_err = quaternion_multiply(self._q_r, quaternion_inverse(self._q_d))
        e_r = _quat_to_rotvec(np.asarray(q_err, dtype=float))
        a_r = (m_ext - self._d_r * self._w_d + self._k_r * e_r) / np.maximum(self._m_r, 1e-6)
        self._w_d += a_r * self._dt
        self._w_d = np.clip(self._w_d, -self._max_w, self._max_w)

        dq = quaternion_from_euler(
            self._w_d[0] * self._dt,
            self._w_d[1] * self._dt,
            self._w_d[2] * self._dt,
            axes="sxyz",
        )
        self._q_d = _normalize_quat(np.asarray(quaternion_multiply(self._q_d, dq), dtype=float))

        q_cmd = self.compute_target_qpos_from_pose(self._x_d, self._q_d, enforce_guard=True)
        if q_cmd is None:
            return
        self.publish_joint_trajectory([q_cmd.tolist()], [0.3])


def main() -> None:
    rclpy.init()
    node = RBPodoAdmittanceJointController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
