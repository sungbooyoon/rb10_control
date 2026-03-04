#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone RB Admittance Controller (does not modify existing controllers).

- Input: rbpodo SystemState.eft (Fx, Fy, Fz, Mx, My, Mz)
- Output: rbpodo move_servo_l with Cartesian velocity-style command
          [vx, vy, vz, wx, wy, wz] in [m/s, rad/s]
"""

import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

try:
    import rbpodo as rb
except Exception:
    rb = None


class RBPodoAdmittanceController(Node):
    def __init__(self) -> None:
        super().__init__("rbpodo_admittance_controller")

        if rb is None:
            raise RuntimeError("rbpodo import failed. Install rbpodo first.")

        # Connection / loop
        self.declare_parameter("robot_ip", "10.0.2.7")
        self.declare_parameter("simulation", False)
        self.declare_parameter("speed_bar", 0.2)
        self.declare_parameter("control_hz", 100.0)
        self.declare_parameter("data_timeout_sec", 0.02)

        # Admittance gains
        self.declare_parameter("k_force_xyz", [0.0010, 0.0010, 0.0008])   # [m/s]/N
        self.declare_parameter("k_torque_rpy", [0.0200, 0.0200, 0.0200])  # [rad/s]/Nm

        # Deadband
        self.declare_parameter("deadband_force_xyz", [2.0, 2.0, 2.0])      # N
        self.declare_parameter("deadband_torque_rpy", [0.2, 0.2, 0.2])     # Nm

        # Limits
        self.declare_parameter("max_vel_xyz", [0.05, 0.05, 0.04])           # m/s
        self.declare_parameter("max_vel_rpy", [0.50, 0.50, 0.50])           # rad/s

        # Low-pass filtering on velocity command
        self.declare_parameter("vel_lpf_alpha", 0.25)  # 0~1, larger=faster response

        # rbpodo servo-l params (same semantic group used in rbpodo_ros2)
        self.declare_parameter("servo_t1", 0.01)
        self.declare_parameter("servo_t2", 0.10)
        self.declare_parameter("servo_gain", 0.5)
        self.declare_parameter("servo_alpha", 0.5)

        # Bias calibration
        self.declare_parameter("auto_bias_samples", 100)

        self._robot_ip = str(self.get_parameter("robot_ip").value)
        self._simulation = bool(self.get_parameter("simulation").value)
        self._speed_bar = float(self.get_parameter("speed_bar").value)
        self._hz = float(self.get_parameter("control_hz").value)
        self._data_timeout = float(self.get_parameter("data_timeout_sec").value)

        self._k_force = np.asarray(self.get_parameter("k_force_xyz").value, dtype=float)
        self._k_torque = np.asarray(self.get_parameter("k_torque_rpy").value, dtype=float)
        self._deadband_f = np.asarray(self.get_parameter("deadband_force_xyz").value, dtype=float)
        self._deadband_m = np.asarray(self.get_parameter("deadband_torque_rpy").value, dtype=float)
        self._max_v = np.asarray(self.get_parameter("max_vel_xyz").value, dtype=float)
        self._max_w = np.asarray(self.get_parameter("max_vel_rpy").value, dtype=float)
        self._vel_alpha = float(self.get_parameter("vel_lpf_alpha").value)

        self._t1 = float(self.get_parameter("servo_t1").value)
        self._t2 = float(self.get_parameter("servo_t2").value)
        self._gain = float(self.get_parameter("servo_gain").value)
        self._alpha = float(self.get_parameter("servo_alpha").value)

        self._auto_bias_samples = int(self.get_parameter("auto_bias_samples").value)

        self._robot = rb.Cobot(self._robot_ip)
        self._rc = rb.ResponseCollector()
        self._data = rb.CobotData(self._robot_ip)
        if self._simulation:
            self._robot.set_operation_mode(self._rc, rb.OperationMode.Simulation)
        self._robot.set_speed_bar(self._rc, self._speed_bar)
        self._robot.flush(self._rc)

        self._wrench_bias = np.zeros(6, dtype=float)
        self._last_cmd = np.zeros(6, dtype=float)
        self._last_wrench: Optional[np.ndarray] = None
        self._tick = 0

        self._calibrate_bias(self._auto_bias_samples)

        self._srv_zero_bias = self.create_service(Trigger, "~/zero_bias", self._zero_bias_cb)
        self._timer = self.create_timer(1.0 / self._hz, self._on_timer)
        self.get_logger().info(
            f"Admittance node started (ip={self._robot_ip}, hz={self._hz:.1f}, sim={self._simulation})"
        )

    def _read_sdata(self):
        try:
            data = self._data.request_data(self._data_timeout)
            if data is None or getattr(data, "sdata", None) is None:
                return None
            return data.sdata
        except Exception as e:
            self.get_logger().warn(f"request_data failed: {e}")
            return None

    @staticmethod
    def _extract_wrench(s) -> Optional[np.ndarray]:
        # rbpodo exposes wrench as scalar fields in SystemData:
        # eft_fx, eft_fy, eft_fz, eft_mx, eft_my, eft_mz
        fx = getattr(s, "eft_fx", None)
        fy = getattr(s, "eft_fy", None)
        fz = getattr(s, "eft_fz", None)
        mx = getattr(s, "eft_mx", None)
        my = getattr(s, "eft_my", None)
        mz = getattr(s, "eft_mz", None)
        if None not in (fx, fy, fz, mx, my, mz):
            w = np.asarray([fx, fy, fz, mx, my, mz], dtype=float)
            if np.all(np.isfinite(w)):
                return w

        # Fallback for any wrapper that exposes packed eft[6]
        eft = getattr(s, "eft", None)
        if eft is not None:
            try:
                w = np.asarray([float(eft[i]) for i in range(6)], dtype=float)
                if np.all(np.isfinite(w)):
                    return w
            except Exception:
                pass
        return None

    def _calibrate_bias(self, samples: int) -> None:
        if samples <= 0:
            return
        acc = np.zeros(6, dtype=float)
        n = 0
        for _ in range(samples):
            s = self._read_sdata()
            if s is None:
                continue
            w = self._extract_wrench(s)
            if w is not None:
                acc += w
                n += 1
        if n > 0:
            self._wrench_bias = acc / float(n)
            self.get_logger().info(
                "Bias calibrated: "
                f"F=({self._wrench_bias[0]:.3f},{self._wrench_bias[1]:.3f},{self._wrench_bias[2]:.3f}) "
                f"M=({self._wrench_bias[3]:.3f},{self._wrench_bias[4]:.3f},{self._wrench_bias[5]:.3f})"
            )
        else:
            self.get_logger().warn("Bias calibration failed (no valid samples). Using zero bias.")

    def _zero_bias_cb(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        del req
        self._calibrate_bias(max(20, self._auto_bias_samples))
        res.success = True
        res.message = "Bias recalibrated."
        return res

    @staticmethod
    def _apply_deadband(x: np.ndarray, db: np.ndarray) -> np.ndarray:
        y = x.copy()
        mask = np.abs(y) < db
        y[mask] = 0.0
        return y

    def _on_timer(self) -> None:
        s = self._read_sdata()
        if s is None:
            return

        wrench = self._extract_wrench(s)
        if wrench is None:
            return
        self._last_wrench = wrench

        w = wrench - self._wrench_bias
        f = self._apply_deadband(w[:3], self._deadband_f)
        m = self._apply_deadband(w[3:], self._deadband_m)

        v = self._k_force * f
        wv = self._k_torque * m

        v = np.clip(v, -self._max_v, self._max_v)
        wv = np.clip(wv, -self._max_w, self._max_w)
        cmd = np.concatenate([v, wv], axis=0)

        # first-order low-pass on command to reduce chatter
        a = float(np.clip(self._vel_alpha, 0.0, 1.0))
        cmd = (1.0 - a) * self._last_cmd + a * cmd
        self._last_cmd = cmd

        try:
            cmd_rb = np.array(
                [
                    cmd[0] * 1000.0,  # m/s -> mm/s
                    cmd[1] * 1000.0,
                    cmd[2] * 1000.0,
                    math.degrees(cmd[3]),  # rad/s -> deg/s
                    math.degrees(cmd[4]),
                    math.degrees(cmd[5]),
                ],
                dtype=float,
            )

            t1 = max(0.002, float(self._t1))
            t2 = float(np.clip(self._t2, 0.021, 0.199))
            gain = max(1e-6, float(self._gain))
            alpha = float(np.clip(self._alpha, 0.001, 0.999))

            self._robot.move_servo_l(self._rc, cmd_rb, t1, t2, gain, alpha)
        except Exception as e:
            self.get_logger().warn(f"move_servo_l failed: {e}")
            return

        self._tick += 1
        if (self._tick % int(max(1.0, self._hz))) == 0:
            self.get_logger().info(
                f"wrench_raw={wrench.round(3).tolist()} cmd_vel={cmd.round(4).tolist()}"
            )


def main() -> None:
    rclpy.init()
    node = RBPodoAdmittanceController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
