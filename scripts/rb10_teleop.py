#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from spacemouse_agent import SpacemouseAgent, SpacemouseConfig
from scripts.rb10_controller_rbpodo import RB10Controller
import time, threading, numpy as np
from tf_transformations import quaternion_from_euler, quaternion_multiply, quaternion_matrix
import rclpy
from geometry_msgs.msg import TwistStamped  # 추가

MAX_V_TRANSL = 0.5   # m/s   (0.20~0.30 권장)
MAX_W_ROT    = 1.2    # rad/s (0.5~1.0 권장)

def _normalize_quat(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q if n == 0 else (q / n)

class TeleopRunner:
    def __init__(self, controller, agent,
                 rate_hz=30.0, traj_duration=0.25, enforce_guard=True,
                 max_dxyz_per_tick=None,      # 1 cm/tick
                 max_drot_per_axis=None,        # 0.2 rad/axis/tick
                 drot_deadband=1e-3,           # 너무 작은 회전 무시
                 delta_in='tool',              # 'tool' or 'base'
                 verbose=False):
        self.ctrl = controller
        self.agent = agent
        self.period = 1.0 / float(rate_hz)
        self.traj_duration = float(traj_duration)
        self.enforce_guard = bool(enforce_guard)
        self.max_dxyz = (MAX_V_TRANSL * self.period) if max_dxyz_per_tick is None else float(max_dxyz_per_tick)
        self.max_drot = (MAX_W_ROT * self.period) if max_drot_per_axis  is None else float(max_drot_per_axis)
        self.deadband = float(drot_deadband)
        self.delta_in = delta_in  # 입력 델타의 기준 프레임
        self.verbose = verbose

        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

        self.pub_delta = controller.create_publisher(TwistStamped, "/rb/teleop_delta", 10)

    def start(self): self._th.start()
    def stop(self):
        self._stop.set()
        self._th.join(timeout=2.0)

    def _loop(self):
        node = self.ctrl
        rate = node.create_rate(1.0 / self.period)
        while not self._stop.is_set():
            t0 = time.time()

            pose = self.ctrl.get_current_ee_pose()
            if pose is None:
                time.sleep(self.period); continue
            pos_cur, quat_cur = pose
            quat_cur = _normalize_quat(quat_cur)
            R_cur = quaternion_matrix(quat_cur)[:3, :3]

            out = self.agent.get_delta()
            if out is None:
                # 퍼블리시: 0 델타 (타임라인 유지)
                msg = TwistStamped()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.header.frame_id = self.delta_in  # 'base' or 'tool'
                msg.twist.linear.x = 0.0
                msg.twist.linear.y = 0.0
                msg.twist.linear.z = 0.0
                msg.twist.angular.x = 0.0
                msg.twist.angular.y = 0.0
                msg.twist.angular.z = 0.0
                self.pub_delta.publish(msg)
                time.sleep(self.period)
                continue

            dxyz, drot, _ = out

            # --- 안전 클램프/데드밴드 (기존 로직) ---
            if dxyz is not None:
                n = float(np.linalg.norm(dxyz))
                if self.max_dxyz > 0 and n > self.max_dxyz:
                    dxyz = dxyz * (self.max_dxyz / n)
            if drot is not None:
                drot = np.asarray(drot, dtype=float)
                if np.linalg.norm(drot) < self.deadband:
                    drot = None
                else:
                    drot = np.clip(drot, -self.max_drot, self.max_drot)

            # --- 퍼블리시용 델타 준비 (없으면 0으로) ---
            dxyz_pub = np.zeros(3, dtype=float) if dxyz is None else np.asarray(dxyz, dtype=float)
            drot_pub = np.zeros(3, dtype=float) if drot is None else np.asarray(drot, dtype=float)

            # ✨ 퍼블리시: 현재 tick의 최종 델타 (클램프/데드밴드 반영)
            msg = TwistStamped()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = self.delta_in  # 'base' or 'tool' (기준 프레임 명시)
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = dxyz_pub.tolist()
            msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = drot_pub.tolist()
            self.pub_delta.publish(msg)

            # --- 타깃 포즈 합성 (BASE 프레임으로 통일) ---
            if dxyz is not None:
                if self.delta_in == 'tool':
                    pos_tgt = pos_cur + R_cur @ dxyz
                else:
                    pos_tgt = pos_cur + dxyz
            else:
                pos_tgt = pos_cur

            if drot is not None:
                dq = quaternion_from_euler(drot[0], drot[1], drot[2], axes='rxyz')
                dq = _normalize_quat(dq)
                if self.delta_in == 'tool':
                    quat_tgt = _normalize_quat(quaternion_multiply(quat_cur, dq))
                else:
                    quat_tgt = _normalize_quat(quaternion_multiply(dq, quat_cur))
            else:
                quat_tgt = quat_cur

            q = self.ctrl.compute_target_qpos_from_pose(pos_tgt, quat_tgt, enforce_guard=self.enforce_guard)
            if q is not None:
                self.ctrl.publish_qpos(q.tolist(), duration=self.traj_duration)
                if self.verbose:
                    self.ctrl.get_logger().info("teleop tick ok")

            rate.sleep()

def main():
    rclpy.init()
    ctrl = RB10Controller()
    agent = SpacemouseAgent(
        SpacemouseConfig(
            translation_scale=0.6,
            angle_scale=1.2,
        ),
        device_path=None,
        verbose=False
    )
    runner = TeleopRunner(ctrl, agent,
                          rate_hz=30.0,
                          traj_duration=0.08, # 보통 traj_duration ≈ 2 × period ~ 3 × period 가 안정적입니다.
                          enforce_guard=True,
                          delta_in='base',      # 스페이스마우스가 EE 로컬 델타일 때
                          verbose=False)

    try:
        runner.start()
        while rclpy.ok():
            rclpy.spin_once(ctrl, timeout_sec=0.1)
    finally:
        runner.stop()
        agent.close()
        ctrl.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
