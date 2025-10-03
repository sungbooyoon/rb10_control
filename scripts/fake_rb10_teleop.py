#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rb10_controller import RB10Controller
from fake_spacemouse_agent import FakeSpacemouseAgent
import time, threading, numpy as np
from tf_transformations import quaternion_from_euler, quaternion_multiply

import rclpy
def _normalize_quat(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q if n == 0 else (q / n)

class TeleopRunner:
    def __init__(self, controller, agent, rate_hz=30.0, traj_duration=0.25, enforce_guard=True,
                 max_dxyz_per_tick=0.01,   # 1 cm/tick 안전 클램프
                 max_drot_per_tick=0.2):   # 0.2 rad/tick 안전 클램프
        self.ctrl = controller
        self.agent = agent
        self.period = 1.0 / rate_hz
        self.traj_duration = float(traj_duration)
        self.enforce_guard = bool(enforce_guard)
        self.max_dxyz = float(max_dxyz_per_tick)
        self.max_drot = float(max_drot_per_tick)
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self): self._th.start()
    def stop(self):
        self._stop.set(); self._th.join(timeout=2.0)

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            pose = self.ctrl.get_current_ee_pose()
            if pose is None:
                time.sleep(self.period); continue
            pos_cur, quat_cur = pose
            quat_cur = _normalize_quat(quat_cur)

            out = self.agent.get_delta()
            if out is None:
                time.sleep(self.period); continue
            dxyz, drot, _ = out

            # per-tick clamp (추가 안전)
            if dxyz is not None:
                n = np.linalg.norm(dxyz)
                if n > self.max_dxyz > 0:
                    dxyz = dxyz * (self.max_dxyz / n)
            if drot is not None:
                m = np.linalg.norm(drot)
                if m > self.max_drot > 0:
                    drot = drot * (self.max_drot / m)

            # 합성
            pos_tgt = pos_cur + (dxyz if dxyz is not None else 0.0)
            if drot is not None:
                dq = quaternion_from_euler(*drot)
                quat_tgt = _normalize_quat(quaternion_multiply(quat_cur, dq))
            else:
                quat_tgt = quat_cur

            q = self.ctrl.compute_target_qpos_from_pose(pos_tgt, quat_tgt, enforce_guard=self.enforce_guard)
            if q is not None:
                self.ctrl.publish_qpos(q.tolist(), duration=self.traj_duration)

            dt = self.period - (time.time() - t0)
            if dt > 0: time.sleep(dt)
            
def main():
    rclpy.init()
    ctrl = RB10Controller()

    # 원 궤적 + 약간의 yaw 흔들림
    agent = FakeSpacemouseAgent(traj='circle', rate_hz=60.0,
                                trans_amp=0.003, rot_amp=0.03, freq_xy=(0.6,0.4), yaw_freq=0.3)

    runner = TeleopRunner(ctrl, agent, rate_hz=30.0, traj_duration=0.25, enforce_guard=True,
                          max_dxyz_per_tick=0.01, max_drot_per_tick=0.2)

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
