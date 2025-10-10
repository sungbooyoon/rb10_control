#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from spacemouse_agent import SpacemouseAgent, SpacemouseConfig
from rb10_controller import RB10Controller
import time, threading, numpy as np
from tf_transformations import quaternion_from_euler, quaternion_multiply, quaternion_matrix
import rclpy


MAX_V_TRANSL = 0.25   # m/s   (0.20~0.30 권장)
MAX_W_ROT    = 0.8    # rad/s (0.5~1.0 권장)

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

    def start(self): self._th.start()
    def stop(self):
        self._stop.set()
        self._th.join(timeout=2.0)

    def _loop(self):
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
                time.sleep(self.period); continue
            dxyz, drot, _ = out

            # --- 안전 클램프/데드밴드 ---
            if dxyz is not None:
                n = float(np.linalg.norm(dxyz))
                if self.max_dxyz > 0 and n > self.max_dxyz:
                    dxyz = dxyz * (self.max_dxyz / n)
            if drot is not None:
                # per-axis clip
                drot = np.asarray(drot, dtype=float)
                if np.linalg.norm(drot) < self.deadband:
                    drot = None
                else:
                    drot = np.clip(drot, -self.max_drot, self.max_drot)

            # --- 타깃 포즈 합성 (BASE 프레임으로 통일) ---
            if dxyz is not None:
                if self.delta_in == 'tool':
                    # EE 로컬 이동 → BASE로 변환
                    pos_tgt = pos_cur + R_cur @ dxyz
                else:
                    # BASE 기준 이동
                    pos_tgt = pos_cur + dxyz
            else:
                pos_tgt = pos_cur

            if drot is not None:
                # EE 회전 델타(바디 고정) → intrinsic rxyz 사용
                dq = quaternion_from_euler(drot[0], drot[1], drot[2], axes='rxyz')
                dq = _normalize_quat(dq)
                if self.delta_in == 'tool':
                    # 바디 고정 회전: q_new = q_cur ⊗ dq
                    quat_tgt = _normalize_quat(quaternion_multiply(quat_cur, dq))
                else:
                    # 월드 고정 회전: q_new = dq ⊗ q_cur
                    quat_tgt = _normalize_quat(quaternion_multiply(dq, quat_cur))
            else:
                quat_tgt = quat_cur

            # --- IK 호출: BASE 기준 절대 포즈 전달 ---
            q = self.ctrl.compute_target_qpos_from_pose(pos_tgt, quat_tgt, enforce_guard=self.enforce_guard)
            if q is not None:
                self.ctrl.publish_qpos(q.tolist(), duration=self.traj_duration)
                if self.verbose:
                    # 간단 로그 (원하면 꺼도 됨)
                    self.ctrl.get_logger().info(f"teleop tick ok")

            # 주기 유지
            dt = self.period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

def main():
    rclpy.init()
    ctrl = RB10Controller()
    agent = SpacemouseAgent(
        SpacemouseConfig(
            translation_scale=0.12,
            angle_scale=0.24,
        ),
        device_path=None,
        verbose=False
    )
    runner = TeleopRunner(ctrl, agent,
                          rate_hz=30.0,
                          traj_duration=0.08, # 보통 traj_duration ≈ 2 × period ~ 3 × period 가 안정적입니다.
                          enforce_guard=True,
                          delta_in='tool',      # 스페이스마우스가 EE 로컬 델타일 때
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
