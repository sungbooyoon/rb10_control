#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:
    import pyspacemouse
except Exception:
    pyspacemouse = None


@dataclass
class SpacemouseConfig:
    # 스케일
    translation_scale: float = 0.06   # m per full deflection
    angle_scale: float = 0.24         # rad per full deflection (소회전 크기)
    # 데드존/감쇠
    deadzone: float = 0.10
    snap_zero_below: float = 0.60     # 0.60 미만은 0으로 스냅(노이즈 컷)
    # 부호 반전 (x,y,z, roll,pitch,yaw)
    invert_control: np.ndarray = field(default_factory=lambda: np.ones(6, dtype=float))
    # 입력 좌표계를 BASE로 맵핑하는 3x3 회전행렬 (device/tool → base)
    R_spacemouse_to_base: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))
    # 회전 벡터의 출력 좌표계: 'tool' (EE 로컬) or 'base' (월드/베이스)
    rot_frame: str = "tool"  # TeleopRunner(delta_in)와 일치시킬 것
    # 버튼 그대로 반환 여부
    return_buttons: bool = True


class SpacemouseAgent:
    """
    - 스페이스마우스를 읽어 Δ를 산출
      dxyz_base: (3,) in meters — 항상 BASE frame
      drot_vec:  (3,) in radians — cfg.rot_frame 기준의 소회전 벡터 (rx, ry, rz)
                 (소회전이므로 프레임 변환은 선형: ω_base = R * ω_tool)
      buttons:   [left, right] (원본 값) 또는 []

    - IK/ROS/로봇 상태 의존성 없음 (입력 매핑 전용)
    """
    def __init__(
        self,
        config: SpacemouseConfig = SpacemouseConfig(),
        device_path: Optional[str] = None,
        verbose: bool = False,
    ):
        if pyspacemouse is None:
            raise RuntimeError("pyspacemouse not available. `pip install pyspacemouse` 필요")

        self.cfg = config
        self._device_path = device_path
        self._verbose = verbose

        self._lock = threading.Lock()
        self._latest_state = None  # pyspacemouse state

        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    # ---------- 내부: 리더 스레드 ----------
    def _reader_loop(self):
        mouse = pyspacemouse.open(self._device_path) if self._device_path else pyspacemouse.open()
        if not mouse:
            raise RuntimeError("Failed to open spacemouse device")
        if self._verbose:
            print("[SpacemouseAgent] device opened")

        while not self._stop_evt.is_set():
            st = mouse.read()
            with self._lock:
                self._latest_state = st
            time.sleep(0.001)

    def close(self):
        self._stop_evt.set()
        self._thread.join(timeout=1.0)

    def get_delta(self) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], list]]:
        """
        Returns:
            dxyz_base: (3,)  [meters]  — BASE frame
            drot_vec:  (3,)  [radians] — cfg.rot_frame('tool' or 'base') frame의 소회전 벡터 (rx, ry, rz)
            buttons:   list  — [left, right] (원본) 또는 []
        """
        with self._lock:
            st = self._latest_state
        if st is None:
            return None

        # raw axes
        raw = np.array([st.x, st.y, st.z, st.roll, st.pitch, st.yaw], dtype=float)

        # invert & deadzone
        v = raw * self.cfg.invert_control
        # 큰 동작 중 잡음 억제: 0.6 미만 축은 0으로 스냅
        if np.max(np.abs(v)) > 0.9:
            v[np.abs(v) < self.cfg.snap_zero_below] = 0.0
        # 일반 deadzone
        v[np.abs(v) < self.cfg.deadzone] = 0.0

        tx, ty, tz, r, p, y = v

        # --- translation: device/tool → BASE
        R = self.cfg.R_spacemouse_to_base  # (3x3)
        dxyz_tool = np.array([tx, ty, tz], dtype=float) * self.cfg.translation_scale
        dxyz_base = R @ dxyz_tool  # 항상 BASE로 반환

        # --- rotation (소회전 벡터): 기본은 tool 프레임, 필요시 base로 변환
        omega_tool = np.array([r, p, y], dtype=float) * self.cfg.angle_scale  # (rx, ry, rz)
        if np.allclose(omega_tool, 0.0, atol=1e-12):
            drot_vec = None
        else:
            if self.cfg.rot_frame == "base":
                drot_vec = R @ omega_tool  # ω_base = R * ω_tool
            else:
                drot_vec = omega_tool      # tool 프레임 그대로

        buttons = st.buttons if self.cfg.return_buttons else []
        return dxyz_base, drot_vec, buttons
