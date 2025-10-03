#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from tf_transformations import (
    quaternion_from_euler, quaternion_multiply
)

try:
    import pyspacemouse
except Exception:
    pyspacemouse = None


@dataclass
class SpacemouseConfig:
    # 스케일
    translation_scale: float = 0.06   # m per full deflection
    angle_scale: float = 0.24         # rad per full deflection
    # 데드존/감쇠
    deadzone: float = 0.10
    snap_zero_below: float = 0.60     # 0.60 미만은 0으로 스냅(노이즈 컷)
    # 부호 반전 (x,y,z, roll,pitch,yaw)
    invert_control: np.ndarray = np.ones(6, dtype=float)
    # 회전 적용 순서: "post" = q_new = q_cur * dq, "pre" = q_new = dq * q_cur
    rotation_apply: str = "post"
    # 입력 좌표계를 BASE로 맵핑하는 3x3 회전행렬 (필요시 사용자 세팅)
    R_spacemouse_to_base: np.ndarray = np.eye(3, dtype=float)


class SpacemouseAgent:
    """
    - 스페이스마우스를 읽어 Δ를 산출 (BASE_LINK frame 기준)
    - 필요시 현재 EE pose를 외부 콜백으로 받아 new pose 합성도 제공
    - IK/ROS/로봇 상태 의존성 없음 (입력 매핑 전용)
    """
    def __init__(
        self,
        config: SpacemouseConfig = SpacemouseConfig(),
        device_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.cfg = config
        self._device_path = device_path
        self._verbose = verbose

        self._lock = threading.Lock()
        self._latest_state = None  # pyspacemouse state

        if pyspacemouse is None:
            raise RuntimeError("pyspacemouse not available. `pip install pyspacemouse` 필요")

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
            dxyz_base: (3,) in meters
            drot_rpy: (3,) in radians or None
            buttons:  [left, right] (원본 값)
        """
        with self._lock:
            st = self._latest_state
        if st is None:
            return None

        # raw axes
        raw = np.array([st.x, st.y, st.z, st.roll, st.pitch, st.yaw], dtype=float)

        # invert & deadzone
        v = raw * self.cfg.invert_control
        if np.max(np.abs(v)) > 0.9:
            # 큰 동작 중 잡음 억제: 0.6 미만 축은 0으로 스냅
            v[np.abs(v) < self.cfg.snap_zero_below] = 0.0
        # 일반 deadzone
        v[np.abs(v) < self.cfg.deadzone] = 0.0

        tx, ty, tz, r, p, y = v

        # 입력축 → BASE frame 맵핑
        R = self.cfg.R_spacemouse_to_base  # (3x3)
        dxyz_base = R @ (np.array([tx, ty, tz], dtype=float) * self.cfg.translation_scale)

        # 소회전 (roll, pitch, yaw) 라디안
        drot_rpy = np.array([r, p, y], dtype=float) * self.cfg.angle_scale
        buttons = st.buttons  # [left, right]
        return dxyz_base, drot_rpy, buttons