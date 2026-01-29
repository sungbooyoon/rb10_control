# fake_spacemouse_agent.py
import time, threading, numpy as np

class FakeSpacemouseAgent:
    """
    SpacemouseAgent와 동일 인터페이스: get_delta() -> (dxyz_base, drot_rpy, buttons)
    - traj: 'circle', 'line', 'still' 선택
    """
    def __init__(self, traj='circle', rate_hz=60.0,
                 trans_amp=0.003, rot_amp=0.03,  # per-tick 크기
                 freq_xy=(0.6, 0.4), yaw_freq=0.3, verbose=False):
        self.traj = traj
        self.period = 1.0 / float(rate_hz)
        self.trans_amp = float(trans_amp)
        self.rot_amp = float(rot_amp)
        self.fx, self.fy = freq_xy
        self.fyaw = yaw_freq
        self.verbose = verbose

        self._lock = threading.Lock()
        self._delta = (np.zeros(3, dtype=float), np.zeros(3, dtype=float), [0,0])
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        t0 = time.time()
        while not self._stop.is_set():
            t = time.time() - t0
            if self.traj == 'circle':
                dx = self.trans_amp * np.sin(2*np.pi*self.fx*t)
                dy = self.trans_amp * np.cos(2*np.pi*self.fy*t)
                dz = 0.0
                droll = 0.0; dpitch = 0.0
                dyaw  = self.rot_amp * np.sin(2*np.pi*self.fyaw*t)
            elif self.traj == 'line':
                dx = self.trans_amp; dy = 0.0; dz = 0.0
                droll = dpitch = dyaw = 0.0
            else:  # 'still'
                dx = dy = dz = 0.0
                droll = dpitch = dyaw = 0.0

            with self._lock:
                self._delta = (np.array([dx,dy,dz], dtype=float),
                               np.array([droll, dpitch, dyaw], dtype=float),
                               [0,0])
            time.sleep(self.period)

    def get_delta(self):
        with self._lock:
            dxyz, drot, btn = self._delta
        return dxyz.copy(), drot.copy(), btn.copy()

    def close(self):
        self._stop.set()
        self._th.join(timeout=1.0)