#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RB10 Demo Playback (robomimic HDF5, ee_pos+ee_quat)
- Loads a robomimic-compatible HDF5
- Selects a trajectory group (demo_0, demo_1, ...)
- Each timestep:
    - Read ee_pos (x,y,z) and ee_quat (xyzw)
    - Call RB10Controller.compute_target_qpos_from_pose(pos, quat)
- Optionally execute on the robot
"""

import os
import sys
import time
import argparse
import json
import h5py
import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor

# Controller (make sure it's on PYTHONPATH)
from rb10_controller import RB10Controller

DEFAULT_HZ = 30.0


class WallRate:
    def __init__(self, hz: float):
        self._period = 1.0 / float(hz)
        self._next = time.perf_counter()
    def sleep(self):
        self._next += self._period
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            # If we lag behind, re-anchor to now
            self._next = time.perf_counter()


def parse_demo_name(demo_arg: str | int) -> str:
    """Accept 'demo_3' or '3' and normalize to 'demo_3'."""
    if isinstance(demo_arg, int):
        return f"demo_{demo_arg}"
    s = str(demo_arg).strip()
    if s.startswith("demo_"):
        return s
    if s.isdigit():
        return f"demo_{s}"
    raise ValueError(f"Invalid demo id: {demo_arg} (use '3' or 'demo_3')")


def load_hdf5_demo(h5_path: str, demo_name: str):
    """
    Returns:
        ee_pos  (N, 3): meters
        ee_quat (N, 4): quaternion (xyzw)
        hz (float): timeline_hz from meta if present, else DEFAULT_HZ
        meta (dict): parsed json from attrs["meta"]
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            raise KeyError("HDF5 missing 'data' group")
        g_data = f["data"]
        if demo_name not in g_data:
            demos = [k for k in g_data.keys() if k.startswith("demo_")]
            raise KeyError(f"{demo_name} not in HDF5. Available: {demos}")
        g_demo = g_data[demo_name]

        if "obs" not in g_demo:
            raise KeyError(f"{demo_name}/obs missing")
        g_obs = g_demo["obs"]

        if "ee_pos" not in g_obs or "ee_quat" not in g_obs:
            raise KeyError(f"{demo_name}/obs must have ee_pos and ee_quat")

        ee_pos  = np.asarray(g_obs["ee_pos"], dtype=np.float64)    # (N,3)
        ee_quat = np.asarray(g_obs["ee_quat"], dtype=np.float64)   # (N,4) xyzw
        N = ee_pos.shape[0]
        if N < 1 or ee_quat.shape[0] != N:
            raise ValueError(f"{demo_name} invalid sample sizes (ee_pos N={ee_pos.shape[0]}, ee_quat N={ee_quat.shape[0]})")

        # meta to get hz if available
        meta = {}
        hz = DEFAULT_HZ
        if "meta" in g_demo.attrs:
            try:
                meta = json.loads(g_demo.attrs["meta"])
                if "timeline_hz" in meta and float(meta["timeline_hz"]) > 0:
                    hz = float(meta["timeline_hz"])
            except Exception:
                pass

    return ee_pos, ee_quat, hz, meta


def main():
    parser = argparse.ArgumentParser(description="RB10 Demo Playback from robomimic HDF5 (ee_pos + ee_quat)")
    parser.add_argument("--h5", type=str, required=True, help="Path to robomimic HDF5 file")
    parser.add_argument("--demo", type=str, required=True, help="Demo id: e.g., '3' or 'demo_3'")
    parser.add_argument("--execute", action="store_true", help="Send joint commands to robot")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default 1.0)")
    parser.add_argument("--start", type=int, default=0, help="Start index (default 0)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--initial-move-seconds", type=float, default=10.0,
                        help="Seconds to move to initial pose before playback (when --execute). Default 10s.")
    args = parser.parse_args()

    demo_name = parse_demo_name(args.demo)
    ee_pos, ee_quat, base_hz, meta = load_hdf5_demo(args.h5, demo_name)

    N = ee_pos.shape[0]
    i0 = max(0, int(args.start))
    i1 = min(N, int(args.end)) if args.end is not None else N
    if i0 >= i1:
        print(f"[ERROR] Invalid range: start={i0}, end={i1}, N={N}")
        sys.exit(1)

    # Effective playback rate
    hz = max(1e-3, float(base_hz) * float(args.speed))
    print(f"[INFO] HDF5={os.path.basename(args.h5)} | demo={demo_name} | N={N} | play {i0}..{i1-1} "
          f"(count={i1-i0}) | base_hz={base_hz:.2f} | speed x{args.speed} -> hz={hz:.2f}")

    rclpy.init(args=None)
    ctrl = RB10Controller()

    executor = MultiThreadedExecutor()
    executor.add_node(ctrl)

    # Move to initial pose if executing
    start_k = i0
    p0 = ee_pos[i0].astype(float)         # (3,)
    q0 = ee_quat[i0].astype(float)        # (4,) xyzw
    q0_rad = ctrl.compute_target_qpos_from_pose(p0, q0, enforce_guard=False)

    if q0_rad is None:
        print("[ERROR] Initial IK failed; aborting playback.")
        try:
            ctrl.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
        sys.exit(1)

    if args.execute:
        try:
            secs = max(0.1, float(args.initial_move_seconds))
            print(f"[INFO] Moving to initial pose ({secs:.1f}s)...")
            ctrl.publish_qpos(np.asarray(q0_rad, dtype=float).tolist(), duration=secs)
        except Exception as e:
            print(f"[WARN] publish_qpos (initial) failed: {e}")
        input("[KEY] Press ENTER to start playback...")
        start_k = i0 + 1  # we've already moved to k=i0

    rate = WallRate(hz)

    for k in range(start_k, i1):
        # Allow controller callbacks to run
        executor.spin_once(timeout_sec=0.0)

        p = ee_pos[k].astype(float)       # (3,)
        q = ee_quat[k].astype(float)      # (4,) xyzw

        # IK
        q_rad = ctrl.compute_target_qpos_from_pose(p, q, enforce_guard=True)
        if q_rad is None:
            print(f"[WARN] k={k}: IK failed; skipping")
        else:
            if args.execute:
                try:
                    ctrl.publish_qpos(np.asarray(q_rad, dtype=float).tolist(), duration=max(0.05, 2.0 / hz))
                except Exception as e:
                    print(f"[WARN] publish_qpos failed at k={k}: {e}")
            else:
                print(f"k={k}: q={np.asarray(q_rad).round(4).tolist()}")

        rate.sleep()

    # cleanup
    try:
        ctrl.destroy_node()
    except Exception:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
