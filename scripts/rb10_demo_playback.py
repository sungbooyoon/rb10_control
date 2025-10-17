#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RB10 Demo Playback (robomimic HDF5)
- Loads a robomimic-compatible HDF5
- Selects a trajectory group (demo_0, demo_1, ...)
- Each timestep:
    - ee_pose = [x,y,z, yaw, pitch, roll]  # YPR = ZYX intrinsic (rzyx)
    - Convert YPR -> quaternion(xyzw)
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
from tf_transformations import quaternion_from_euler

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
        ee_pose (N, 6): [x,y,z, yaw, pitch, roll] (meters, radians; YPR=ZYX)
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
            # Show available demos for convenience
            demos = [k for k in g_data.keys() if k.startswith("demo_")]
            raise KeyError(f"{demo_name} not in HDF5. Available: {demos}")
        g_demo = g_data[demo_name]

        # obs / ee_pose (we stored pos + YPR here)
        if "obs" not in g_demo or "ee_pose" not in g_demo["obs"]:
            raise KeyError(f"{demo_name}/obs/ee_pose missing")

        ee_pose = np.asarray(g_demo["obs"]["ee_pose"], dtype=np.float64)  # (N, 6)
        N = ee_pose.shape[0]
        if N < 1:
            raise ValueError(f"{demo_name} has no samples")

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

    return ee_pose, hz, meta


def ypr_to_quat_xyzw(yaw, pitch, roll):
    """
    Convert YPR (ZYX intrinsic) into quaternion (xyzw).
    tf_transformations:
      quaternion_from_euler expects angles in the order that matches 'axes'.
      For R = Rz(yaw) * Ry(pitch) * Rx(roll), use axes='rzyx' and pass (yaw, pitch, roll).
    """
    qx, qy, qz, qw = quaternion_from_euler(yaw, pitch, roll, axes='rzyx')
    return np.array([qx, qy, qz, qw], dtype=float)


def main():
    parser = argparse.ArgumentParser(description="RB10 Demo Playback from robomimic HDF5")
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
    ee_pose, base_hz, meta = load_hdf5_demo(args.h5, demo_name)

    # ee_pose: [x,y,z, yaw, pitch, roll]  (YPR = ZYX intrinsic)
    N = ee_pose.shape[0]
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
    p0 = ee_pose[i0, :3].astype(float)
    y0, pch0, r0 = ee_pose[i0, 3:].astype(float)
    q0_xyzw = ypr_to_quat_xyzw(y0, pch0, r0)
    q0_rad = ctrl.compute_target_qpos_from_pose(p0, q0_xyzw, enforce_guard=False)

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
            secs = float(args.initial_move_seconds)
            secs = max(0.1, secs)
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

        # Extract pose at step k
        p = ee_pose[k, :3].astype(float)            # meters
        yaw, pitch, roll = ee_pose[k, 3:].astype(float)  # radians (YPR = ZYX)
        q_xyzw = ypr_to_quat_xyzw(yaw, pitch, roll)

        # IK
        q_rad = ctrl.compute_target_qpos_from_pose(p, q_xyzw, enforce_guard=True)
        if q_rad is None:
            print(f"[WARN] k={k}: IK failed; skipping")
        else:
            if args.execute:
                try:
                    # You can set a shorter smoothing duration if needed
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
