#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RB10 Demo Playback
- Loads a recorded NPZ demo (created by rb10_demo_recorder.py)
- For each timestep, converts tcp_pos [m,rad(ZYX)] to quaternion (xyzw)
- Calls RB10Controller.compute_target_qpos_from_pose(pos, q_xyzw)
- Optionally executes the joint command on the robot
"""
import os, sys, glob, time, argparse
import numpy as np
from tf_transformations import quaternion_from_euler

# ROS2 & controller
import rclpy
from rclpy.node import Node

# Expect rb10_controller.py to be discoverable in PYTHONPATH
from rb10_controller import RB10Controller

HZ = 30.0
PERIOD = 1.0 / HZ


def latest_npz(dirpath: str) -> str:
    files = sorted(glob.glob(os.path.join(dirpath, 'log_*.npz')))
    return files[-1] if files else None


def load_demo(path: str):
    data = np.load(path)
    tcp_pos = data['tcp_pos']            # [N,6] : [x,y,z (m), rx,ry,rz (rad, ZYX)]
    freedrive = data['freedrive']        # [N] : 0/1
    stamp = data['stamp'] if 'stamp' in data else None
    return tcp_pos, freedrive, stamp


def to_quat_xyzw_from_zyx(rx, ry, rz):
    """Build quaternion xyzw from ZYX Euler (R = Rz * Ry * Rx)."""
    # tf_transformations expects angles in the order matching 'axes'
    # For R = Rz(ψ)*Ry(θ)*Rx(φ), use axes='rzyx' and pass (ψ, θ, φ)
    qx, qy, qz, qw = quaternion_from_euler(rz, ry, rx, axes='rzyx')
    return np.array([qx, qy, qz, qw], dtype=float)


def try_send_joints(ctrl: RB10Controller, q_rad: np.ndarray, hold_sec: float = 0.0):
    """Try common send methods if available; otherwise, no-op."""
    # Prefer an explicit streaming/servo method if you have one; here we try common names.
    for name in ('send_joints', 'command_joints', 'move_joints'):
        if hasattr(ctrl, name):
            try:
                getattr(ctrl, name)(q_rad.tolist(), hold_sec)
                return True
            except Exception as e:
                print(f"[WARN] {name} failed: {e}")
    return False


def main():
    parser = argparse.ArgumentParser(description='RB10 Demo Playback')
    parser.add_argument('--file', type=str, default=None, help='Path to .npz demo (default: newest in ../dataset)')
    parser.add_argument('--execute', action='store_true', help='Send joint commands to robot while playing back')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier (default 1.0)')
    parser.add_argument('--start', type=int, default=0, help='Start index (default 0)')
    parser.add_argument('--end', type=int, default=None, help='End index (exclusive)')
    args = parser.parse_args()

    path = args.file or latest_npz('../dataset')
    if not path or not os.path.isfile(path):
        print('[ERROR] No .npz demo found. Use --file PATH or record with rb10_demo_recorder.py first.')
        sys.exit(1)

    tcp_pos_all, freedrive, stamp = load_demo(path)
    mask = (freedrive == 1)
    if not np.any(mask):
        print('[ERROR] No freedrive==1 samples found in the demo.')
        sys.exit(1)
    tcp_pos = tcp_pos_all[mask]
    if stamp is not None:
        stamp = stamp[mask]
    N = tcp_pos.shape[0]
    i0 = max(0, args.start)
    i1 = min(N, args.end) if args.end is not None else N
    if i0 >= i1:
        print(f'[ERROR] Invalid range: start={i0}, end={i1}, N={N}')
        sys.exit(1)

    print(f"[INFO] Playing demo {os.path.basename(path)} | freedrive frames {i0}..{i1-1} (N={i1-i0}/{N}) | speed x{args.speed}")

    rclpy.init(args=None)
    node = Node('rb10_demo_playback')

    # Initialize controller (adjust constructor as needed for your environment)
    ctrl = RB10Controller(node=node)

    start_k = i0

    p0 = tcp_pos[i0, :3].astype(float)
    rx0, ry0, rz0 = tcp_pos[i0, 3:].astype(float)
    q0_xyzw = to_quat_xyzw_from_zyx(rx0, ry0, rz0)
    q0_rad = ctrl.compute_target_qpos_from_pose(p0, q0_xyzw, enforce_guard=False)

    if q0_rad is None:
        print("[ERROR] Initial IK failed; aborting playback.")
        try:
            if hasattr(ctrl, 'destroy_node'):
                ctrl.destroy_node()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    else:
        if args.execute:
            # --- Move to initial pose over 10 seconds ---
            try:
                print("[INFO] Moving to initial pose (10s)...")
                ctrl.publish_qpos(np.asarray(q0_rad, dtype=float).tolist(), duration=10.0)
            except Exception as e:
                print(f"[WARN] publish_qpos (initial) failed: {e}")
            input("[KEY] Press ENTER to start playback...")
            start_k = i0 + 1  # we've already moved to k=i0

    next_t = time.perf_counter()
    for k in range(start_k, i1):
        p = tcp_pos[k, :3].astype(float)  # meters
        rx, ry, rz = tcp_pos[k, 3:].astype(float)  # radians (ZYX convention)

        q_xyzw = to_quat_xyzw_from_zyx(rx, ry, rz)

        # Compute target joints from pose
        q_rad = ctrl.compute_target_qpos_from_pose(p, q_xyzw, enforce_guard=True)

        if q_rad is None:
            print(f"[WARN] k={k}: IK failed; skipping")
        else:
            # Execute or print
            if args.execute:
                try:
                    ctrl.publish_qpos(np.asarray(q_rad, dtype=float).tolist(), duration=0.05)
                except Exception as e:
                    print(f"[WARN] publish_qpos failed at k={k}: {e}")
            else:
                print(f"k={k}: q={np.asarray(q_rad).round(4).tolist()}")

        # pacing
        period = PERIOD / max(1e-6, args.speed)
        next_t += period
        sleep_dur = next_t - time.perf_counter()
        if sleep_dur > 0:
            time.sleep(sleep_dur)
        else:
            next_t = time.perf_counter()

    # cleanup
    try:
        if hasattr(ctrl, 'destroy_node'):
            ctrl.destroy_node()
    except Exception:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
