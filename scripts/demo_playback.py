#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Playback (robomimic HDF5, action: Δpos(3) + 6D rot(6) in 'actions')
- Loads a robomimic-compatible HDF5
- Selects a trajectory group (demo_0, demo_1, ...)
- Reconstructs absolute pose sequence by integrating:
    p_{t+1} = p_t + Δp_t
    R_{t+1} = R_t @ R_rel_t(6D)
- Each timestep:
    - Call RB10Controller.compute_target_qpos_from_pose(pos, quat_xyzw)
- Optionally execute on the robot

If 'actions' is missing, falls back to obs/ee_pos + obs/ee_quat direct playback.
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
from scripts.rbpodo_controller_movej import RB10Controller

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


# ====================== Rotation helpers (6D <-> R, R <-> quat) ======================
def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n

def sixd_to_rotmat(sixd: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation rep (Zhou et al. CVPR'19) to 3x3 rotation matrix.
    sixd: shape (6,), interpreted as [r11,r21,r31, r12,r22,r32] (first two columns)
    """
    a1 = sixd[:3].astype(np.float64)
    a2 = sixd[3:6].astype(np.float64)
    b1 = _normalize(a1)
    # remove b1 component from a2, then normalize
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = _normalize(a2_orth)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)  # columns
    return R

def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion (xyzw).
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=np.float64)
    # normalize for safety
    q /= max(1e-12, np.linalg.norm(q))
    return q

def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion (xyzw) -> 3x3 rotation matrix."""
    x, y, z, w = q.astype(np.float64)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz),  1 - 2*(xx + zz),      2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),    1 - 2*(xx + yy)]
    ], dtype=np.float64)
    return R
# ======================================================================================


def load_hdf5_for_actions(h5_path: str, demo_name: str):
    """
    Returns:
        ee_pos  (N,3)  absolute positions from obs (for seed / fallback)
        ee_quat (N,4)  absolute quats from obs (for seed / fallback)
        a_pos   (N,3)  Δpos (possibly normalized)   -- from actions[:, :3]
        a_rot6  (N,6)  6D relative rotation (possibly normalized) -- from actions[:, 3:]
        hz (float)
        meta (dict)
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

        # meta / hz
        meta = {}
        hz = DEFAULT_HZ
        if "meta" in g_demo.attrs:
            try:
                meta = json.loads(g_demo.attrs["meta"])
                if "timeline_hz" in meta and float(meta["timeline_hz"]) > 0:
                    hz = float(meta["timeline_hz"])
            except Exception:
                pass

        # obs (for seed / fallback)
        if "obs" not in g_demo:
            raise KeyError(f"{demo_name}/obs missing")
        g_obs = g_demo["obs"]
        if "ee_pos" not in g_obs or "ee_quat" not in g_obs:
            raise KeyError(f"{demo_name}/obs must have ee_pos and ee_quat")
        ee_pos = np.asarray(g_obs["ee_pos"], dtype=np.float64)    # (N,3)
        ee_quat = np.asarray(g_obs["ee_quat"], dtype=np.float64)  # (N,4)
        N = ee_pos.shape[0]
        if N < 1 or ee_quat.shape[0] != N:
            raise ValueError(f"{demo_name} invalid sample sizes (ee_pos N={ee_pos.shape[0]}, ee_quat N={ee_quat.shape[0]})")

        # actions (flat 9D): [Δx,Δy,Δz, r11,r21,r31, r12,r22,r32]
        a_pos = None
        a_rot6 = None
        if "actions" in g_demo:
            acts = np.asarray(g_demo["actions"], dtype=np.float64)  # (N,9) or (M,9)
            if acts.ndim != 2 or acts.shape[1] != 9:
                raise ValueError(f"{demo_name}/actions must have shape (N,9); got {acts.shape}")
            if acts.shape[0] != N:
                # Allow length mismatch but warn (rare). We'll crop to min length.
                L = min(N, acts.shape[0])
                print(f"[WARN] {demo_name}: actions length {acts.shape[0]} != obs length {N}, cropping to {L}")
                ee_pos = ee_pos[:L]
                ee_quat = ee_quat[:L]
                acts = acts[:L]
            a_pos  = acts[:, :3]
            a_rot6 = acts[:, 3:]

    return ee_pos, ee_quat, a_pos, a_rot6, hz, meta


def main():
    parser = argparse.ArgumentParser(description="Demo Playback from robomimic HDF5 (Δpos + 6D rel rot in 'actions', or fallback to obs)")
    parser.add_argument("--h5", type=str, required=True, help="Path to robomimic HDF5 file")
    parser.add_argument("--demo", type=str, required=True, help="Demo id: e.g., '3' or 'demo_3'")
    parser.add_argument("--execute", action="store_true", help="Send joint commands to robot")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default 1.0)")
    parser.add_argument("--start", type=int, default=0, help="Start index (default 0)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--initial-move-seconds", type=float, default=10.0,
                        help="Seconds to move to initial pose before playback (when --execute). Default 10s.")
    parser.add_argument("--prefer-actions", action="store_true", default=True,
                        help="Prefer reconstructing from action deltas (default True).")
    parser.add_argument("--no-prefer-actions", dest="prefer_actions", action="store_false")
    args = parser.parse_args()

    demo_name = parse_demo_name(args.demo)
    ee_pos, ee_quat, a_pos, a_rot6, base_hz, meta = load_hdf5_for_actions(args.h5, demo_name)

    N = ee_pos.shape[0]
    i0 = max(0, int(args.start))
    i1 = min(N, int(args.end)) if args.end is not None else N
    if i0 >= i1:
        print(f"[ERROR] Invalid range: start={i0}, end={i1}, N={N}")
        sys.exit(1)

    # Determine playback source
    use_actions = (a_pos is not None) and (a_rot6 is not None) and args.prefer_actions

    # Effective playback rate
    hz = max(1e-3, float(base_hz) * float(args.speed))
    src = "actions(Δpos+6D)" if use_actions else "obs(absolute ee_pos+ee_quat)"
    print(f"[INFO] HDF5={os.path.basename(args.h5)} | demo={demo_name} | N={N} | play {i0}..{i1-1} "
          f"(count={i1-i0}) | base_hz={base_hz:.2f} | speed x{args.speed} -> hz={hz:.2f} | source={src}")

    # If using actions, undo normalization if needed
    if use_actions and ("action_scale" in (meta or {})) and (meta["action_scale"] is not None):
        scale = np.asarray(meta["action_scale"], dtype=np.float64)  # len 9: [s_px,s_py,s_pz, s_r1..s_r6]
        if scale.shape == (9,):
            s_pos = scale[:3]
            s_rot = scale[3:]
            a_pos  = a_pos  * s_pos[None, :]
            a_rot6 = a_rot6 * s_rot[None, :]
        else:
            print(f"[WARN] meta.action_scale shape {scale.shape} != (9,), skip denorm.")

    # Build absolute pose sequence to feed controller
    if use_actions:
        # seed from obs at i0
        p_curr = ee_pos[i0].astype(np.float64).copy()
        q_curr = ee_quat[i0].astype(np.float64).copy()
        R_curr = quat_xyzw_to_rotmat(q_curr)

        # prepare arrays for playback (same length as loop count)
        seq_pos = [p_curr.copy()]
        seq_quat = [q_curr.copy()]

        # integrate forward using actions at indices i0..i1-2 to produce i0+1..i1-1
        for k in range(i0, i1 - 1):
            dp = a_pos[k]                        # Δpos at step k
            R_rel = sixd_to_rotmat(a_rot6[k])    # relative rotation for step k
            p_next = p_curr + dp
            R_next = R_curr @ R_rel
            q_next = rotmat_to_quat_xyzw(R_next)

            seq_pos.append(p_next.copy())
            seq_quat.append(q_next.copy())

            p_curr, R_curr = p_next, R_next

        seq_pos = np.asarray(seq_pos, dtype=np.float64)
        seq_quat = np.asarray(seq_quat, dtype=np.float64)
    else:
        # fallback: just use obs absolute trajectory slice
        seq_pos = ee_pos[i0:i1].astype(np.float64)
        seq_quat = ee_quat[i0:i1].astype(np.float64)

    # Init ROS / controller
    rclpy.init(args=None)
    ctrl = RB10Controller()
    executor = MultiThreadedExecutor()
    executor.add_node(ctrl)

    # Move to initial pose if executing
    p0 = seq_pos[0].astype(float)      # (3,)
    q0 = seq_quat[0].astype(float)     # (4,) xyzw
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
            ctrl.publish_joint_trajectory([np.asarray(q0_rad, dtype=float).tolist()], [secs])
        except Exception as e:
            print(f"[WARN] publish_joint_trajectory (initial) failed: {e}")
        input("[KEY] Press ENTER to start playback...")

    # Playback
    rate = WallRate(hz)
    for idx in range(seq_pos.shape[0]):
        executor.spin_once(timeout_sec=0.0)

        p = seq_pos[idx].astype(float)
        q = seq_quat[idx].astype(float)

        q_rad = ctrl.compute_target_qpos_from_pose(p, q, enforce_guard=True)
        if q_rad is None:
            print(f"[WARN] k={i0+idx}: IK failed; skipping")
        else:
            if args.execute:
                try:
                    ctrl.publish_joint_trajectory(
                        [np.asarray(q_rad, dtype=float).tolist()],
                        [max(0.05, 2.0 / hz)],
                    )
                except Exception as e:
                    print(f"[WARN] publish_joint_trajectory failed at k={i0+idx}: {e}")
            else:
                print(f"k={i0+idx}: q={np.asarray(q_rad).round(4).tolist()}")

        rate.sleep()

    # cleanup
    try:
        ctrl.destroy_node()
    except Exception:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
