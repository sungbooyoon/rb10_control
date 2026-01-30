#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add actions (9D) + goal (seam_origin, seam_rot6d, seam_length) to a robomimic-style HDF5.

Actions:
- Δpos (3): pos[t+1] - pos[t]
- rot6d (6): first two columns of R_rel, where R_rel = R(q[t])^T * R(q[t+1])
Actions are stored with length T (same as obs); last action is zeros.

Goal (per timestep, repeated across T):
- seam_origin: (3,)
- seam_rot6d: (6,) from provided seam quaternion xyzw
- seam_length: (1,) constant 0.3

Seam assignment by demo index i:
- seam_id = (i // 4) % 18 + 1

Example:
  python add_actions_and_goal.py \
    --hdf5 /path/to/demo_20260122.hdf5 \
    --out  /path/to/demo_20260122_actions_goal.hdf5 \
    --overwrite
"""

from __future__ import annotations

import argparse
import shutil

import h5py
import numpy as np


# -------------------------
# Quaternion / Rotation utils
# -------------------------

def q_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return q / n


def ensure_quat_continuity(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Make quaternion signs continuous over time (q and -q represent same rotation).
    """
    q = q_normalize(q_xyzw)
    dots = np.sum(q[1:] * q[:-1], axis=-1)
    flip = dots < 0
    q_out = q.copy()
    q_out[1:][flip] *= -1.0
    return q_out


def quat_xyzw_to_R(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert quaternion(s) in xyzw to rotation matrix/matrices.
    Input: (4,) or (N,4)
    Output: (3,3) or (N,3,3)
    """
    q = q_normalize(q_xyzw.astype(np.float64))
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = ww + xx - yy - zz
    R[..., 0, 1] = 2 * (xy - zw)
    R[..., 0, 2] = 2 * (xz + yw)

    R[..., 1, 0] = 2 * (xy + zw)
    R[..., 1, 1] = ww - xx + yy - zz
    R[..., 1, 2] = 2 * (yz - xw)

    R[..., 2, 0] = 2 * (xz - yw)
    R[..., 2, 1] = 2 * (yz + xw)
    R[..., 2, 2] = ww - xx - yy + zz
    return R


def rotmat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """
    R: (..., 3, 3)
    return: (..., 6) = [R[:,0], R[:,1]] (concat first 2 columns)
    """
    c1 = R[..., :, 0]  # (..., 3)
    c2 = R[..., :, 1]  # (..., 3)
    return np.concatenate([c1, c2], axis=-1)


def quat_xyzw_to_rot6d(q_xyzw: np.ndarray) -> np.ndarray:
    """
    q_xyzw: (...,4) xyzw
    returns rot6d: (...,6)
    """
    R = quat_xyzw_to_R(q_xyzw)
    return rotmat_to_rot6d(R)


# -------------------------
# Action computation
# -------------------------

def compute_actions_rot6d(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """
    pos: (T,3)
    quat: (T,4) xyzw
    returns actions: (T,9) [dpos(3), rot6d(6)]
    """
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"ee_pos must be (T,3). Got {pos.shape}")
    if quat_xyzw.ndim != 2 or quat_xyzw.shape[1] != 4:
        raise ValueError(f"ee_quat must be (T,4) xyzw. Got {quat_xyzw.shape}")
    if pos.shape[0] != quat_xyzw.shape[0]:
        raise ValueError(f"T mismatch: pos {pos.shape[0]} vs quat {quat_xyzw.shape[0]}")

    T = pos.shape[0]
    if T == 0:
        return np.zeros((0, 9), dtype=np.float32)
    if T == 1:
        return np.zeros((1, 9), dtype=np.float32)

    q = ensure_quat_continuity(quat_xyzw)
    dpos = pos[1:] - pos[:-1]  # (T-1,3)

    R = quat_xyzw_to_R(q)  # (T,3,3)
    R_rel = np.matmul(np.transpose(R[:-1], (0, 2, 1)), R[1:])  # (T-1,3,3)
    rot6d = rotmat_to_rot6d(R_rel)  # (T-1,6)

    actions = np.zeros((T, 9), dtype=np.float32)
    actions[:-1, 0:3] = dpos.astype(np.float32)
    actions[:-1, 3:9] = rot6d.astype(np.float32)
    return actions


# -------------------------
# Seam spec + assignment
# -------------------------

SEAM_LENGTH_CONST = 0.3  # meters

# seam_id -> (origin_xyz, quat_xyzw)
SEAMS = {
    1:  ((0.779, -0.390, 1.136), (0.5, -0.5, -0.5, 0.5)),
    2:  ((0.779, -0.390, 0.836), (0.5, -0.5, -0.5, 0.5)),
    3:  ((0.779, -0.390, 0.536), (0.5, -0.5, -0.5, 0.5)),
    4:  ((0.779, -0.090, 1.136), (0.5, -0.5, -0.5, 0.5)),
    5:  ((0.779, -0.090, 0.836), (0.5, -0.5, -0.5, 0.5)),
    6:  ((0.779, -0.090, 0.536), (0.5, -0.5, -0.5, 0.5)),
    7:  ((0.779,  0.210, 1.136), (0.5, -0.5, -0.5, 0.5)),
    8:  ((0.779,  0.210, 0.836), (0.5, -0.5, -0.5, 0.5)),
    9:  ((0.779,  0.210, 0.536), (0.5, -0.5, -0.5, 0.5)),
    10: ((0.779, -0.390, 0.836), (0.5, -0.5, -0.5, 0.5)),
    11: ((0.779, -0.390, 0.536), (0.5, -0.5, -0.5, 0.5)),
    12: ((0.779, -0.390, 0.236), (0.5, -0.5, -0.5, 0.5)),
    13: ((0.779, -0.090, 0.836), (0.5, -0.5, -0.5, 0.5)),
    14: ((0.779, -0.090, 0.536), (0.5, -0.5, -0.5, 0.5)),
    15: ((0.779, -0.090, 0.236), (0.5, -0.5, -0.5, 0.5)),
    16: ((0.779,  0.210, 0.836), (0.5, -0.5, -0.5, 0.5)),
    17: ((0.779,  0.210, 0.536), (0.5, -0.5, -0.5, 0.5)),
    18: ((0.779,  0.210, 0.236), (0.5, -0.5, -0.5, 0.5)),
}


def seam_id_for_demo(demo_name: str) -> int:
    """
    demo_0..demo_3 -> seam 1
    demo_4..demo_7 -> seam 2
    ...
    seam 18 then wraps back to seam 1
    """
    if not demo_name.startswith("demo_"):
        raise ValueError(f"Unexpected demo name: {demo_name}")
    idx = int(demo_name[5:])
    return (idx // 4) % 18 + 1


def write_goal_group(g: h5py.Group, T: int, seam_id: int, overwrite: bool = True):
    """
    Create (or overwrite) /data/demo_x/goal/{seam_origin,seam_rot6d,seam_length}
    with shapes (T,3), (T,6), (T,1)
    """
    if seam_id not in SEAMS:
        raise KeyError(f"Unknown seam_id={seam_id}")

    origin_xyz, quat_xyzw = SEAMS[seam_id]
    origin = np.array(origin_xyz, dtype=np.float32)  # (3,)
    q = np.array(quat_xyzw, dtype=np.float32)       # (4,)
    seam_rot6d = quat_xyzw_to_rot6d(q[None, :]).astype(np.float32)[0]  # (6,)
    seam_length = np.array([SEAM_LENGTH_CONST], dtype=np.float32)      # (1,)

    # repeat for T timesteps
    origin_T = np.repeat(origin[None, :], T, axis=0)           # (T,3)
    rot6d_T = np.repeat(seam_rot6d[None, :], T, axis=0)        # (T,6)
    length_T = np.repeat(seam_length[None, :], T, axis=0)      # (T,1)

    if "goal" not in g:
        goal = g.create_group("goal")
    else:
        goal = g["goal"]

    def _write(name: str, arr: np.ndarray):
        if name in goal:
            if overwrite:
                del goal[name]
            else:
                return False
        goal.create_dataset(
            name,
            data=arr,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=True,
        )
        return True

    _write("seam_origin", origin_T)
    _write("seam_rot6d", rot6d_T)
    _write("seam_length", length_T)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, help="input hdf5 (e.g., demo_20260122.hdf5)")
    ap.add_argument("--out", required=True, help="output hdf5")
    ap.add_argument("--pos-key", default="ee_pos", help="obs key for ee position")
    ap.add_argument("--quat-key", default="ee_quat", help="obs key for ee quaternion (xyzw)")
    ap.add_argument("--action-key", default="actions", help="dataset name for actions")
    ap.add_argument("--overwrite", action="store_true", help="overwrite actions/goal if already exist")
    args = ap.parse_args()

    shutil.copy(args.hdf5, args.out)

    with h5py.File(args.out, "r+") as f:
        if "data" not in f:
            raise KeyError("No '/data' group found. Is this a robomimic-style hdf5?")
        data = f["data"]

        # /data 아래에 mask 등이 섞여있어도 demo_숫자만 처리
        demo_keys = [k for k in data.keys() if k.startswith("demo_") and k[5:].isdigit()]
        demo_keys = sorted(demo_keys, key=lambda x: int(x[5:]))

        if not demo_keys:
            raise RuntimeError("No demo_* groups found under /data")

        n_written = 0
        n_skipped = 0

        for dk in demo_keys:
            g = data[dk]
            if "obs" not in g:
                print(f"[skip] {dk}: no obs group")
                n_skipped += 1
                continue

            obs = g["obs"]
            if args.pos_key not in obs or args.quat_key not in obs:
                print(f"[skip] {dk}: missing obs/{args.pos_key} or obs/{args.quat_key}")
                n_skipped += 1
                continue

            pos = obs[args.pos_key][...]
            quat = obs[args.quat_key][...]

            # ---- actions (T,9)
            actions = compute_actions_rot6d(pos, quat)

            if args.action_key in g:
                if not args.overwrite:
                    print(f"[keep] {dk}: '{args.action_key}' already exists (use --overwrite to replace)")
                else:
                    del g[args.action_key]
                    g.create_dataset(
                        args.action_key,
                        data=actions,
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                        chunks=True,
                    )
            else:
                g.create_dataset(
                    args.action_key,
                    data=actions,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    chunks=True,
                )

            # ---- goal (repeat across T)
            T = int(pos.shape[0])
            seam_id = seam_id_for_demo(dk)
            write_goal_group(g, T=T, seam_id=seam_id, overwrite=args.overwrite)

            # num_samples 맞추기(선택)
            try:
                g.attrs["num_samples"] = int(actions.shape[0])
            except Exception:
                pass

            n_written += 1
            print(f"[ok] {dk}: actions {actions.shape} + goal(seam_{seam_id})")

        print(f"\nDone. written={n_written}, skipped={n_skipped}, out={args.out}")


if __name__ == "__main__":
    main()