#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add actions (9D) to a robomimic-style HDF5 by computing EE pose deltas from obs:
- Δpos (3): pos[t+1] - pos[t]
- rot6d (6): first two columns of R_rel, where R_rel = R(q[t])^T * R(q[t+1])
Actions are stored with length T (same as obs); last action is zeros.

Example:
  python add_actions_from_ee_obs_rot6d.py \
    --hdf5 /path/to/demo_20260122.hdf5 \
    --out  /path/to/demo_20260122_with_actions.hdf5 \
    --overwrite
"""

from __future__ import annotations

import argparse
import shutil

import h5py
import numpy as np


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

    # Standard quaternion->R (right-handed) for xyzw
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
    return: (..., 6) = [R[:,0], R[:,1]] (column-major concat of first 2 cols)
    """
    c1 = R[..., :, 0]  # (..., 3)
    c2 = R[..., :, 1]  # (..., 3)
    return np.concatenate([c1, c2], axis=-1)


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
    # R_rel[t] = R[t]^T * R[t+1]
    R_rel = np.matmul(np.transpose(R[:-1], (0, 2, 1)), R[1:])  # (T-1,3,3)
    rot6d = rotmat_to_rot6d(R_rel)  # (T-1,6)

    actions = np.zeros((T, 9), dtype=np.float32)
    actions[:-1, 0:3] = dpos.astype(np.float32)
    actions[:-1, 3:9] = rot6d.astype(np.float32)
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, help="input hdf5 (e.g., demo_20260122.hdf5)")
    ap.add_argument("--out", required=True, help="output hdf5")
    ap.add_argument("--pos-key", default="ee_pos", help="obs key for ee position")
    ap.add_argument("--quat-key", default="ee_quat", help="obs key for ee quaternion (xyzw)")
    ap.add_argument("--action-key", default="actions", help="dataset name for actions")
    ap.add_argument("--overwrite", action="store_true", help="overwrite actions if already exist")
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

            actions = compute_actions_rot6d(pos, quat)

            if args.action_key in g:
                if not args.overwrite:
                    print(f"[keep] {dk}: '{args.action_key}' already exists (use --overwrite to replace)")
                    n_skipped += 1
                    continue
                del g[args.action_key]

            g.create_dataset(
                args.action_key,
                data=actions,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                chunks=True,
            )
            # robomimic에서 보통 num_samples를 actions 길이로 맞추는 편이라, 있으면 갱신(선택)
            try:
                g.attrs["num_samples"] = int(actions.shape[0])
            except Exception:
                pass

            n_written += 1
            print(f"[ok] {dk}: wrote {args.action_key} shape={actions.shape}")

        print(f"\nDone. written={n_written}, skipped={n_skipped}, out={args.out}")


if __name__ == "__main__":
    main()