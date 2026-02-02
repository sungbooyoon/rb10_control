#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild HDF5 using NPZ X_local_crop (pos+quat 7D) as ee_pos/ee_quat ground truth,
and crop ALL other time-series obs (joint_pos, rgb, etc.) from HDF5 using NPZ crop indices.
Then add actions (9D) + goal(seam_*).

What changes vs previous version:
- No phase_len, no W_phase_crop exp-map.
- ee_pos/ee_quat are replaced by NPZ X_local_crop (N,7) sliced per demo via demo_ptr_crop.
- Other obs are sliced from HDF5 by crop_s/crop_e, BUT we auto-correct the window length
  to match the per-demo X_local_crop segment length (seg_len), to keep robomimic consistency.

NPZ required:
- X_local_crop      (N,7)  = [pos(3), quat(4)]
- demo_ptr_crop     (D+1,)
- crop_s, crop_e    (D,), (D,)  (or other key aliases supported below)

HDF5:
- robomimic style: /data/demo_x/obs/*

Example:
  python rebuild_hdf5_from_xlocalcrop.py \
    --npz  /home/sungboo/rb10_control/dataset/demo_20260122_final.npz \
    --hdf5 /home/sungboo/rb10_control/data/demo_20260122_224+224_final.hdf5 \
    --out  /home/sungboo/rb10_control/data/demo_20260122_xlocalcrop_actions_goal.hdf5 \
    --overwrite
"""

from __future__ import annotations

import argparse
import shutil
from typing import Dict, Optional, List, Tuple

import h5py
import numpy as np


# -------------------------
# Basic utils
# -------------------------

def _first_existing(npz: Dict[str, np.ndarray], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in npz:
            return k
    return None


def _h5_write(g: h5py.Group, name: str, arr: np.ndarray, overwrite: bool):
    if name in g:
        if overwrite:
            del g[name]
        else:
            return
    g.create_dataset(
        name, data=arr,
        compression="gzip", compression_opts=4,
        shuffle=True, chunks=True,
    )


def is_timeseries_dataset(dset: h5py.Dataset, T_ref: int) -> bool:
    if dset.shape is None or len(dset.shape) == 0:
        return False
    return int(dset.shape[0]) == int(T_ref)


# -------------------------
# Quaternion / Rotation utils
# -------------------------

def q_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return q / n


def ensure_quat_continuity(q_xyzw: np.ndarray) -> np.ndarray:
    q = q_normalize(q_xyzw.astype(np.float64))
    if q.shape[0] <= 1:
        return q.astype(np.float64)
    q_out = q.copy()
    for t in range(1, q.shape[0]):
        if np.dot(q_out[t - 1], q_out[t]) < 0:
            q_out[t] *= -1.0
    return q_out


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz)
    if q.shape[-1] != 4:
        raise ValueError(f"quat must be (...,4). got {q.shape}")
    return np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)


def quat_xyzw_to_R(q_xyzw: np.ndarray) -> np.ndarray:
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
    c1 = R[..., :, 0]
    c2 = R[..., :, 1]
    return np.concatenate([c1, c2], axis=-1)


def compute_actions_rot6d(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
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
    dpos = pos[1:] - pos[:-1]

    R = quat_xyzw_to_R(q)
    R_rel = np.matmul(np.transpose(R[:-1], (0, 2, 1)), R[1:])
    rot6d = rotmat_to_rot6d(R_rel)

    actions = np.zeros((T, 9), dtype=np.float32)
    actions[:-1, 0:3] = dpos.astype(np.float32)
    actions[:-1, 3:9] = rot6d.astype(np.float32)
    return actions


# -------------------------
# Seam spec + assignment (네 코드 그대로)
# -------------------------

SEAM_LENGTH_CONST = 0.3

def q_mul(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    a = np.asarray(a_xyzw, dtype=np.float64)
    b = np.asarray(b_xyzw, dtype=np.float64)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return np.stack([x, y, z, w], axis=-1)


Q_VERT = np.array([0.5, -0.5, -0.5, 0.5], dtype=np.float64)
Q_Z90  = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)], dtype=np.float64)
Q_HORZ = q_normalize(q_mul(Q_Z90, Q_VERT))

SEAMS = {
    1:  ((0.779, -0.390, 1.136), tuple(Q_VERT.tolist())),
    2:  ((0.779, -0.390, 0.836), tuple(Q_VERT.tolist())),
    3:  ((0.779, -0.390, 0.536), tuple(Q_VERT.tolist())),
    4:  ((0.779, -0.090, 1.136), tuple(Q_VERT.tolist())),
    5:  ((0.779, -0.090, 0.836), tuple(Q_VERT.tolist())),
    6:  ((0.779, -0.090, 0.536), tuple(Q_VERT.tolist())),
    7:  ((0.779,  0.210, 1.136), tuple(Q_VERT.tolist())),
    8:  ((0.779,  0.210, 0.836), tuple(Q_VERT.tolist())),
    9:  ((0.779,  0.210, 0.536), tuple(Q_VERT.tolist())),

    10: ((0.779, -0.390, 0.836), tuple(Q_HORZ.tolist())),
    11: ((0.779, -0.390, 0.536), tuple(Q_HORZ.tolist())),
    12: ((0.779, -0.390, 0.236), tuple(Q_HORZ.tolist())),
    13: ((0.779, -0.090, 0.836), tuple(Q_HORZ.tolist())),
    14: ((0.779, -0.090, 0.536), tuple(Q_HORZ.tolist())),
    15: ((0.779, -0.090, 0.236), tuple(Q_HORZ.tolist())),
    16: ((0.779,  0.210, 0.836), tuple(Q_HORZ.tolist())),
    17: ((0.779,  0.210, 0.536), tuple(Q_HORZ.tolist())),
    18: ((0.779,  0.210, 0.236), tuple(Q_HORZ.tolist())),
}


def seam_id_for_demo(demo_name: str) -> int:
    if not demo_name.startswith("demo_"):
        raise ValueError(f"Unexpected demo name: {demo_name}")
    idx = int(demo_name[5:])
    return (idx // 4) % 18 + 1


def quat_xyzw_to_rot6d(q_xyzw: np.ndarray) -> np.ndarray:
    R = quat_xyzw_to_R(q_xyzw)
    return rotmat_to_rot6d(R)


def write_goal_into_obs(demo_g: h5py.Group, T: int, seam_id: int, overwrite: bool = True):
    obs = demo_g["obs"]
    origin_xyz, quat_xyzw = SEAMS[seam_id]
    origin = np.array(origin_xyz, dtype=np.float32)
    q = np.array(quat_xyzw, dtype=np.float32)

    seam_rot6d = quat_xyzw_to_rot6d(q[None, :]).astype(np.float32)[0]
    seam_length = np.array([SEAM_LENGTH_CONST], dtype=np.float32)

    origin_T = np.repeat(origin[None, :], T, axis=0)
    rot6d_T  = np.repeat(seam_rot6d[None, :], T, axis=0)
    len_T    = np.repeat(seam_length[None, :], T, axis=0)

    _h5_write(obs, "seam_origin", origin_T, overwrite)
    _h5_write(obs, "seam_rot6d", rot6d_T, overwrite)
    _h5_write(obs, "seam_length", len_T, overwrite)


def write_goal_group(demo_g: h5py.Group, T: int, seam_id: int, overwrite: bool = True):
    goal = demo_g.require_group("goal")
    origin_xyz, quat_xyzw = SEAMS[seam_id]
    origin = np.array(origin_xyz, dtype=np.float32)
    q = np.array(quat_xyzw, dtype=np.float32)

    seam_rot6d = quat_xyzw_to_rot6d(q[None, :]).astype(np.float32)[0]
    seam_length = np.array([SEAM_LENGTH_CONST], dtype=np.float32)

    origin_T = np.repeat(origin[None, :], T, axis=0)
    rot6d_T  = np.repeat(seam_rot6d[None, :], T, axis=0)
    len_T    = np.repeat(seam_length[None, :], T, axis=0)

    _h5_write(goal, "seam_origin", origin_T, overwrite)
    _h5_write(goal, "seam_rot6d", rot6d_T, overwrite)
    _h5_write(goal, "seam_length", len_T, overwrite)


# -------------------------
# NPZ loading: crop indices + X_local_crop segments
# -------------------------

def load_crop_indices_from_npz(npz: Dict[str, np.ndarray], n_demos: int) -> Tuple[np.ndarray, np.ndarray]:
    start_key = _first_existing(npz, ["crop_s", "crop_start_idx", "crop_start", "start_idx_crop", "start_idx"])
    end_key   = _first_existing(npz, ["crop_e", "crop_end_idx", "crop_end", "end_idx_crop", "end_idx"])
    if start_key is None or end_key is None:
        raise KeyError(f"NPZ missing crop_s/crop_e (or aliases). keys={sorted(list(npz.keys()))}")

    s = np.asarray(npz[start_key]).astype(np.int64).reshape(-1)
    e = np.asarray(npz[end_key]).astype(np.int64).reshape(-1)
    if s.shape[0] != n_demos or e.shape[0] != n_demos:
        raise ValueError(f"crop index length mismatch: s={s.shape}, e={e.shape}, n_demos={n_demos}")
    return s, e


def load_xlocalcrop_segment(npz: Dict[str, np.ndarray], demo_i: int, quat_order: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pos (T,3), quat_xyzw (T,4) from X_local_crop using demo_ptr_crop
    """
    if "X_local_crop" not in npz or "demo_ptr_crop" not in npz:
        raise KeyError("NPZ must contain X_local_crop and demo_ptr_crop")

    Xc = np.asarray(npz["X_local_crop"], dtype=np.float32)  # (N,7)
    ptr = np.asarray(npz["demo_ptr_crop"]).astype(np.int64).reshape(-1)

    s = int(ptr[demo_i])
    e = int(ptr[demo_i + 1])
    seg = Xc[s:e]  # (T,7)
    if seg.ndim != 2 or seg.shape[1] != 7:
        raise ValueError(f"X_local_crop segment must be (T,7). got {seg.shape}")

    pos = seg[:, 0:3].astype(np.float32)
    quat = seg[:, 3:7].astype(np.float32)

    if quat_order.lower() == "wxyz":
        quat = quat_wxyz_to_xyzw(quat)
    elif quat_order.lower() != "xyzw":
        raise ValueError("--quat-order must be xyzw or wxyz")

    quat = ensure_quat_continuity(quat).astype(np.float32)
    return pos, quat


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pos-key", default="ee_pos")
    ap.add_argument("--quat-key", default="ee_quat")
    ap.add_argument("--action-key", default="actions")
    ap.add_argument("--quat-order", default="xyzw", choices=["xyzw", "wxyz"],
                    help="Quaternion order stored in X_local_crop (default: xyzw)")
    args = ap.parse_args()

    npz_obj = np.load(args.npz, allow_pickle=True)
    npz: Dict[str, np.ndarray] = {k: npz_obj[k] for k in npz_obj.files}

    shutil.copy(args.hdf5, args.out)

    with h5py.File(args.out, "r+") as f:
        if "data" not in f:
            raise KeyError("No '/data' group found.")
        data = f["data"]

        demo_keys = [k for k in data.keys() if k.startswith("demo_") and k[5:].isdigit()]
        demo_keys = sorted(demo_keys, key=lambda x: int(x[5:]))

        if not demo_keys:
            raise RuntimeError("No demo_* found under /data")

        n_demos = len(demo_keys)

        # sanity: NPZ demo_ptr_crop should match demo count
        if "demo_ptr_crop" not in npz:
            raise KeyError("NPZ missing demo_ptr_crop")
        ptr = np.asarray(npz["demo_ptr_crop"]).astype(np.int64).reshape(-1)
        if ptr.shape[0] != n_demos + 1:
            raise ValueError(f"demo_ptr_crop shape={ptr.shape} but expected (n_demos+1)={(n_demos+1,)}")

        crop_s, crop_e = load_crop_indices_from_npz(npz, n_demos=n_demos)

        n_ok, n_skip = 0, 0

        for i, dk in enumerate(demo_keys):
            g = data[dk]
            if "obs" not in g:
                print(f"[skip] {dk}: no obs")
                n_skip += 1
                continue
            obs = g["obs"]

            # reference length in HDF5 (pre-crop stream length)
            if args.pos_key not in obs:
                print(f"[skip] {dk}: missing obs/{args.pos_key}")
                n_skip += 1
                continue

            T_full = int(obs[args.pos_key].shape[0])

            # load target pose segment length from NPZ X_local_crop
            pos_npz, quat_npz = load_xlocalcrop_segment(npz, demo_i=i, quat_order=args.quat_order)
            seg_len = int(pos_npz.shape[0])

            # NPZ crop indices give an initial window; we will correct to seg_len
            s0 = int(crop_s[i])
            e0 = int(crop_e[i])

            # clamp initial
            s_cl = max(0, min(s0, T_full - 1))
            e_cl = max(0, min(e0, T_full))
            if e_cl <= s_cl:
                print(f"[skip] {dk}: invalid initial crop [{s0}:{e0}) clamped [{s_cl}:{e_cl})")
                n_skip += 1
                continue

            # ---- FORCE window length = seg_len (robomimic consistency)
            s2 = s_cl
            e2 = s2 + seg_len

            if e2 > T_full:
                e2 = T_full
                s2 = e2 - seg_len

            if s2 < 0 or e2 <= s2:
                raise RuntimeError(
                    f"{dk}: cannot fit seg_len={seg_len} into T_full={T_full} "
                    f"(initial [{s_cl}:{e_cl}))"
                )

            s_cl, e_cl = int(s2), int(e2)
            crop_len = int(e_cl - s_cl)
            if crop_len != seg_len:
                raise RuntimeError(
                    f"{dk}: crop_len({crop_len}) != seg_len({seg_len}) after correction "
                    f"[{s_cl}:{e_cl}) T_full={T_full}"
                )

            if (e0 - s0) != seg_len:
                # informative only
                print(f"[fix] {dk}: initial crop_len={e0-s0} -> corrected [{s_cl}:{e_cl}) len={seg_len}")

            # ---- (A) crop all other time-series obs by corrected window
            for key in list(obs.keys()):
                if key in (args.pos_key, args.quat_key):
                    continue
                dset = obs[key]
                if isinstance(dset, h5py.Dataset) and is_timeseries_dataset(dset, T_full):
                    arr = dset[s_cl:e_cl]
                    _h5_write(obs, key, arr, overwrite=args.overwrite)

            # ---- (B) overwrite ee_pos/ee_quat from NPZ X_local_crop
            _h5_write(obs, args.pos_key, pos_npz.astype(np.float32), overwrite=args.overwrite)
            _h5_write(obs, args.quat_key, quat_npz.astype(np.float32), overwrite=args.overwrite)

            # ---- (C) actions
            actions = compute_actions_rot6d(pos_npz.astype(np.float32), quat_npz.astype(np.float32))
            _h5_write(g, args.action_key, actions, overwrite=args.overwrite)

            # ---- (D) goal (repeat across T)
            T = int(seg_len)
            seam_id = seam_id_for_demo(dk)
            write_goal_group(g, T=T, seam_id=seam_id, overwrite=args.overwrite)
            write_goal_into_obs(g, T=T, seam_id=seam_id, overwrite=args.overwrite)

            try:
                g.attrs["num_samples"] = int(T)
            except Exception:
                pass

            n_ok += 1
            print(f"[ok] {dk}: window[{s_cl}:{e_cl}) len={seg_len} | ee_pose<-X_local_crop | actions={actions.shape} | seam={seam_id}")

        print(f"\nDone. ok={n_ok}, skipped={n_skip}")
        print(f"out: {args.out}")


if __name__ == "__main__":
    main()