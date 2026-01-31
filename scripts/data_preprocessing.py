#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-demo contact-origin local frame transform + verification plots (NO z-flip)
+ crop & save around contact segment
+ NEW:
  - event-based phase alignment (chosen_idx -> phase 0, contact_end -> phase 1)
  - quaternion sign-unwrapping (within demo)
  - SO(3) log-map to R^3 (rotvec) relative to reference at chosen_idx
  - save phase-aligned fixed-length sequences for MP training:
      X_phase_crop : local position (phase_len, 3) per demo concatenated
      W_phase_crop : local rotvec   (phase_len, 3) per demo concatenated
      demo_ptr_phase, demo_names_phase
      phase_grid

Hard constraints:
  - require_contact = ON
  - require_stable  = ON

Plane:
  d(t) = n·p(t) - plane_offset

Contact:
  first_contact = first index where d(t) <= contact_d_thresh

Stability:
  find first index where BOTH:
    d(t) <= contact_d_thresh
    |v_n_smooth(t)| <= vn_eps
  for stable_len consecutive steps (searched after first_contact + min_after_contact)

Local frame per demo:
  z = wall normal n  (from wall_quat)
  y = progression direction averaged over stable window, projected onto plane
  x = y × z (right-handed), then re-orthogonalize y = z × x
Origin:
  origin = pos[chosen_idx]  and force z_local(chosen)=0

Crop:
  [chosen_idx-pad .. last_contact+pad] where last_contact found by z_local sign crossing

Phase alignment (event-based):
  chosen_idx_crop -> phase 0
  contact_end_idx_crop -> phase 1
  piecewise-linear time warp:
    [0 .. chosen]   -> [0 .. 0] (collapsed), handled by mapping chosen->0 and earlier portion to [0, eps]
    [chosen .. end] -> [0 .. 1]
  Practically:
    build a monotonic phase per sample:
      phase[t] = (t - chosen) / max(1, (contact_end - chosen))
    then resample onto fixed phase grid [0..1] with interpolation.
  (This is robust, and keeps your crop pad samples.)

SO(3) log map:
  - unwrap quaternion signs within demo
  - reference orientation = q_local[chosen]
  - R_rel(t) = R_ref^T R(t)
  - w(t) = log(R_rel(t)) in R^3 (axis-angle vector)

Saves:
  - original outputs + X_local, X_crop, X_local_crop
  - NEW: X_phase_crop, W_phase_crop, phase_grid, demo_ptr_phase, demo_names_phase
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


# ----------------------------
# Quaternion utils (xyzw)
# ----------------------------
def q_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, xyzw."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float64)


def q_conj(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64)


def R_from_q(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from quaternion (xyzw). Returns 3x3."""
    x, y, z, w = q_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def q_from_R(R: np.ndarray) -> np.ndarray:
    """Quaternion (xyzw) from rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S

    return q_normalize(np.array([x, y, z, w], dtype=np.float64))


def unwrap_quat_signs(quat: np.ndarray) -> np.ndarray:
    """
    Ensure temporal continuity: if dot(q[t], q[t-1]) < 0, flip q[t].
    quat: (T,4) xyzw
    """
    q = np.asarray(quat, dtype=np.float64).copy()
    if q.shape[0] <= 1:
        return q
    q[0] = q_normalize(q[0])
    for t in range(1, q.shape[0]):
        q[t] = q_normalize(q[t])
        if float(np.dot(q[t], q[t - 1])) < 0.0:
            q[t] *= -1.0
    return q


# ----------------------------
# SO(3) log map (rotation matrix -> rotvec)
# ----------------------------
def so3_log(R: np.ndarray) -> np.ndarray:
    """
    Log map from SO(3) to R^3 (rotation vector).
    Uses stable acos+vee formula; handles small angles.
    """
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))

    if theta < 1e-8:
        # small-angle approx: vee(R - R^T)/2
        w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
        return w

    # general case
    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=np.float64)
    return theta * w


def quat_to_rotvec_rel(quat_local: np.ndarray, ref_idx: int) -> np.ndarray:
    """
    quat_local: (T,4) xyzw (should be unwrapped)
    ref_idx: chosen_idx (>=0)
    Returns rotvec w(t) where:
      R_rel = R_ref^T R_t
      w = log(R_rel) in R^3
    """
    ql = np.asarray(quat_local, dtype=np.float64)
    T = ql.shape[0]
    if ref_idx < 0 or ref_idx >= T:
        return np.zeros((T, 3), dtype=np.float64)

    R_ref = R_from_q(ql[ref_idx])
    R_ref_T = R_ref.T
    w = np.zeros((T, 3), dtype=np.float64)
    for t in range(T):
        Rt = R_from_q(ql[t])
        R_rel = R_ref_T @ Rt
        w[t] = so3_log(R_rel)
    return w


# ----------------------------
# Geometry helpers
# ----------------------------
def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def project_to_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - np.dot(v, n) * n


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average with edge padding. win=1 -> identity."""
    x = np.asarray(x, dtype=np.float64)
    if win <= 1:
        return x.copy()
    win = int(win)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(xp, kernel, mode="valid")


def _first_stable_index(mask: np.ndarray, stable_len: int) -> int:
    T = mask.shape[0]
    if stable_len <= 1:
        idx = np.where(mask)[0]
        return int(idx[0]) if idx.size else -1
    if T < stable_len:
        return -1
    s = np.convolve(mask.astype(np.int32), np.ones(stable_len, dtype=np.int32), mode="valid")
    hits = np.where(s == stable_len)[0]
    return int(hits[0]) if hits.size else -1


def compute_d_and_vn(pos: np.ndarray, n: np.ndarray, plane_offset: float) -> tuple[np.ndarray, np.ndarray]:
    d = pos @ n - plane_offset
    dp = np.diff(pos, axis=0)
    vn = np.zeros((pos.shape[0],), dtype=np.float64)
    vn[1:] = dp @ n  # m/step
    return d, vn


def stable_window_direction(
    pos: np.ndarray,
    n: np.ndarray,
    stable_idx: int,
    stable_len: int,
    use_mean_diffs: bool = True,
) -> np.ndarray:
    """Average motion direction within the stable window, projected to plane."""
    if stable_idx < 0:
        return np.zeros((3,), dtype=np.float64)

    a = stable_idx
    b = min(pos.shape[0] - 1, stable_idx + stable_len - 1)
    if b <= a:
        return np.zeros((3,), dtype=np.float64)

    if use_mean_diffs:
        seg = pos[a : b + 1]
        v = np.diff(seg, axis=0).mean(axis=0)
    else:
        v = pos[b] - pos[a]

    return unit(project_to_plane(v, n))


def find_contact_and_stable(
    d: np.ndarray,
    vn_smooth: np.ndarray,
    vn_eps: float,
    stable_len: int,
    min_after_contact: int,
    contact_d_thresh: float,
    min_start: int,
) -> tuple[int, int]:
    """
    HARD:
      - search starts only after min_start
      - first_contact: first t>=min_start with d(t) <= thresh
      - stable: stable_len consecutive steps of (d<=thresh AND |vn_smooth|<=vn_eps),
        searched after max(min_start, first_contact + min_after_contact)
    """
    T = int(d.shape[0])
    if T == 0:
        return -1, -1

    start0 = min(T, max(0, min_start))

    cc = np.where(d[start0:] <= contact_d_thresh)[0]
    if cc.size == 0:
        return -1, -1
    first_contact = start0 + int(cc[0])

    stable_mask = (d <= contact_d_thresh) & (np.abs(vn_smooth) <= vn_eps)
    start = min(T, max(start0, first_contact + max(0, int(min_after_contact))))
    stable_mask[:start] = False

    stable_idx = _first_stable_index(stable_mask, stable_len)
    return first_contact, stable_idx


def build_local_frame_from_demo(
    pos: np.ndarray,
    wall_normal: np.ndarray,
    plane_offset: float,
    contact_window: int,
    dist_eps: float,
    vn_eps: float,
    stable_len: int,
    min_after_contact: int,
    vn_smooth_win: int,
    min_start: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      origin, R_local_to_base, chosen_idx, stable_idx, first_contact, d, vn_raw, vn_smooth
    """
    z = unit(wall_normal)

    d, vn_raw = compute_d_and_vn(pos, z, plane_offset)
    vn_smooth = moving_average(vn_raw, vn_smooth_win)

    first_c, stable_idx = find_contact_and_stable(
        d=d,
        vn_smooth=vn_smooth,
        vn_eps=vn_eps,
        stable_len=stable_len,
        min_after_contact=min_after_contact,
        contact_d_thresh=dist_eps,
        min_start=min_start,
    )

    if stable_idx < 0:
        # dummy frame
        origin = pos[0].copy()
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, z)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y = unit(project_to_plane(tmp, z))
        x = unit(np.cross(y, z))
        y = unit(np.cross(z, x))
        R = np.column_stack([x, y, z])
        return origin, R, -1, stable_idx, first_c, d, vn_raw, vn_smooth

    chosen_idx = stable_idx
    origin = pos[chosen_idx].copy()

    # your original behavior: stable_len=100 for direction estimation
    y = stable_window_direction(pos, z, stable_idx=chosen_idx, stable_len=100, use_mean_diffs=True)

    if np.linalg.norm(y) < 1e-9:
        i1 = min(chosen_idx + contact_window, pos.shape[0] - 1)
        y = unit(project_to_plane(pos[i1] - pos[chosen_idx], z))

    if np.linalg.norm(y) < 1e-9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, z)) > 0.9:
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y = unit(project_to_plane(tmp, z))

    x = unit(np.cross(y, z))
    if np.linalg.norm(x) < 1e-9:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x = unit(np.cross(tmp, z))

    y = unit(np.cross(z, x))
    R_l2b = np.column_stack([x, y, z])
    return origin, R_l2b, chosen_idx, stable_idx, first_c, d, vn_raw, vn_smooth


def transform_demo_to_local(
    pos: np.ndarray,
    quat: np.ndarray,
    origin: np.ndarray,
    R_local_to_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    R_b2l = R_local_to_base.T
    pos_local = (R_b2l @ (pos - origin).T).T

    q_b2l = q_from_R(R_b2l)
    quat_local = np.empty_like(quat, dtype=np.float64)
    for i in range(quat.shape[0]):
        quat_local[i] = q_mul(q_b2l, q_normalize(quat[i]))
    return pos_local, quat_local


# ----------------------------
# contact segment from z_local (sign-crossing)
# ----------------------------
def find_contact_segment_from_zlocal_exact(z_local: np.ndarray, chosen_idx: int) -> tuple[int, int]:
    """
    start = chosen_idx
    end   = last t >= chosen_idx s.t. z[t] <= 0 and z[t+1] > 0  (leaves plane)
    If no crossing found, end = T-1.
    """
    z = np.asarray(z_local, dtype=np.float64).reshape(-1)
    T = int(z.shape[0])
    if chosen_idx < 0 or chosen_idx >= T:
        return -1, -1
    if T < 2:
        return chosen_idx, chosen_idx

    a = chosen_idx
    z0 = z[a : T - 1]
    z1 = z[a + 1 : T]

    cross = np.where((z0 <= 0.0) & (z1 > 0.0))[0]
    if cross.size == 0:
        return chosen_idx, T - 1

    last_t = a + int(cross[-1])
    return chosen_idx, last_t


# ----------------------------
# Event-based phase alignment resampling
# ----------------------------
def resample_piecewise_timewarp_3seg(
    Y: np.ndarray,              # (T,dim)
    chosen_idx: int,            # within [0, T)
    contact_end_idx: int,       # within [0, T)
    phase_grid: np.ndarray,     # (P,) in [0,1]
    a_ratio: float = 0.15,      # approach portion in [0,1]
    b_ratio: float = 0.15,      # depart portion in [0,1]
) -> np.ndarray:
    """
    3-segment piecewise time-warp onto phase_grid in [0,1].

    Segments (by indices):
      approach: [0 .. chosen_idx]           -> phase [0 .. a_ratio]
      work:     [chosen_idx .. contact_end] -> phase [a_ratio .. 1-b_ratio]
      depart:   [contact_end .. T-1]        -> phase [1-b_ratio .. 1]

    If segments are degenerate, it falls back gracefully.
    Uses linear interpolation, no extrapolation outside [0,1].
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    T, dim = Y.shape
    P = int(phase_grid.shape[0])
    out = np.zeros((P, dim), dtype=np.float64)

    # guard ratios
    a = float(np.clip(a_ratio, 0.0, 0.49))
    b = float(np.clip(b_ratio, 0.0, 0.49))
    if a + b >= 0.98:
        a, b = 0.15, 0.15

    # invalid chosen -> uniform time resample
    if chosen_idx < 0 or chosen_idx >= T:
        tt = np.linspace(0.0, 1.0, T)
        for k in range(dim):
            out[:, k] = np.interp(phase_grid, tt, Y[:, k])
        return out

    # clamp contact_end
    if contact_end_idx < 0:
        contact_end_idx = T - 1
    contact_end_idx = int(np.clip(contact_end_idx, 0, T - 1))

    # enforce order: chosen <= contact_end
    chosen_idx = int(chosen_idx)
    if contact_end_idx < chosen_idx:
        contact_end_idx = chosen_idx

    # build phase per sample (monotonic)
    phase = np.zeros((T,), dtype=np.float64)

    # ----- segment 1: approach -----
    if chosen_idx == 0:
        phase[0] = a
    else:
        # map t=0 -> 0, t=chosen -> a
        phase[: chosen_idx + 1] = np.linspace(0.0, a, chosen_idx + 1)

    # ----- segment 2: work -----
    if contact_end_idx == chosen_idx:
        phase[chosen_idx : contact_end_idx + 1] = a  # collapsed work
    else:
        # map chosen -> a, contact_end -> 1-b
        phase[chosen_idx : contact_end_idx + 1] = np.linspace(
            a, 1.0 - b, (contact_end_idx - chosen_idx) + 1
        )

    # ----- segment 3: depart -----
    if contact_end_idx == T - 1:
        phase[T - 1] = 1.0
    else:
        # map contact_end -> 1-b, last -> 1
        phase[contact_end_idx:] = np.linspace(
            1.0 - b, 1.0, (T - contact_end_idx)
        )

    # final monotonic safety
    for t in range(1, T):
        if phase[t] < phase[t - 1]:
            phase[t] = phase[t - 1]
    phase = np.clip(phase, 0.0, 1.0)

    # interpolate each dimension
    for k in range(dim):
        out[:, k] = np.interp(phase_grid, phase, Y[:, k])
    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", default="/home/sungboo/rb10_control/dataset/demo_20260122.npz")
    ap.add_argument("--out_npz", default="/home/sungboo/rb10_control/dataset/demo_20260122_final.npz")

    # plane / normal
    ap.add_argument("--plane_offset", type=float, default=-0.779)
    ap.add_argument("--wall_quat", type=float, nargs=4, default=[0.5, -0.5, -0.5, 0.5])

    # thresholds
    ap.add_argument("--dist_eps", type=float, default=0.003)
    ap.add_argument("--vn_eps", type=float, default=0.0005)
    ap.add_argument("--stable_len", type=int, default=20)
    ap.add_argument("--min_after_contact", type=int, default=0)
    ap.add_argument("--vn_smooth_win", type=int, default=9)
    ap.add_argument("--min_start", type=int, default=30)
    ap.add_argument("--contact_window", type=int, default=10)

    ap.add_argument("--crop_pad", type=int, default=20)
    ap.add_argument("--phase_pad", type=int, default=0.1)

    # phase alignment (hard-ish)
    ap.add_argument("--phase_len", type=int, default=200, help="Fixed length for phase-aligned sequences.")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_demo_indices", type=int, nargs="*", default=[0, 2, 36, 208])
    ap.add_argument("--img_dir", default="/home/sungboo/rb10_control/images/demo_20260122/preprocessing")
    args = ap.parse_args()

    in_path = Path(args.in_npz)
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    dnpz = np.load(in_path, allow_pickle=True)
    if "X" not in dnpz or "demo_ptr" not in dnpz:
        raise KeyError("NPZ must contain at least keys: X, demo_ptr")

    X = dnpz["X"]
    ptr = dnpz["demo_ptr"].astype(np.int64)
    D = ptr.shape[0] - 1
    names = dnpz["demo_names"] if "demo_names" in dnpz else np.array([f"demo_{i}" for i in range(D)], dtype=object)

    has_skill = "skill_id" in dnpz
    has_t = "t" in dnpz
    skill_full = dnpz["skill_id"] if has_skill else None
    t_full = dnpz["t"] if has_t else None

    # wall normal in base: R * +Z
    R_w = R_from_q(np.array(args.wall_quat, dtype=np.float64))
    wall_normal = unit(R_w @ np.array([0.0, 0.0, 1.0], dtype=np.float64))

    # full-length local output (same size as X)
    X_local = np.empty_like(X, dtype=np.float32)
    frame_origin = np.zeros((D, 3), dtype=np.float64)
    frame_R = np.zeros((D, 3, 3), dtype=np.float64)

    chosen_index = np.full((D,), -1, dtype=np.int32)
    stable_index = np.full((D,), -1, dtype=np.int32)
    first_contact_index = np.full((D,), -1, dtype=np.int32)

    # Cropped buffers
    X_crop_list: list[np.ndarray] = []
    Xlocal_crop_list: list[np.ndarray] = []
    skill_crop_list: list[np.ndarray] = []
    t_crop_list: list[np.ndarray] = []

    demo_names_crop: list[object] = []
    demo_ptr_crop = [0]

    kept_orig_demo_index: list[int] = []
    crop_s_list: list[int] = []
    crop_e_list: list[int] = []
    contact_start_list: list[int] = []
    contact_end_list: list[int] = []
    chosen_new_list: list[int] = []
    stable_new_list: list[int] = []
    first_contact_new_list: list[int] = []

    # For optional plots
    debug_series = {}
    plot_set = set(args.plot_demo_indices)

    for i in range(D):
        s, e = int(ptr[i]), int(ptr[i + 1])
        pos = X[s:e, 0:3].astype(np.float64)
        quat = X[s:e, 3:7].astype(np.float64)

        c_start = c_end = -1
        crop_s = crop_e = -1

        origin, R_l2b, chosen_idx, stable_idx, first_c, d_series, vn_raw, vn_smooth = build_local_frame_from_demo(
            pos=pos,
            wall_normal=wall_normal,
            plane_offset=args.plane_offset,
            contact_window=args.contact_window,
            dist_eps=args.dist_eps,
            vn_eps=args.vn_eps,
            stable_len=args.stable_len,
            min_after_contact=args.min_after_contact,
            vn_smooth_win=args.vn_smooth_win,
            min_start=args.min_start,
        )

        pos_l, quat_l = transform_demo_to_local(pos, quat, origin, R_l2b)

        # enforce z_local(chosen) == 0
        if chosen_idx >= 0:
            z0 = float(pos_l[chosen_idx, 2])
            pos_l[:, 2] -= z0
            pos_l[chosen_idx, 2] = 0.0

        # fill full-length outputs
        X_local[s:e, 0:3] = pos_l.astype(np.float32)
        X_local[s:e, 3:7] = quat_l.astype(np.float32)

        frame_origin[i] = origin
        frame_R[i] = R_l2b
        chosen_index[i] = chosen_idx
        stable_index[i] = stable_idx
        first_contact_index[i] = first_c

        # crop around contact segment
        if chosen_idx >= 0:
            zloc = pos_l[:, 2].astype(np.float64)
            c_start, c_end = find_contact_segment_from_zlocal_exact(z_local=zloc, chosen_idx=chosen_idx)

            if c_start >= 0 and c_end >= 0:
                pad = max(0, int(args.crop_pad))
                crop_s = max(0, c_start - pad)
                crop_e = min(zloc.shape[0], c_end + pad + 1)  # exclusive

                X_demo = X[s:e].astype(np.float32)
                Xl_demo = np.zeros_like(X_demo, dtype=np.float32)
                Xl_demo[:, 0:3] = pos_l.astype(np.float32)
                Xl_demo[:, 3:7] = quat_l.astype(np.float32)

                Xc = X_demo[crop_s:crop_e]
                Xlc = Xl_demo[crop_s:crop_e]

                if has_skill:
                    sc = np.asarray(skill_full[s:e])[crop_s:crop_e]
                    skill_crop_list.append(sc.astype(skill_full.dtype, copy=False))
                if has_t:
                    tc = np.asarray(t_full[s:e])[crop_s:crop_e]
                    t_crop_list.append(tc.astype(t_full.dtype, copy=False))

                chosen_new = chosen_idx - crop_s
                stable_new = stable_idx - crop_s if stable_idx >= 0 else -1
                first_new = first_c - crop_s if first_c >= 0 else -1
                c_start_new = c_start - crop_s
                c_end_new = c_end - crop_s

                X_crop_list.append(Xc)
                Xlocal_crop_list.append(Xlc)
                demo_names_crop.append(names[i])

                kept_orig_demo_index.append(i)
                crop_s_list.append(crop_s)
                crop_e_list.append(crop_e)
                contact_start_list.append(c_start_new)
                contact_end_list.append(c_end_new)
                chosen_new_list.append(chosen_new)
                stable_new_list.append(stable_new)
                first_contact_new_list.append(first_new)

                demo_ptr_crop.append(demo_ptr_crop[-1] + Xc.shape[0])

        if i in plot_set:
            debug_series[i] = {
                "d": d_series.astype(np.float32),
                "vn_raw": vn_raw.astype(np.float32),
                "vn_smooth": vn_smooth.astype(np.float32),
                "z_local": pos_l[:, 2].astype(np.float32),
                "chosen_idx": int(chosen_idx),
                "stable_idx": int(stable_idx),
                "first_contact": int(first_c),
                "name": str(names[i]),
                "c_start": int(c_start),
                "c_end": int(c_end),
                "crop_s": int(crop_s),
                "crop_e": int(crop_e),
            }

    # ----------------------------
    # save base outputs
    # ----------------------------
    out = {k: dnpz[k] for k in dnpz.files}

    out["X_local"] = X_local
    out["frame_origin"] = frame_origin
    out["frame_R_local_to_base"] = frame_R
    out["chosen_index"] = chosen_index
    out["stable_index"] = stable_index
    out["first_contact_index"] = first_contact_index

    out["wall_normal_base"] = wall_normal
    out["plane_offset"] = np.array([args.plane_offset], dtype=np.float64)
    out["wall_quat_xyzw"] = np.array(args.wall_quat, dtype=np.float64)

    out["dist_eps"] = np.array([args.dist_eps], dtype=np.float64)
    out["vn_eps"] = np.array([args.vn_eps], dtype=np.float64)
    out["stable_len"] = np.array([args.stable_len], dtype=np.int32)
    out["min_after_contact"] = np.array([args.min_after_contact], dtype=np.int32)
    out["vn_smooth_win"] = np.array([args.vn_smooth_win], dtype=np.int32)
    out["crop_pad"] = np.array([args.crop_pad], dtype=np.int32)

    # ----------------------------
    # finalize cropped arrays
    # ----------------------------
    if len(X_crop_list) == 0:
        print("[warn] No demos kept for cropped output (no chosen/stable found).")
        X_crop = np.zeros((0, X.shape[1]), dtype=np.float32)
        X_local_crop = np.zeros((0, X.shape[1]), dtype=np.float32)
        demo_ptr_crop_arr = np.array([0], dtype=np.int64)
        demo_names_crop_arr = np.array([], dtype=object)

        if has_skill:
            out["skill_id_crop"] = np.zeros((0,), dtype=skill_full.dtype)
        if has_t:
            out["t_crop"] = np.zeros((0,), dtype=t_full.dtype)

        out["X_crop"] = X_crop
        out["X_local_crop"] = X_local_crop
        out["demo_ptr_crop"] = demo_ptr_crop_arr
        out["demo_names_crop"] = demo_names_crop_arr

        # empty phase outputs too
        out["phase_grid"] = np.linspace(0.0, 1.0, int(args.phase_len), dtype=np.float64)
        out["X_phase_crop"] = np.zeros((0, 3), dtype=np.float32)
        out["W_phase_crop"] = np.zeros((0, 3), dtype=np.float32)
        out["demo_ptr_phase"] = np.array([0], dtype=np.int64)
        out["demo_names_phase"] = np.array([], dtype=object)

    else:
        X_crop = np.concatenate(X_crop_list, axis=0).astype(np.float32)
        X_local_crop = np.concatenate(Xlocal_crop_list, axis=0).astype(np.float32)
        demo_ptr_crop_arr = np.array(demo_ptr_crop, dtype=np.int64)
        demo_names_crop_arr = np.array(demo_names_crop, dtype=object)

        out["X_crop"] = X_crop
        out["X_local_crop"] = X_local_crop
        out["demo_ptr_crop"] = demo_ptr_crop_arr
        out["demo_names_crop"] = demo_names_crop_arr

        if has_skill:
            out["skill_id_crop"] = np.concatenate(skill_crop_list, axis=0).astype(skill_full.dtype, copy=False)
        if has_t:
            out["t_crop"] = np.concatenate(t_crop_list, axis=0).astype(t_full.dtype, copy=False)

        out["kept_orig_demo_index"] = np.array(kept_orig_demo_index, dtype=np.int32)
        out["crop_s"] = np.array(crop_s_list, dtype=np.int32)
        out["crop_e"] = np.array(crop_e_list, dtype=np.int32)
        out["contact_start_idx"] = np.array(contact_start_list, dtype=np.int32)
        out["contact_end_idx"] = np.array(contact_end_list, dtype=np.int32)
        out["chosen_index_crop"] = np.array(chosen_new_list, dtype=np.int32)
        out["stable_index_crop"] = np.array(stable_new_list, dtype=np.int32)
        out["first_contact_index_crop"] = np.array(first_contact_new_list, dtype=np.int32)

        out["contact_def"] = np.array(
            ["contact segment := [chosen_idx .. last idx where z_local <=0 then leaves to >0] (sign-crossing), then padded by crop_pad"],
            dtype=object,
        )

        # ----------------------------
        # NEW: Phase alignment + SO(3) log-map (MP-ready)
        # ----------------------------
        phase_len = int(args.phase_len)
        phase_grid = np.linspace(0.0, 1.0, phase_len, dtype=np.float64)
        out["phase_grid"] = phase_grid

        # build per-demo fixed-length sequences
        X_phase_list = []
        W_phase_list = []
        demo_ptr_phase = [0]
        demo_names_phase = []

        ptrc = demo_ptr_crop_arr
        Dc = int(ptrc.shape[0] - 1)

        # require these indices
        chosen_crop = out["chosen_index_crop"]
        cend_crop = out["contact_end_idx"]

        for di in range(Dc):
            a, b = int(ptrc[di]), int(ptrc[di + 1])
            if b <= a:
                continue

            Xlc_demo = X_local_crop[a:b]               # (T,dim) local
            pos_l = Xlc_demo[:, 0:3].astype(np.float64)
            quat_l = Xlc_demo[:, 3:7].astype(np.float64)

            ci = int(chosen_crop[di])  # chosen index within this cropped demo
            ce = int(cend_crop[di])    # contact end within this cropped demo

            # 1) quaternion sign unwrap (within demo)
            quat_l_u = unwrap_quat_signs(quat_l)

            # 2) rotvec (log map) relative to orientation at chosen
            w = quat_to_rotvec_rel(quat_l_u, ref_idx=ci)  # (T,3)

            # 3) event-based phase resample
            pos_rs = resample_piecewise_timewarp_3seg(
                pos_l, chosen_idx=ci, contact_end_idx=ce, phase_grid=phase_grid,
                a_ratio=args.phase_pad, b_ratio=args.phase_pad
            )
            w_rs = resample_piecewise_timewarp_3seg(
                w, chosen_idx=ci, contact_end_idx=ce, phase_grid=phase_grid,
                a_ratio=args.phase_pad, b_ratio=args.phase_pad
            )

            X_phase_list.append(pos_rs.astype(np.float32))
            W_phase_list.append(w_rs.astype(np.float32))
            demo_names_phase.append(demo_names_crop_arr[di])
            demo_ptr_phase.append(demo_ptr_phase[-1] + phase_len)

        out["X_phase_crop"] = np.concatenate(X_phase_list, axis=0).reshape(-1, 3).astype(np.float32)
        out["W_phase_crop"] = np.concatenate(W_phase_list, axis=0).reshape(-1, 3).astype(np.float32)
        out["demo_ptr_phase"] = np.array(demo_ptr_phase, dtype=np.int64)
        out["demo_names_phase"] = np.array(demo_names_phase, dtype=object)

    # save
    np.savez_compressed(out_path, **out)

    # ----------------------------
    # plots (optional)
    # ----------------------------
    if args.plot:
        import matplotlib.pyplot as plt

        def _set_axes_equal(ax):
            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()
            x_range = abs(x_limits[1] - x_limits[0]); x_mid = np.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0]); y_mid = np.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0]); z_mid = np.mean(z_limits)
            r = 0.5 * max([x_range, y_range, z_range])
            ax.set_xlim3d([x_mid - r, x_mid + r])
            ax.set_ylim3d([y_mid - r, y_mid + r])
            ax.set_zlim3d([z_mid - r, z_mid + r])

        def _draw_frame(ax, origin_b, R_l2b, L=0.05):
            o = origin_b.reshape(3,)
            ex = R_l2b[:, 0]; ey = R_l2b[:, 1]; ez = R_l2b[:, 2]
            x = o + L * ex
            y = o + L * ey
            z = o + L * ez
            ax.plot([o[0], x[0]], [o[1], x[1]], [o[2], x[2]])
            ax.plot([o[0], y[0]], [o[1], y[1]], [o[2], y[2]])
            ax.plot([o[0], z[0]], [o[1], z[1]], [o[2], z[2]])
            ax.text(x[0], x[1], x[2], "Lx")
            ax.text(y[0], y[1], y[2], "Ly")
            ax.text(z[0], z[1], z[2], "Lz")

        axis_len = 0.05

        for i in args.plot_demo_indices:
            if i < 0 or i >= D:
                continue
            s, e = int(ptr[i]), int(ptr[i + 1])
            pos_b = X[s:e, 0:3].astype(np.float64)
            pos_l = X_local[s:e, 0:3].astype(np.float64)

            origin_b = frame_origin[i].astype(np.float64)
            R_l2b = frame_R[i].astype(np.float64)

            ci = int(chosen_index[i])
            si = int(stable_index[i])
            fc = int(first_contact_index[i])

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(pos_b[:, 0], pos_b[:, 1], pos_b[:, 2])
            ax.scatter(pos_b[0, 0], pos_b[0, 1], pos_b[0, 2], s=25)
            ax.scatter(pos_b[-1, 0], pos_b[-1, 1], pos_b[-1, 2], s=25)
            if fc >= 0 and fc < pos_b.shape[0]:
                ax.scatter(pos_b[fc, 0], pos_b[fc, 1], pos_b[fc, 2], s=35)
            if si >= 0 and si < pos_b.shape[0]:
                ax.scatter(pos_b[si, 0], pos_b[si, 1], pos_b[si, 2], s=45)
            if ci >= 0 and ci < pos_b.shape[0]:
                ax.scatter(pos_b[ci, 0], pos_b[ci, 1], pos_b[ci, 2], s=55)

            _draw_frame(ax, origin_b, R_l2b, L=axis_len)
            ax.set_title(f"BASE traj + LOCAL frame | demo={i} {names[i]} | first={fc} stable={si} chosen={ci}")
            ax.set_xlabel("x_base [m]"); ax.set_ylabel("y_base [m]"); ax.set_zlabel("z_base [m]")
            _set_axes_equal(ax)
            plt.tight_layout()
            plt.savefig(str(img_dir / f"demo_{i:03d}_3d_base.png"))
            plt.close(fig)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(pos_l[:, 0], pos_l[:, 1], pos_l[:, 2])
            ax.scatter(pos_l[0, 0], pos_l[0, 1], pos_l[0, 2], s=25)
            ax.scatter(pos_l[-1, 0], pos_l[-1, 1], pos_l[-1, 2], s=25)

            L = axis_len
            ax.plot([0, L], [0, 0], [0, 0]); ax.text(L, 0, 0, "Lx")
            ax.plot([0, 0], [0, L], [0, 0]); ax.text(0, L, 0, "Ly")
            ax.plot([0, 0], [0, 0], [0, L]); ax.text(0, 0, L, "Lz")

            ax.set_title(f"LOCAL traj + LOCAL axes | demo={i} {names[i]} | first={fc} stable={si} chosen={ci}")
            ax.set_xlabel("x_local [m]"); ax.set_ylabel("y_local [m]"); ax.set_zlabel("z_local [m]")
            _set_axes_equal(ax)
            plt.tight_layout()
            plt.savefig(str(img_dir / f"demo_{i:03d}_3d_local.png"))
            plt.close(fig)

        for i in args.plot_demo_indices:
            if i not in debug_series:
                continue

            ds = debug_series[i]
            d_series = ds["d"]
            vn_raw = ds["vn_raw"]
            vn_sm = ds["vn_smooth"]
            zloc = ds["z_local"]

            T = d_series.shape[0]
            tt = np.arange(T)

            fc = int(ds["first_contact"])
            si = int(ds["stable_idx"])
            ci = int(ds["chosen_idx"])
            crop_s = int(ds.get("crop_s", -1))
            crop_e = int(ds.get("crop_e", -1))
            c_end = int(ds.get("c_end", -1))

            plt.figure(figsize=(11, 9))

            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(tt, d_series)
            ax1.axhline(args.dist_eps, linestyle=":")
            ax1.axhline(0.0, linestyle="--")
            ax1.set_ylabel("d(t) [m]  (need d<=dist_eps)")

            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(tt, vn_raw, label="vn_raw")
            ax2.plot(tt, vn_sm, label="vn_smooth")
            ax2.axhline(+args.vn_eps, linestyle=":")
            ax2.axhline(-args.vn_eps, linestyle=":")
            ax2.axhline(0.0, linestyle="--")
            ax2.set_ylabel("v_n [m/step]")
            ax2.legend(loc="upper right")

            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(tt, zloc)
            ax3.axhline(0.0, linestyle="--")
            ax3.set_xlabel("t [step]")
            ax3.set_ylabel("z_local(t) [m]")

            if ci >= 0:
                ax1.axvline(ci, linestyle="-")
                ax2.axvline(ci, linestyle="-")
                ax3.axvline(ci, linestyle="-")

            if crop_s >= 0:
                ax1.axvline(crop_s, linestyle=":")
                ax2.axvline(crop_s, linestyle=":")
                ax3.axvline(crop_s, linestyle=":")

            if crop_e >= 1:
                ax1.axvline(crop_e - 1, linestyle=":")
                ax2.axvline(crop_e - 1, linestyle=":")
                ax3.axvline(crop_e - 1, linestyle=":")

            if c_end >= 0:
                ax1.axvline(c_end, linestyle="-")
                ax2.axvline(c_end, linestyle="-")
                ax3.axvline(c_end, linestyle="-")

            ax1.set_title(f"{i} {ds['name']}  first={fc}  stable={si}  chosen={ci}")
            plt.tight_layout()
            plt.savefig(str(img_dir / f"demo_{i:03d}_d_vnraw_vnsm_zlocal.png"))
            plt.close()

        # phase-aligned sanity plot for a few cropped demos
        if ("X_phase_crop" in out) and (out["demo_ptr_phase"].shape[0] > 1):
            ptrp = out["demo_ptr_phase"].astype(np.int64)
            Xp = out["X_phase_crop"].astype(np.float64)  # (K*P,3)
            Wp = out["W_phase_crop"].astype(np.float64)  # (K*P,3)
            ph = out["phase_grid"].astype(np.float64)
            K = int(ptrp.shape[0] - 1)
            
            show = list(range(K))
            # z overlay
            plt.figure(figsize=(10, 4))
            for di in show:
                a, b = int(ptrp[di]), int(ptrp[di + 1])
                z = Xp[a:b, 2]
                plt.plot(ph, z)
            plt.axhline(0.0, linestyle="--")
            plt.title("phase-aligned z_local overlay")
            plt.xlabel("phase"); plt.ylabel("z_local [m]")
            plt.tight_layout()
            plt.savefig(str(img_dir / "phase_overlay_zlocal.png"))
            plt.close()

            # rotvec-norm overlay
            plt.figure(figsize=(10, 4))
            for di in show:
                a, b = int(ptrp[di]), int(ptrp[di + 1])
                wn = np.linalg.norm(Wp[a:b], axis=1)
                plt.plot(ph, wn)
            plt.title("phase-aligned ||rotvec|| overlay")
            plt.xlabel("phase"); plt.ylabel("||w|| [rad]")
            plt.tight_layout()
            plt.savefig(str(img_dir / "phase_overlay_rotvecnorm.png"))
            plt.close()

    # ----------------------------
    # summary
    # ----------------------------
    n_contact = int(np.sum(first_contact_index >= 0))
    n_stable = int(np.sum(stable_index >= 0))
    n_chosen = int(np.sum(chosen_index >= 0))
    n_kept = int(len(out.get("kept_orig_demo_index", [])))

    print(f"[saved] {out_path}")
    print(f"  demos: {D}")
    print(f"  contact_found: {n_contact}/{D}")
    print(f"  stable_found:  {n_stable}/{D}")
    print(f"  chosen_found:  {n_chosen}/{D}")
    print(f"  cropped_kept:  {n_kept}/{D}")
    if n_kept > 0:
        print(f"  cropped total steps: {int(out['demo_ptr_crop'][-1])}")
        if "X_phase_crop" in out:
            print(f"  phase_len: {int(out['phase_grid'].shape[0])}, phase demos: {int(out['demo_ptr_phase'].shape[0]-1)}")
            print(f"  X_phase_crop: {out['X_phase_crop'].shape}, W_phase_crop: {out['W_phase_crop'].shape}")
    print(f"  wall_normal(base): {wall_normal}")
    if args.plot:
        print(f"  figures: {img_dir}")


if __name__ == "__main__":
    main()
