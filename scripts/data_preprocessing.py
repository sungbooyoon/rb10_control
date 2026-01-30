#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-demo contact-origin local frame transform + verification plots (NO z-flip).

Wall / plane:
- wall_normal is computed from wall_quat (xyzw):
    n = R(wall_quat) * [0,0,1]   (unit)
- plane is defined as:
    n·p = plane_offset
- signed distance:
    d(t) = n·p(t) - plane_offset
  (If n is unit, d has units of meters.)

Contact & origin selection:
- First, restrict to "contact side" half-space using:
    contact_mask = (d <= 0)   # you can change to >= if your convention differs
- Then find a "stable contact" index where BOTH hold:
    |d(t)| <= dist_eps        (near plane)
    |v_n(t)| <= vn_eps        (normal velocity stabilized)
  for stable_len consecutive steps,
  starting from first_contact + min_after_contact.

Local frame per demo:
- z-axis: n (wall normal in base/world)
- origin: projection of the chosen stable point onto the plane
- x-axis: estimated stroke tangent after chosen index, projected to plane
- y-axis: z × x

Outputs:
- out_npz with:
    X_local: (N,7) local poses (concat order preserved)
    frame_origin: (D,3)
    frame_R_local_to_base: (D,3,3)
    contact_index: (D,) chosen stable index (or first_contact fallback)
    first_contact_index: (D,)
    + all original arrays copied through
- Plots per selected demo (saved to img_dir):
    demo_XXX_d_vn_zlocal.png  : d(t), v_n(t), z_local(t) with markers
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


# ----------------------------
# Geometry helpers
# ----------------------------
def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def project_to_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Remove normal component."""
    return v - np.dot(v, n) * n


def _first_stable_index(mask: np.ndarray, stable_len: int) -> int:
    """
    mask: (T,) bool, True where condition holds
    return: first index i such that mask[i:i+stable_len] are all True, else -1
    """
    T = mask.shape[0]
    if stable_len <= 1:
        idx = np.where(mask)[0]
        return int(idx[0]) if idx.size else -1
    if T < stable_len:
        return -1
    s = np.convolve(mask.astype(np.int32), np.ones(stable_len, dtype=np.int32), mode="valid")
    hits = np.where(s == stable_len)[0]
    return int(hits[0]) if hits.size else -1


def find_contact_and_stable_index(
    pos: np.ndarray,
    n: np.ndarray,
    plane_offset: float,
    dist_eps: float,
    vn_eps: float,
    stable_len: int,
    min_after_contact: int,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """
    Returns:
      first_contact_idx, stable_idx, d(T,), vn(T,)
    """
    d = pos @ n - plane_offset  # (T,)

    cc = np.where(d <= 0.0)[0]
    if cc.size == 0:
        # no contact half-space
        vn = np.zeros((pos.shape[0],), dtype=np.float64)
        return -1, -1, d, vn

    first_contact = int(cc[0])

    dp = np.diff(pos, axis=0)
    vn = np.zeros((pos.shape[0],), dtype=np.float64)
    vn[1:] = dp @ n  # m/step

    stable_mask = (np.abs(d) <= dist_eps) & (np.abs(vn) <= vn_eps)

    start = min(pos.shape[0], first_contact + max(0, int(min_after_contact)))
    stable_mask[:start] = False

    stable_idx = _first_stable_index(stable_mask, stable_len)
    return first_contact, stable_idx, d, vn


def build_local_frame_from_demo(
    pos: np.ndarray,
    wall_normal: np.ndarray,
    plane_offset: float,
    contact_window: int,
    dist_eps: float,
    vn_eps: float,
    stable_len: int,
    min_after_contact: int,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray]:
    """
    Returns:
      origin (3,), R_local_to_base (3,3) columns [x y z],
      chosen_idx (int), first_contact_idx (int),
      d(T,), vn(T,)
    """
    n = unit(wall_normal)

    first_contact, stable_idx, d, vn = find_contact_and_stable_index(
        pos=pos,
        n=n,
        plane_offset=plane_offset,
        dist_eps=dist_eps,
        vn_eps=vn_eps,
        stable_len=stable_len,
        min_after_contact=min_after_contact,
    )

    if first_contact < 0:
        # fallback: no contact at all
        origin = pos[0].copy()
        z = n
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, z)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x = unit(np.cross(tmp, z))
        y = unit(np.cross(z, x))
        R = np.column_stack([x, y, z])
        return origin, R, -1, -1, d, vn

    chosen_idx = stable_idx if stable_idx >= 0 else first_contact

    # origin = projection of chosen point onto plane
    p_c = pos[chosen_idx].copy()
    dist = float(np.dot(p_c, n) - plane_offset)
    origin = p_c - dist * n

    # tangent direction after chosen index
    i1 = min(chosen_idx + contact_window, pos.shape[0] - 1)
    if i1 <= chosen_idx:
        dp2 = pos[min(chosen_idx + 1, pos.shape[0] - 1)] - pos[chosen_idx]
    else:
        dp2 = pos[i1] - pos[chosen_idx]

    tang = project_to_plane(dp2, n)
    if np.linalg.norm(tang) < 1e-9:
        j2 = min(chosen_idx + max(3, contact_window), pos.shape[0])
        seg = pos[chosen_idx:j2]
        diffs = np.diff(seg, axis=0)
        diffs = np.array([project_to_plane(v, n) for v in diffs])
        tang = diffs.sum(axis=0)

    if np.linalg.norm(tang) < 1e-9:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, n)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x = unit(np.cross(tmp, n))
    else:
        x = unit(tang)

    z = n
    y = unit(np.cross(z, x))
    x = unit(np.cross(y, z))
    R = np.column_stack([x, y, z])  # local->base

    return origin, R, chosen_idx, first_contact, d, vn


def transform_demo_to_local(pos: np.ndarray,
                            quat: np.ndarray,
                            origin: np.ndarray,
                            R_local_to_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Base/world -> per-demo local
    """
    R_b2l = R_local_to_base.T
    pos_local = (R_b2l @ (pos - origin).T).T

    q_b2l = q_from_R(R_b2l)
    quat_local = np.empty_like(quat, dtype=np.float64)
    for i in range(quat.shape[0]):
        quat_local[i] = q_mul(q_b2l, q_normalize(quat[i]))

    return pos_local, quat_local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", default="/home/sungboo/rb10_control/dataset/demo_20260122.npz")
    ap.add_argument("--out_npz", default="/home/sungboo/rb10_control/dataset/demo_20260122_local.npz")

    # plane params
    ap.add_argument("--plane_offset", type=float, default=-0.779)
    ap.add_argument("--wall_quat", type=float, nargs=4, default=[0.5, -0.5, -0.5, 0.5])  # xyzw

    # stability params
    ap.add_argument("--dist_eps", type=float, default=0.003, help="|d| <= dist_eps (m), default 3mm")
    ap.add_argument("--vn_eps", type=float, default=0.001, help="|v_n| <= vn_eps (m/step)")
    ap.add_argument("--stable_len", type=int, default=8, help="consecutive steps for stability")
    ap.add_argument("--min_after_contact", type=int, default=0, help="start stability search after first_contact + this")

    # local frame tangent estimation
    ap.add_argument("--contact_window", type=int, default=10)

    # plotting
    ap.add_argument("--plot_demo_indices", type=int, nargs="*", default=[0, 4, 8, 12])
    ap.add_argument("--img_dir", default="/home/sungboo/rb10_control/images/demo_20260122")

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

    # wall normal in base: R * +Z
    R_w = R_from_q(np.array(args.wall_quat, dtype=np.float64))
    wall_normal = unit(R_w @ np.array([0.0, 0.0, 1.0], dtype=np.float64))

    X_local = np.empty_like(X, dtype=np.float32)
    frame_origin = np.zeros((D, 3), dtype=np.float64)
    frame_R = np.zeros((D, 3, 3), dtype=np.float64)
    contact_index = np.full((D,), -1, dtype=np.int32)
    first_contact_index = np.full((D,), -1, dtype=np.int32)

    # store debug series for plotted demos only (save in separate dict)
    debug_series = {}

    for i in range(D):
        s, e = int(ptr[i]), int(ptr[i + 1])
        pos = X[s:e, 0:3].astype(np.float64)
        quat = X[s:e, 3:7].astype(np.float64)

        origin, R_l2b, chosen_idx, first_c, d_series, vn_series = build_local_frame_from_demo(
            pos=pos,
            wall_normal=wall_normal,
            plane_offset=args.plane_offset,
            contact_window=args.contact_window,
            dist_eps=args.dist_eps,
            vn_eps=args.vn_eps,
            stable_len=args.stable_len,
            min_after_contact=args.min_after_contact
        )

        pos_l, quat_l = transform_demo_to_local(pos, quat, origin, R_l2b)

        X_local[s:e, 0:3] = pos_l.astype(np.float32)
        X_local[s:e, 3:7] = quat_l.astype(np.float32)

        frame_origin[i] = origin
        frame_R[i] = R_l2b
        contact_index[i] = chosen_idx
        first_contact_index[i] = first_c

        if i in set(args.plot_demo_indices):
            debug_series[i] = {
                "d": d_series.astype(np.float32),
                "vn": vn_series.astype(np.float32),
                "z_local": pos_l[:, 2].astype(np.float32),
                "chosen_idx": int(chosen_idx),
                "first_contact": int(first_c),
                "name": str(names[i]),
            }

    # save npz (keep original keys too)
    out = {k: dnpz[k] for k in dnpz.files}
    out["X_local"] = X_local
    out["frame_origin"] = frame_origin
    out["frame_R_local_to_base"] = frame_R
    out["contact_index"] = contact_index
    out["first_contact_index"] = first_contact_index
    out["wall_normal_base"] = wall_normal
    out["plane_offset"] = np.array([args.plane_offset], dtype=np.float64)
    out["wall_quat_xyzw"] = np.array(args.wall_quat, dtype=np.float64)

    np.savez_compressed(out_path, **out)

    # ---- Plot d(t), vn(t), z_local(t) for selected demos
    import matplotlib.pyplot as plt

    for i in args.plot_demo_indices:
        if i not in debug_series:
            continue

        ds = debug_series[i]
        d_series = ds["d"]
        vn_series = ds["vn"]
        zloc = ds["z_local"]
        T = d_series.shape[0]
        tt = np.arange(T)

        fc = ds["first_contact"]
        ci = ds["chosen_idx"]

        # 1 figure, 3 panels (d, vn, z_local)
        plt.figure(figsize=(10, 8))

        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(tt, d_series)
        ax1.axhline(+args.dist_eps, linestyle=":")
        ax1.axhline(-args.dist_eps, linestyle=":")
        ax1.axhline(0.0, linestyle="--")
        if fc >= 0:
            ax1.axvline(fc, linestyle="--")
        if ci >= 0:
            ax1.axvline(ci, linestyle="-")
        ax1.set_ylabel("d(t) [m]")
        ax1.set_title(f"{i} {ds['name']}  first_contact={fc}  chosen={ci}")

        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(tt, vn_series)
        ax2.axhline(+args.vn_eps, linestyle=":")
        ax2.axhline(-args.vn_eps, linestyle=":")
        ax2.axhline(0.0, linestyle="--")
        if fc >= 0:
            ax2.axvline(fc, linestyle="--")
        if ci >= 0:
            ax2.axvline(ci, linestyle="-")
        ax2.set_ylabel("v_n(t) [m/step]")

        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(tt, zloc)
        ax3.axhline(0.0, linestyle="--")
        if fc >= 0:
            ax3.axvline(fc, linestyle="--")
        if ci >= 0:
            ax3.axvline(ci, linestyle="-")
        ax3.set_xlabel("t [step]")
        ax3.set_ylabel("z_local(t) [m]")

        plt.tight_layout()
        plt.savefig(str(img_dir / f"demo_{i:03d}_d_vn_zlocal.png"))
        plt.close()

    # ---- Summary stats
    n_contact_side = int(np.sum(first_contact_index >= 0))
    n_stable = int(np.sum(contact_index >= 0))
    print(f"[saved] {out_path}")
    print(f"  demos: {D}")
    print(f"  contact_side_found: {n_contact_side}/{D}")
    print(f"  chosen_found: {n_stable}/{D}  (stable window if possible; else fallback to first_contact)")
    print(f"  wall_normal(base): {wall_normal}")
    print(f"  figures: {img_dir}/demo_XXX_d_vn_zlocal.png")


if __name__ == "__main__":
    main()