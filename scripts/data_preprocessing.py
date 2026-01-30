#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-demo contact-origin local frame transform + verification plots (NO z-flip)

Hard constraints (no CLI options):
  - require_contact = ON  : stable search happens only after d(t) <= contact_d_thresh
  - require_stable  = ON  : if stable not found -> chosen_idx = -1 and we do NOT build a meaningful frame

Plane:
  d(t) = n·p(t) - plane_offset    (signed distance to plane along normal n)

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
  origin = pos[chosen_idx]    (stable point itself; then we force z_local(chosen)=0)

Saves:
  out_npz with X_local + frame metadata + indices
  per-demo plot images (d, v_n raw/smooth, z_local)
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
) -> tuple[int, int]:
    """
    HARD require_contact:
      - first_contact exists only if d <= contact_d_thresh appears.
    HARD require_stable:
      - stable exists only if (d<=thresh) & (|vn_smooth|<=vn_eps) holds for stable_len consecutive steps.
    Returns: (first_contact_idx, stable_idx) (both -1 if not found)
    """
    cc = np.where(d <= contact_d_thresh)[0]
    if cc.size == 0:
        return -1, -1
    first_contact = int(cc[0])

    stable_mask = (d <= contact_d_thresh) & (np.abs(vn_smooth) <= vn_eps)
    start = min(d.shape[0], first_contact + max(0, int(min_after_contact)))
    stable_mask[:start] = False

    stable_idx = _first_stable_index(stable_mask, stable_len)
    return first_contact, stable_idx


def build_local_frame_from_demo(
    pos: np.ndarray,
    wall_normal: np.ndarray,
    plane_offset: float,
    contact_window: int,
    dist_eps: float,  # still used as contact_d_thresh (tolerance)
    vn_eps: float,
    stable_len: int,
    min_after_contact: int,
    vn_smooth_win: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    HARD behavior:
      - require_contact = ON  (need d <= dist_eps)
      - require_stable  = ON  (need stable window; else chosen=-1 and frame is dummy)

    Contact threshold:
      d <= dist_eps  (NOTE: dist_eps is now "contact tolerance" not "|d|<=...")

    Axes:
      z = wall normal (from wall_quat)
      y = progression direction in stable window, projected to plane
      x = y × z (right-handed), then re-orthogonalize y = z × x

    Origin:
      origin = pos[chosen_idx]  (chosen_idx == stable_idx)
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
    )

    # HARD require_stable
    if stable_idx < 0:
        # dummy frame (keeps pipeline alive, but chosen stays -1)
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

    # Y axis from stable window
    y = stable_window_direction(pos, z, stable_idx=chosen_idx, stable_len=100, use_mean_diffs=True)

    if np.linalg.norm(y) < 1e-9:
        # fallback: short lookahead
        i1 = min(chosen_idx + contact_window, pos.shape[0] - 1)
        y = unit(project_to_plane(pos[i1] - pos[chosen_idx], z))

    if np.linalg.norm(y) < 1e-9:
        # last resort: arbitrary in-plane direction
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, z)) > 0.9:
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y = unit(project_to_plane(tmp, z))

    # X axis (right-handed)
    x = unit(np.cross(y, z))
    if np.linalg.norm(x) < 1e-9:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x = unit(np.cross(tmp, z))

    # re-orthogonalize y
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", default="/home/sungboo/rb10_control/dataset/demo_20260122.npz")
    ap.add_argument("--out_npz", default="/home/sungboo/rb10_control/dataset/demo_20260122_local.npz")

    # plane / normal
    ap.add_argument("--plane_offset", type=float, default=-0.779)
    ap.add_argument("--wall_quat", type=float, nargs=4, default=[0.5, -0.5, -0.5, 0.5])

    # thresholds
    ap.add_argument("--dist_eps", type=float, default=0.003, help="Contact tolerance for d(t): require d(t) <= dist_eps (m).")
    ap.add_argument("--vn_eps", type=float, default=0.0003, help="Normal velocity stability threshold (m/step).")
    ap.add_argument("--stable_len", type=int, default=20, help="Consecutive steps for stability.")
    ap.add_argument("--min_after_contact", type=int, default=0, help="Start stable search after first_contact + this.")

    ap.add_argument("--vn_smooth_win", type=int, default=9, help="Moving average window for vn (odd recommended).")
    ap.add_argument("--contact_window", type=int, default=10)

    # plotting
    ap.add_argument("--plot", action="store_true")
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

    chosen_index = np.full((D,), -1, dtype=np.int32)
    stable_index = np.full((D,), -1, dtype=np.int32)
    first_contact_index = np.full((D,), -1, dtype=np.int32)

    debug_series = {}
    plot_set = set(args.plot_demo_indices)

    for i in range(D):
        s, e = int(ptr[i]), int(ptr[i + 1])
        pos = X[s:e, 0:3].astype(np.float64)
        quat = X[s:e, 3:7].astype(np.float64)

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
        )

        pos_l, quat_l = transform_demo_to_local(pos, quat, origin, R_l2b)

        # enforce z_local(chosen) == 0 when chosen exists
        if chosen_idx >= 0:
            pos_l[:, 2] -= pos_l[chosen_idx, 2]

        X_local[s:e, 0:3] = pos_l.astype(np.float32)
        X_local[s:e, 3:7] = quat_l.astype(np.float32)

        frame_origin[i] = origin
        frame_R[i] = R_l2b
        chosen_index[i] = chosen_idx
        stable_index[i] = stable_idx
        first_contact_index[i] = first_c

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
            }

    # save npz
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
    np.savez_compressed(out_path, **out)

    
    if args.plot:
        # ---- 3D plots: per-demo trajectory + local frame axes
        # Save: demo_XXX_3d_base.png, demo_XXX_3d_local.png
        import matplotlib.pyplot as plt

        def _set_axes_equal(ax):
            x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
            x_range = abs(x_limits[1] - x_limits[0]); x_mid = np.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0]); y_mid = np.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0]); z_mid = np.mean(z_limits)
            r = 0.5 * max([x_range, y_range, z_range])
            ax.set_xlim3d([x_mid - r, x_mid + r])
            ax.set_ylim3d([y_mid - r, y_mid + r])
            ax.set_zlim3d([z_mid - r, z_mid + r])

        def _draw_frame(ax, origin_b, R_l2b, L=0.05):
            """
            Draw local axes in BASE coords.
            R_l2b columns are local unit axes expressed in base frame.
            """
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

        axis_len = 0.05  # meters (원하면 args로 빼도 됨)

        for i in args.plot_demo_indices:
            if i < 0 or i >= D:
                continue

            s, e = int(ptr[i]), int(ptr[i + 1])
            pos_b = X[s:e, 0:3].astype(np.float64)         # base trajectory
            pos_l = X_local[s:e, 0:3].astype(np.float64)   # local trajectory (already computed)

            origin_b = frame_origin[i].astype(np.float64)
            R_l2b = frame_R[i].astype(np.float64)

            ci = int(chosen_index[i])
            si = int(stable_index[i])
            fc = int(first_contact_index[i])

            # (A) BASE view: base traj + local frame axes
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(pos_b[:, 0], pos_b[:, 1], pos_b[:, 2])
            ax.scatter(pos_b[0, 0], pos_b[0, 1], pos_b[0, 2], s=25)      # start
            ax.scatter(pos_b[-1, 0], pos_b[-1, 1], pos_b[-1, 2], s=25)   # end

            # mark contact/stable/chosen points if valid
            if fc >= 0 and fc < pos_b.shape[0]:
                ax.scatter(pos_b[fc, 0], pos_b[fc, 1], pos_b[fc, 2], s=35)
            if si >= 0 and si < pos_b.shape[0]:
                ax.scatter(pos_b[si, 0], pos_b[si, 1], pos_b[si, 2], s=45)
            if ci >= 0 and ci < pos_b.shape[0]:
                ax.scatter(pos_b[ci, 0], pos_b[ci, 1], pos_b[ci, 2], s=55)

            _draw_frame(ax, origin_b, R_l2b, L=axis_len)
            ax.set_title(f"BASE traj + LOCAL frame (R_l2b) | demo={i} {names[i]} | first={fc} stable={si} chosen={ci}")
            ax.set_xlabel("x_base [m]"); ax.set_ylabel("y_base [m]"); ax.set_zlabel("z_base [m]")
            _set_axes_equal(ax)
            plt.tight_layout()
            plt.savefig(str(img_dir / f"demo_{i:03d}_3d_base.png"))
            plt.close(fig)

            # (B) LOCAL view: local traj + identity local axes
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(pos_l[:, 0], pos_l[:, 1], pos_l[:, 2])
            ax.scatter(pos_l[0, 0], pos_l[0, 1], pos_l[0, 2], s=25)
            ax.scatter(pos_l[-1, 0], pos_l[-1, 1], pos_l[-1, 2], s=25)

            # axes at origin in local coords
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

        # ---- plots
        import matplotlib.pyplot as plt

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

            fc = ds["first_contact"]
            si = ds["stable_idx"]
            ci = ds["chosen_idx"]

            plt.figure(figsize=(11, 9))

            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(tt, d_series)
            ax1.axhline(args.dist_eps, linestyle=":")
            ax1.axhline(0.0, linestyle="--")
            if fc >= 0:
                ax1.axvline(fc, linestyle="--")
            if si >= 0:
                ax1.axvline(si, linestyle="-")
            ax1.set_ylabel("d(t) [m]  (need d<=dist_eps)")

            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(tt, vn_raw, label="vn_raw")
            ax2.plot(tt, vn_sm, label="vn_smooth")
            ax2.axhline(+args.vn_eps, linestyle=":")
            ax2.axhline(-args.vn_eps, linestyle=":")
            ax2.axhline(0.0, linestyle="--")
            if fc >= 0:
                ax2.axvline(fc, linestyle="--")
            if si >= 0:
                ax2.axvline(si, linestyle="-")
            ax2.set_ylabel("v_n [m/step]")
            ax2.legend(loc="upper right")

            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(tt, zloc)
            ax3.axhline(0.0, linestyle="--")
            if fc >= 0:
                ax3.axvline(fc, linestyle="--")
            if si >= 0:
                ax3.axvline(si, linestyle="-")
            ax3.set_xlabel("t [step]")
            ax3.set_ylabel("z_local(t) [m]")

            ax1.set_title(f"{i} {ds['name']}  first={fc}  stable={si}  chosen={ci}")
            plt.tight_layout()
            plt.savefig(str(img_dir / f"demo_{i:03d}_d_vnraw_vnsm_zlocal.png"))
            plt.close()

        n_contact = int(np.sum(first_contact_index >= 0))
        n_stable = int(np.sum(stable_index >= 0))
        n_chosen = int(np.sum(chosen_index >= 0))

        print(f"[saved] {out_path}")
        print(f"  demos: {D}")
        print(f"  contact_found: {n_contact}/{D}   (require d<=dist_eps)")
        print(f"  stable_found:  {n_stable}/{D}   (require d<=dist_eps AND |vn_smooth|<=vn_eps for stable_len)")
        print(f"  chosen_found:  {n_chosen}/{D}   (require_stable=ON)")
        print(f"  wall_normal(base): {wall_normal}")
        print(f"  figures: {img_dir}/demo_XXX_d_vnraw_vnsm_zlocal.png")


if __name__ == "__main__":
    main()