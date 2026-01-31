#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skill library builder (phase-aligned) with:
- Cartesian DMP per demo (train-only by default; open_loop only for RMSE eval / plotting)
- ProMP per skill (prior)
- (Optional) Conditional ProMP per demo via multiple via points (contact start & end xyz)

Core change you asked:
- After skill_to_demo_idxs is built, we use ONE common function to build:
    Y_list / used / csce_list
- From that, we compute rep_top5 FIRST (per skill).
- Then DMP / ProMP both use the same prepared lists.
- RMSE is computed for BOTH models:
    * over ALL demos in the skill
    * over rep_top5 demos in the skill
  (and same idea for conditional ProMP, if enabled)

NPZ required:
  - X_phase_crop (N,3)
  - W_phase_crop (N,3)
  - demo_ptr_phase (D+1,)
  - demo_ptr_crop  (D+1,)
  - skill_id_crop (N_crop,) OR demo_skill_id_crop (D,)

NPZ optional (recommended):
  - contact_start_idx_crop (D,)  # crop-timeline index per demo
  - contact_end_idx_crop   (D,)
  - phase_grid (T,) (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

from movement_primitives.dmp import CartesianDMP
from movement_primitives.promp import ProMP

import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr


# try importing via_points helper (preferred)
try:
    from movement_primitives.promp import via_points as promp_via_points  # type: ignore
    _HAS_VIA_POINTS = True
except Exception:
    promp_via_points = None
    _HAS_VIA_POINTS = False


# -----------------------------
# skill id helpers
# -----------------------------
def infer_demo_skill_ids_from_skill_id_crop(skill_id_crop: np.ndarray, ptr_crop: np.ndarray) -> np.ndarray:
    skill_id_crop = np.asarray(skill_id_crop).astype(np.int64)
    ptr_crop = np.asarray(ptr_crop).astype(np.int64)
    D = int(ptr_crop.shape[0] - 1)

    if skill_id_crop.shape[0] != int(ptr_crop[-1]):
        raise ValueError(
            f"skill_id_crop length ({skill_id_crop.shape[0]}) must equal ptr_crop[-1] ({int(ptr_crop[-1])})."
        )

    demo_sid = np.full((D,), -1, dtype=np.int64)
    for i in range(D):
        s, e = int(ptr_crop[i]), int(ptr_crop[i + 1])
        seg = skill_id_crop[s:e]
        if seg.size == 0:
            continue
        vals, cnt = np.unique(seg, return_counts=True)
        demo_sid[i] = int(vals[np.argmax(cnt)])
    return demo_sid


def infer_demo_skill_ids(npz: dict, ptr_crop: np.ndarray) -> np.ndarray:
    D = int(ptr_crop.shape[0] - 1)

    if "demo_skill_id_crop" in npz:
        demo_sid = np.asarray(npz["demo_skill_id_crop"]).astype(np.int64)
        if demo_sid.shape[0] != D:
            raise ValueError(f"demo_skill_id_crop length {demo_sid.shape[0]} != D {D}")
        return demo_sid

    if "skill_id_crop" in npz:
        return infer_demo_skill_ids_from_skill_id_crop(npz["skill_id_crop"], ptr_crop)

    raise KeyError(
        "Missing skill ids. Need one of:\n"
        "  - demo_skill_id_crop (D,)\n"
        "  - skill_id_crop (N_crop,) aligned with demo_ptr_crop\n"
    )


# -----------------------------
# DMP utils: y6 <-> pqs
# -----------------------------
def y6_to_pqs(y6: np.ndarray) -> np.ndarray:
    """
    y6: (T,6) = [x,y,z, wx,wy,wz] where w* is rotvec (compact axis-angle)
    returns: PQS sequence expected by CartesianDMP / pytransform3d
    """
    y6 = np.asarray(y6, dtype=np.float64)
    if y6.ndim != 2 or y6.shape[1] != 6:
        raise ValueError(f"y6 must be (T,6), got {y6.shape}")

    transforms = []
    for i in range(y6.shape[0]):
        p = y6[i, 0:3]
        rvec = y6[i, 3:6]
        R = pr.matrix_from_compact_axis_angle(rvec)
        A2B = pt.transform_from(R=R, p=p)
        transforms.append(A2B)

    transforms = np.stack(transforms, axis=0)  # (T,4,4)
    pqs = ptr.pqs_from_transforms(transforms)
    return np.asarray(pqs, dtype=np.float64)


def pqs_to_y6(pqs: np.ndarray) -> np.ndarray:
    """
    pqs -> transforms -> (T,6) = xyz + rotvec
    """
    pqs = np.asarray(pqs, dtype=np.float64)
    transforms = ptr.transforms_from_pqs(pqs)  # (T,4,4)

    Tn = transforms.shape[0]
    y6 = np.zeros((Tn, 6), dtype=np.float64)
    for i in range(Tn):
        A2B = transforms[i]
        p = A2B[:3, 3]
        R = A2B[:3, :3]
        rvec = pr.compact_axis_angle_from_matrix(R)
        y6[i, 0:3] = p
        y6[i, 3:6] = rvec
    return y6


# -----------------------------
# Shared: resample
# -----------------------------
def resample(y: np.ndarray, Tnew: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    Told = y.shape[0]
    if Told == Tnew:
        return y
    x_old = np.linspace(0.0, 1.0, Told)
    x_new = np.linspace(0.0, 1.0, Tnew)
    out = np.zeros((Tnew, y.shape[1]), dtype=np.float64)
    for d in range(y.shape[1]):
        out[:, d] = np.interp(x_new, x_old, y[:, d])
    return out


# -----------------------------
# Cartesian DMP (train-only by default)
# -----------------------------
def fit_dmp(
    y: np.ndarray,
    dt: float,
    n_weights: int,
    do_open_loop: bool = False,
) -> dict:
    """
    Fit CartesianDMP to a single demo trajectory y (T,6) = xyz + rotvec.

    - do_open_loop=False => train-only
    - do_open_loop=True  => also returns y_hat + rmse (over 6D)
    """
    y = np.asarray(y, dtype=np.float64)
    Tn, D = y.shape
    if D != 6:
        raise ValueError(f"CartesianDMP expects y as (T,6)=xyz+rotvec, got {y.shape}")
    if Tn < 2:
        raise ValueError("Trajectory too short for DMP.")

    exec_time = (Tn - 1) * dt
    T_sec = np.linspace(0.0, exec_time, Tn, dtype=np.float64)

    Y_pqs = y6_to_pqs(y)
    dmp = CartesianDMP(execution_time=exec_time, dt=dt, n_weights_per_dim=n_weights)

    ok = False
    for call in (
        lambda: dmp.imitate(T_sec, Y_pqs),
        lambda: dmp.imitate(T=T_sec, Y=Y_pqs),
        lambda: dmp.imitate(T=T_sec, y=Y_pqs),
    ):
        try:
            call()
            ok = True
            break
        except TypeError:
            continue
    if not ok:
        raise RuntimeError("CartesianDMP imitate() signature mismatch. Paste the traceback.")

    y_hat = None
    rmse = None
    if do_open_loop:
        out = dmp.open_loop()
        if isinstance(out, tuple) and len(out) == 2:
            _, Y_hat_pqs = out
        else:
            Y_hat_pqs = out
        y_hat = pqs_to_y6(np.asarray(Y_hat_pqs, dtype=np.float64))
        TT = min(len(y_hat), len(y))
        rmse = float(np.sqrt(np.mean((y_hat[:TT] - y[:TT]) ** 2)))

    return {"model": dmp, "T": int(Tn), "exec_time": float(exec_time), "y_hat": y_hat, "rmse": rmse}


def dmp_open_loop_y6(dmp: CartesianDMP) -> np.ndarray:
    out = dmp.open_loop()
    if isinstance(out, tuple) and len(out) == 2:
        _, Y_hat_pqs = out
    else:
        Y_hat_pqs = out
    return pqs_to_y6(np.asarray(Y_hat_pqs, dtype=np.float64))


# -----------------------------
# ProMP
# -----------------------------
def fit_promp(Y_list: list[np.ndarray], n_basis: int) -> dict:
    if len(Y_list) == 0:
        raise ValueError("Empty Y_list.")

    lengths_raw = np.array([y.shape[0] for y in Y_list], dtype=np.int64)
    T_common = max(int(np.median(lengths_raw)), 2)
    D = int(Y_list[0].shape[1])

    Y_rs = [resample(np.asarray(y, dtype=np.float64), T_common) for y in Y_list]
    Ys = np.stack(Y_rs, axis=0)  # (N,T,D)
    N = int(Ys.shape[0])

    t = np.linspace(0.0, 1.0, T_common, dtype=np.float64)
    Ts = np.tile(t[None, :], (N, 1))  # (N,T)

    promp = ProMP(n_dims=D, n_weights_per_dim=n_basis)
    try:
        promp.imitate(Ts, Ys)
    except Exception as e:
        raise RuntimeError(
            "ProMP imitate() failed.\n"
            f"Ts shape={Ts.shape} (need (N,T)), Ys shape={Ys.shape} (need (N,T,D))\n"
            f"lengths_raw(first20)={lengths_raw[:20]}\n"
            f"Error: {repr(e)}"
        )

    y_mean = None
    y_var = None
    try:
        y_mean = np.asarray(promp.mean_trajectory(t), dtype=np.float64)  # (T,D)
        y_var = np.asarray(promp.var_trajectory(t), dtype=np.float64)    # (T,D) maybe
    except Exception:
        y_mean = None
        y_var = None

    return {"model": promp, "T_common": int(T_common), "t": t, "y_mean": y_mean, "y_var": y_var}


# -----------------------------
# Conditional ProMP (multiple via points)
# -----------------------------
def _get_contact_phase_indices_for_demo(
    data: dict,
    demo_i: int,
    ptrc: np.ndarray,
    phase_len: int,
    pre_steps: int,
    post_steps: int,
) -> tuple[int, int]:
    # case 1: per-demo crop indices exist
    if ("contact_start_idx_crop" in data) and ("contact_end_idx_crop" in data):
        cs_all = np.asarray(data["contact_start_idx_crop"], dtype=np.int64).reshape(-1)
        ce_all = np.asarray(data["contact_end_idx_crop"], dtype=np.int64).reshape(-1)

        if demo_i < cs_all.shape[0] and demo_i < ce_all.shape[0]:
            cs_crop = int(cs_all[demo_i])
            ce_crop = int(ce_all[demo_i])

            sc, ec = int(ptrc[demo_i]), int(ptrc[demo_i + 1])
            len_crop = int(ec - sc)
            if len_crop > 1:
                cs_crop = int(np.clip(cs_crop, 0, len_crop - 1))
                ce_crop = int(np.clip(ce_crop, 0, len_crop - 1))

                cs_idx = int(np.round(cs_crop / (len_crop - 1) * (phase_len - 1)))
                ce_idx = int(np.round(ce_crop / (len_crop - 1) * (phase_len - 1)))

                cs_idx = int(np.clip(cs_idx, 0, phase_len - 1))
                ce_idx = int(np.clip(ce_idx, 0, phase_len - 1))
                if ce_idx < cs_idx:
                    cs_idx, ce_idx = ce_idx, cs_idx
                return cs_idx, ce_idx

    # case 2: fallback
    cs_idx = int(np.clip(pre_steps, 0, phase_len - 1))
    ce_idx = int(np.clip(phase_len - 1 - post_steps, 0, phase_len - 1))
    if ce_idx < cs_idx:
        ce_idx = cs_idx
    return cs_idx, ce_idx


def _mean_at_time(promp: ProMP, t_scalar: float) -> np.ndarray:
    mu = np.asarray(promp.mean_trajectory(np.array([t_scalar], dtype=np.float64)), dtype=np.float64)
    return mu.reshape(-1)  # (D,)


def condition_promp_multiple_viapoints_xyz_start_end(
    promp: ProMP,
    phase_grid: np.ndarray,
    cs_idx: int,
    ce_idx: int,
    y_start_xyz: np.ndarray,
    y_end_xyz: np.ndarray,
    xyz_cov: float,
    loose_cov_other: float = 1e6,
) -> tuple[ProMP, dict]:
    """
    Multiple viapoints:
      - constrain xyz at (cs, ce) with small covariance
      - leave other dims (rotvec) almost unconstrained by using very large covariance
        and setting their targets to the prior mean at those times.
    """
    t = np.asarray(phase_grid, dtype=np.float64).reshape(-1)
    t_cs = float(t[cs_idx])
    t_ce = float(t[ce_idx])

    y_start_xyz = np.asarray(y_start_xyz, dtype=np.float64).reshape(3)
    y_end_xyz = np.asarray(y_end_xyz, dtype=np.float64).reshape(3)

    y_cs_full = _mean_at_time(promp, t_cs)
    y_ce_full = _mean_at_time(promp, t_ce)
    D = int(y_cs_full.shape[0])

    y_cs_full[:3] = y_start_xyz
    y_ce_full[:3] = y_end_xyz

    cov_cs = np.ones((D,), dtype=np.float64) * float(loose_cov_other)
    cov_ce = np.ones((D,), dtype=np.float64) * float(loose_cov_other)
    cov_cs[:3] = float(xyz_cov)
    cov_ce[:3] = float(xyz_cov)

    ts = np.array([t_cs, t_ce], dtype=np.float64)

    debug = {
        "method": None,
        "t_cs": t_cs,
        "t_ce": t_ce,
        "cs_idx": int(cs_idx),
        "ce_idx": int(ce_idx),
        "D": int(D),
        "used_via_points": False,
    }

    # ---- Try via_points first ----
    if _HAS_VIA_POINTS and (promp_via_points is not None):
        tries = []

        # Try A: y_cond (M,D), cov (M,D)
        tries.append((
            "via_points_y(M,D)_cov(M,D)",
            dict(
                y_cond=np.stack([y_cs_full, y_ce_full], axis=0),
                y_conditional_cov=np.stack([cov_cs, cov_ce], axis=0),
                ts=ts,
            )
        ))

        # Try B: flattened
        tries.append((
            "via_points_y(M*D)_cov(M*D)",
            dict(
                y_cond=np.concatenate([y_cs_full, y_ce_full], axis=0),
                y_conditional_cov=np.concatenate([cov_cs, cov_ce], axis=0),
                ts=ts,
            )
        ))

        for name, kwargs in tries:
            try:
                cpromp = promp_via_points(promp=promp, **kwargs)  # type: ignore
                debug["method"] = name
                debug["used_via_points"] = True
                return cpromp, debug
            except Exception:
                pass

    # ---- Fallback: chain condition_position twice ----
    try:
        cpromp = promp.condition_position(y_cs_full, y_cov=cov_cs, t=t_cs, t_max=1.0)
        cpromp = cpromp.condition_position(y_ce_full, y_cov=cov_ce, t=t_ce, t_max=1.0)
        debug["method"] = "fallback_chain_condition_position"
        debug["used_via_points"] = False
        return cpromp, debug
    except Exception as e:
        raise RuntimeError(f"Conditional ProMP failed both via_points and fallback chaining. Error: {repr(e)}")


# -----------------------------
# Metrics
# -----------------------------
def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def summarize_errors(errs: list[float]) -> dict:
    x = np.asarray(errs, dtype=np.float64)
    if x.size == 0:
        return {"mean": None, "median": None, "max": None, "per_demo": x}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
        "per_demo": x,
    }


def eval_rmse_vs_ref(Y_list: list[np.ndarray], Yref_list: list[np.ndarray]) -> dict:
    """
    Per-demo RMSE between Y_list[i] and Yref_list[i].
    Returns summaries for:
      - rmse_all: 6D
      - rmse_pos: xyz
      - rmse_rot: rotvec(3)
    """
    if len(Y_list) != len(Yref_list):
        raise ValueError("Y_list and Yref_list must have the same length")

    all_err, pos_err, rot_err = [], [], []
    for y, yref in zip(Y_list, Yref_list):
        y = np.asarray(y, dtype=np.float64)
        yref = np.asarray(yref, dtype=np.float64)

        if y.shape[0] != yref.shape[0]:
            y = resample(y, yref.shape[0])
        if y.shape[1] != yref.shape[1]:
            raise ValueError(f"Dim mismatch: y={y.shape}, yref={yref.shape}")

        all_err.append(_rmse(y[:, :6], yref[:, :6]))
        pos_err.append(_rmse(y[:, 0:3], yref[:, 0:3]))
        rot_err.append(_rmse(y[:, 3:6], yref[:, 3:6]))

    return {
        "n": int(len(Y_list)),
        "rmse_all": summarize_errors(all_err),
        "rmse_pos": summarize_errors(pos_err),
        "rmse_rot": summarize_errors(rot_err),
    }


def eval_promp_rmse_vs_mean(Y_list: list[np.ndarray], y_mean: np.ndarray) -> dict:
    y_mean = np.asarray(y_mean, dtype=np.float64)
    Yref_list = [y_mean for _ in range(len(Y_list))]
    return eval_rmse_vs_ref(Y_list=Y_list, Yref_list=Yref_list)


def eval_dmp_rmse_pos_rot_all(y_demo: np.ndarray, y_hat: np.ndarray) -> dict:
    y_demo = np.asarray(y_demo, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)

    if y_demo.shape[0] != y_hat.shape[0]:
        y_hat = resample(y_hat, y_demo.shape[0])

    rmse_all = _rmse(y_demo[:, :6], y_hat[:, :6])
    rmse_pos = _rmse(y_demo[:, 0:3], y_hat[:, 0:3])
    rmse_rot = _rmse(y_demo[:, 3:6], y_hat[:, 3:6])
    return {"rmse_all": rmse_all, "rmse_pos": rmse_pos, "rmse_rot": rmse_rot}


def contact_length_from_xyz(y_demo: np.ndarray, cs_idx: int, ce_idx: int) -> float:
    p0 = np.asarray(y_demo[cs_idx, 0:3], dtype=np.float64)
    p1 = np.asarray(y_demo[ce_idx, 0:3], dtype=np.float64)
    return float(np.linalg.norm(p1 - p0))


# -----------------------------
# Representative demo selection (top5)
# -----------------------------
def ee_stability_score(y_demo: np.ndarray, cs_idx: int, ce_idx: int) -> float:
    """
    Smaller is better.
    Heuristic stability of EE pose within contact interval:
      - position smoothness: std of velocity magnitude
      - rotation smoothness: std of rotvec velocity magnitude
    """
    y_demo = np.asarray(y_demo, dtype=np.float64)
    Tn = y_demo.shape[0]
    cs = int(np.clip(cs_idx, 0, Tn - 1))
    ce = int(np.clip(ce_idx, 0, Tn - 1))
    if ce <= cs:
        return 1e9

    seg = y_demo[cs:ce + 1]
    if seg.shape[0] < 3:
        return 1e9

    dpos = np.diff(seg[:, 0:3], axis=0)
    drot = np.diff(seg[:, 3:6], axis=0)

    vpos = np.linalg.norm(dpos, axis=1)
    vrot = np.linalg.norm(drot, axis=1)

    score = float(np.std(vpos) + 0.1 * np.mean(vpos) + np.std(vrot) + 0.1 * np.mean(vrot))
    return score


def pick_rep_top5_demos_for_skill(
    Y_list: list[np.ndarray],
    used: list[int],
    csce_list: list[tuple[int, int]],
    topk: int = 5,
    near_median_keep: int | None = None,
) -> dict:
    """
    Policy:
      1) contact length L_i
      2) keep demos near median length (default: keep min(n, max(2*topk, 10)))
      3) among kept demos, pick topk with best (lowest) ee_stability_score
    """
    assert len(Y_list) == len(used) == len(csce_list)
    n = len(used)
    if n == 0:
        return {"top5_demo_indices_phase": [], "median_contact_len": None, "table": []}

    lengths = np.zeros((n,), dtype=np.float64)
    stabs = np.zeros((n,), dtype=np.float64)
    for j, (y_demo, (cs, ce)) in enumerate(zip(Y_list, csce_list)):
        lengths[j] = contact_length_from_xyz(y_demo, cs, ce)
        stabs[j] = ee_stability_score(y_demo, cs, ce)

    med = float(np.median(lengths))
    dist = np.abs(lengths - med)

    if near_median_keep is None:
        near_median_keep = int(min(n, max(2 * topk, 10)))

    keep_idx = np.argsort(dist)[:near_median_keep]         # close-to-median first
    keep_idx = keep_idx[np.argsort(stabs[keep_idx])]       # stable first
    pick_idx = keep_idx[: min(topk, keep_idx.shape[0])]

    table = []
    for j in pick_idx.tolist():
        table.append({
            "demo_index_phase": int(used[j]),
            "contact_len": float(lengths[j]),
            "len_dist_to_median": float(dist[j]),
            "stability": float(stabs[j]),
        })

    return {
        "top5_demo_indices_phase": [int(used[j]) for j in pick_idx.tolist()],
        "median_contact_len": float(med),
        "table": table,
    }


# -----------------------------
# Plot util
# -----------------------------
def plot_overlay_6d(
    y: np.ndarray,
    overlays: list[tuple[str, np.ndarray]],
    title: str,
    out_png: Path,
    t: np.ndarray | None = None,
    vlines: dict[str, int] | None = None,
):
    y = np.asarray(y, dtype=np.float64)
    Tn = y.shape[0]
    tt = np.arange(Tn) if t is None else np.asarray(t).reshape(-1)

    labels = ["x", "y", "z", "wx", "wy", "wz"]
    plt.figure(figsize=(12, 14))

    for k in range(6):
        ax = plt.subplot(6, 1, k + 1)
        ax.plot(tt, y[:, k], label="demo")

        for name, yhat in overlays:
            yhat = np.asarray(yhat, dtype=np.float64)
            if yhat.shape[0] != Tn:
                yhat = resample(yhat, Tn)
            ax.plot(tt, yhat[:, k], label=name)

        if vlines:
            for nm, idx in vlines.items():
                if idx is None:
                    continue
                if 0 <= idx < Tn:
                    ax.axvline(tt[idx], linestyle="--")
                    ax.text(tt[idx], ax.get_ylim()[1], f" {nm}", rotation=90, va="top")

        ax.set_ylabel(labels[k])
        if k == 0:
            ax.set_title(title)
        if k == 5:
            ax.set_xlabel("phase_idx" if t is None else "t")

        if k == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_skill_means_1to8(
    skill_means: dict[int, np.ndarray],
    t: np.ndarray,
    out_png: Path,
    title: str,
):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    labels = ["x", "y", "z", "wx", "wy", "wz"]

    plt.figure(figsize=(14, 10))
    for k in range(6):
        ax = plt.subplot(6, 1, k + 1)
        for sid in range(1, 9):
            if sid not in skill_means:
                continue
            ym = np.asarray(skill_means[sid], dtype=np.float64)
            if ym.shape[0] != t.shape[0]:
                ym = resample(ym, t.shape[0])
            if ym.shape[1] < 6:
                continue
            ax.plot(t, ym[:, k], label=f"skill {sid}")
        ax.set_ylabel(labels[k])
        if k == 0:
            ax.set_title(title)
            ax.legend(loc="upper right", ncol=4, fontsize=9)
        if k == 5:
            ax.set_xlabel("t (0..1)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


# -----------------------------
# Filtering demos by index
# -----------------------------
def filter_demos_by_index(
    X_phase: np.ndarray,
    W_phase: np.ndarray,
    ptr_phase: np.ndarray,
    ptr_crop: np.ndarray,
    demo_skill: np.ndarray,
    drop_ids: list[int],
):
    drop_ids = sorted(set(int(i) for i in drop_ids))
    D = ptr_phase.shape[0] - 1
    keep = [i for i in range(D) if i not in drop_ids]
    if len(keep) == D:
        return X_phase, W_phase, ptr_phase, ptr_crop, demo_skill

    Xp_new, Wp_new = [], []
    ptrp_new = [0]
    ptrc_new = [0]
    demo_skill_new = []

    for i in keep:
        sp, ep = int(ptr_phase[i]), int(ptr_phase[i + 1])
        Xp_new.append(X_phase[sp:ep])
        Wp_new.append(W_phase[sp:ep])
        ptrp_new.append(ptrp_new[-1] + (ep - sp))

        sc, ec = int(ptr_crop[i]), int(ptr_crop[i + 1])
        ptrc_new.append(ptrc_new[-1] + (ec - sc))

        demo_skill_new.append(int(demo_skill[i]))

    Xp_new = np.concatenate(Xp_new, axis=0)
    Wp_new = np.concatenate(Wp_new, axis=0)

    return (
        Xp_new,
        Wp_new,
        np.asarray(ptrp_new, dtype=np.int64),
        np.asarray(ptrc_new, dtype=np.int64),
        np.asarray(demo_skill_new, dtype=np.int64),
    )


# -----------------------------
# Common builder: Y_list / used / csce_list (per skill)
# -----------------------------
def build_skill_demo_lists(
    sid: int,
    demo_idxs: list[int],
    Y_all: np.ndarray,
    ptrp: np.ndarray,
    ptrc: np.ndarray,
    data: dict,
    phase_len: int,
    min_len: int,
    pre_contact_steps: int,
    post_contact_steps: int,
) -> dict:
    """
    Returns:
      {
        "Y_list": list[np.ndarray],       # each (T,6)
        "used": list[int],               # demo index in phase timeline
        "csce_list": list[(cs,ce)],      # indices in phase timeline (0..phase_len-1)
      }
    """
    Y_list: list[np.ndarray] = []
    used: list[int] = []
    csce_list: list[tuple[int, int]] = []

    for i in demo_idxs:
        sp, ep = int(ptrp[i]), int(ptrp[i + 1])
        y = np.asarray(Y_all[sp:ep], dtype=np.float64)
        if y.shape[0] < min_len:
            continue

        cs_idx, ce_idx = _get_contact_phase_indices_for_demo(
            data=data,
            demo_i=i,
            ptrc=ptrc,
            phase_len=phase_len,
            pre_steps=int(pre_contact_steps),
            post_steps=int(post_contact_steps),
        )

        Y_list.append(y)
        used.append(i)
        csce_list.append((cs_idx, ce_idx))

    return {"Y_list": Y_list, "used": used, "csce_list": csce_list}


def subset_by_demo_indices(
    Y_list: list[np.ndarray],
    used: list[int],
    csce_list: list[tuple[int, int]],
    keep_demo_indices: list[int],
) -> tuple[list[np.ndarray], list[int], list[tuple[int, int]]]:
    keep_set = set(int(x) for x in keep_demo_indices)
    Y2, U2, C2 = [], [], []
    for y, u, c in zip(Y_list, used, csce_list):
        if int(u) in keep_set:
            Y2.append(y)
            U2.append(u)
            C2.append(c)
    return Y2, U2, C2


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/sungboo/rb10_control/dataset/demo_20260122_final.npz")
    ap.add_argument("--out", default="/home/sungboo/rb10_control/dataset/putty_skill_library.pkl")

    ap.add_argument("--model", choices=["dmp", "promp", "both"], default="both")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--min_len", type=int, default=10)

    # DMP
    ap.add_argument("--dmp_n_weights", type=int, default=50)

    # ProMP
    ap.add_argument("--promp_n_basis", type=int, default=25)
    ap.add_argument("--promp_min_demos", type=int, default=3)

    # Conditional ProMP
    ap.add_argument("--use_cond_promp", action="store_true")
    ap.add_argument("--cond_xyz_cov", type=float, default=1e-4)
    ap.add_argument("--pre_contact_steps", type=int, default=10)
    ap.add_argument("--post_contact_steps", type=int, default=10)

    # Plot
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_demo", type=int, default=0)
    ap.add_argument("--plot_dir", default="/home/sungboo/rb10_control/images/demo_20260122/skill_library")

    # Drop
    ap.add_argument("--drop_demos", type=int, nargs="*", default=[36, 57, 98, 202])

    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_path = Path(args.out)
    plot_dir = Path(args.plot_dir)

    data = np.load(npz_path, allow_pickle=True)

    # Required keys
    required = ["X_phase_crop", "W_phase_crop", "demo_ptr_phase", "demo_ptr_crop"]
    for k in required:
        if k not in data:
            raise KeyError(f"NPZ must contain key: {k}")
    if ("skill_id_crop" not in data) and ("demo_skill_id_crop" not in data):
        raise KeyError("NPZ must contain: skill_id_crop (preferred) or demo_skill_id_crop")

    Xp = np.asarray(data["X_phase_crop"], dtype=np.float64)      # (N,3)
    Wp = np.asarray(data["W_phase_crop"], dtype=np.float64)      # (N,3)
    ptrp = np.asarray(data["demo_ptr_phase"], dtype=np.int64)    # (D+1,)
    ptrc = np.asarray(data["demo_ptr_crop"], dtype=np.int64)     # (D+1,)

    Dc_phase = int(ptrp.shape[0] - 1)
    Dc_crop = int(ptrc.shape[0] - 1)
    if Dc_phase != Dc_crop:
        raise ValueError(f"Mismatch demos: D_phase={Dc_phase} != D_crop={Dc_crop}.")

    # Phase grid
    phase_grid = np.asarray(data["phase_grid"], dtype=np.float64) if "phase_grid" in data else None
    if phase_grid is None:
        if Dc_phase <= 0:
            raise ValueError("No phase demos found.")
        phase_len = int(ptrp[1] - ptrp[0])
        phase_grid = np.linspace(0.0, 1.0, phase_len, dtype=np.float64)
    else:
        phase_len = int(phase_grid.shape[0])

    demo_skill = infer_demo_skill_ids(data, ptrc)

    # Drop demos early
    if len(args.drop_demos) > 0:
        drop_set = sorted(set(int(i) for i in args.drop_demos))
        print(f"[info] Dropping demos: {drop_set}")
        bad = [i for i in drop_set if i < 0 or i >= Dc_phase]
        if bad:
            raise ValueError(f"--drop_demos out-of-range: {bad} (valid 0..{Dc_phase-1})")

        Xp, Wp, ptrp, ptrc, demo_skill = filter_demos_by_index(
            X_phase=Xp,
            W_phase=Wp,
            ptr_phase=ptrp,
            ptr_crop=ptrc,
            demo_skill=demo_skill,
            drop_ids=drop_set,
        )
        Dc_phase = int(ptrp.shape[0] - 1)

    # Build 6D trajectory per demo
    Y_all = np.concatenate([Xp, Wp], axis=1)  # (N,6)
    if Y_all.shape[1] != 6:
        raise ValueError(f"Expected 6D (xyz+rotvec), got {Y_all.shape}")

    # Group by skill
    skill_to_demo_idxs: dict[int, list[int]] = {}
    for i in range(Dc_phase):
        sid = int(demo_skill[i])
        if sid >= 0:
            skill_to_demo_idxs.setdefault(sid, []).append(i)

    # Storage
    library = {
        "rep_top5": {},
        "dmp": {},
        "promp": {},
        "cond_promp": {},
    }
    stats = {
        "rep_top5": {},
        "dmp": {},
        "promp": {},
        "cond_promp": {},
    }

    # Also keep per-skill prepared demo lists (so DMP and ProMP share them)
    prepared: dict[int, dict] = {}

    # --------------------
    # Step 1) Prepare lists + rep_top5 per skill
    # --------------------
    for sid, demo_idxs in skill_to_demo_idxs.items():
        pack = build_skill_demo_lists(
            sid=sid,
            demo_idxs=demo_idxs,
            Y_all=Y_all,
            ptrp=ptrp,
            ptrc=ptrc,
            data=data,
            phase_len=phase_len,
            min_len=int(args.min_len),
            pre_contact_steps=int(args.pre_contact_steps),
            post_contact_steps=int(args.post_contact_steps),
        )
        Y_list = pack["Y_list"]
        used = pack["used"]
        csce_list = pack["csce_list"]

        prepared[int(sid)] = pack

        rep = pick_rep_top5_demos_for_skill(
            Y_list=Y_list,
            used=used,
            csce_list=csce_list,
            topk=5,
            near_median_keep=None,
        )
        library["rep_top5"][int(sid)] = rep
        stats["rep_top5"][int(sid)] = {
            "n_demos_kept_for_modeling": int(len(used)),
            "median_contact_len": rep.get("median_contact_len", None),
            "top5_demo_indices_phase": rep.get("top5_demo_indices_phase", []),
        }

        # Print selection table (optional but handy)
        if len(rep.get("top5_demo_indices_phase", [])) > 0:
            print(f"\n[rep-top5] skill {sid} | median_len={rep['median_contact_len']:.6g}")
            for rank, row in enumerate(rep["table"], start=1):
                print(
                    f"  #{rank:02d} demo={row['demo_index_phase']:>4d} | "
                    f"L={row['contact_len']:.6g} | |L-med|={row['len_dist_to_median']:.6g} | "
                    f"stab={row['stability']:.6g}"
                )

    # --------------------
    # Step 2) DMP per demo (train-only), then compute RMSE (ALL + top5)
    # --------------------
    if args.model in ("dmp", "both"):
        for sid in sorted(prepared.keys()):
            Y_list = prepared[sid]["Y_list"]
            used = prepared[sid]["used"]

            if len(Y_list) == 0:
                continue

            dmp_items = []
            # train DMP per demo
            for y_demo, di in zip(Y_list, used):
                fit = fit_dmp(
                    y=y_demo,
                    dt=float(args.dt),
                    n_weights=int(args.dmp_n_weights),
                    do_open_loop=False,  # train-only
                )
                dmp_items.append({
                    "demo_index_phase": int(di),
                    "dmp": fit["model"],
                    "T": int(fit["T"]),
                    "exec_time": float(fit["exec_time"]),
                })

            if len(dmp_items) == 0:
                continue

            library["dmp"][sid] = {
                "skill_id": int(sid),
                "dmp_per_demo": dmp_items,
                "n_demos": int(len(dmp_items)),
                "dmp_n_weights": int(args.dmp_n_weights),
                "dt": float(args.dt),
                "train_only": True,
            }

            # eval RMSE for ALL demos
            dmp_map = {int(it["demo_index_phase"]): it["dmp"] for it in dmp_items}
            rmse_pos_all, rmse_rot_all, rmse_all_all = [], [], []
            for y_demo, di in zip(Y_list, used):
                dmp = dmp_map.get(int(di), None)
                if dmp is None:
                    continue
                y_hat = dmp_open_loop_y6(dmp)
                m = eval_dmp_rmse_pos_rot_all(y_demo=y_demo, y_hat=y_hat)
                rmse_pos_all.append(m["rmse_pos"])
                rmse_rot_all.append(m["rmse_rot"])
                rmse_all_all.append(m["rmse_all"])

            # eval RMSE for TOP5 demos
            top5 = library["rep_top5"].get(sid, {}).get("top5_demo_indices_phase", [])
            Y_top5, U_top5, _ = subset_by_demo_indices(
                Y_list=Y_list,
                used=used,
                csce_list=prepared[sid]["csce_list"],
                keep_demo_indices=top5,
            )
            rmse_pos_t5, rmse_rot_t5, rmse_all_t5 = [], [], []
            for y_demo, di in zip(Y_top5, U_top5):
                dmp = dmp_map.get(int(di), None)
                if dmp is None:
                    continue
                y_hat = dmp_open_loop_y6(dmp)
                m = eval_dmp_rmse_pos_rot_all(y_demo=y_demo, y_hat=y_hat)
                rmse_pos_t5.append(m["rmse_pos"])
                rmse_rot_t5.append(m["rmse_rot"])
                rmse_all_t5.append(m["rmse_all"])

            stats["dmp"][sid] = {
                "n_demos": int(len(dmp_items)),
                "rmse_all_demos": {
                    "rmse_pos": summarize_errors(rmse_pos_all),
                    "rmse_rot": summarize_errors(rmse_rot_all),
                    "rmse_all": summarize_errors(rmse_all_all),
                },
                "rmse_top5": {
                    "top5_demo_indices_phase": list(map(int, top5)),
                    "n_top5_found": int(len(rmse_pos_t5)),
                    "rmse_pos": summarize_errors(rmse_pos_t5),
                    "rmse_rot": summarize_errors(rmse_rot_t5),
                    "rmse_all": summarize_errors(rmse_all_t5),
                },
            }

    # --------------------
    # Step 3) ProMP per skill (prior), then compute RMSE (ALL + top5)
    # (+ optional conditional ProMP RMSE ALL + top5)
    # --------------------
    if args.model in ("promp", "both"):
        for sid in sorted(prepared.keys()):
            Y_list = prepared[sid]["Y_list"]
            used = prepared[sid]["used"]
            csce_list = prepared[sid]["csce_list"]

            if len(Y_list) < int(args.promp_min_demos):
                continue

            fit = fit_promp(Y_list=Y_list, n_basis=int(args.promp_n_basis))
            promp = fit["model"]
            y_mean = fit["y_mean"]

            # Prior RMSE (ALL demos)
            metrics_prior_all = None
            metrics_prior_top5 = None
            if y_mean is not None:
                metrics_prior_all = eval_promp_rmse_vs_mean(Y_list=Y_list, y_mean=y_mean)

                top5 = library["rep_top5"].get(sid, {}).get("top5_demo_indices_phase", [])
                Y_top5, U_top5, _ = subset_by_demo_indices(
                    Y_list=Y_list,
                    used=used,
                    csce_list=csce_list,
                    keep_demo_indices=top5,
                )
                if len(Y_top5) > 0:
                    metrics_prior_top5 = eval_promp_rmse_vs_mean(Y_list=Y_top5, y_mean=y_mean)

            library["promp"][sid] = {
                "skill_id": int(sid),
                "promp": promp,
                "T_common": int(fit["T_common"]),
                "t": fit["t"],
                "y_mean": y_mean,
                "y_var": fit["y_var"],
                "used_demo_indices_phase": np.array(used, dtype=np.int32),
                "metrics_prior_all": metrics_prior_all,
                "metrics_prior_top5": metrics_prior_top5,
            }

            stats["promp"][sid] = {
                "n_demos": int(len(used)),
                "T_common": int(fit["T_common"]),
                "rmse_all_demos": metrics_prior_all,
                "rmse_top5": {
                    "top5_demo_indices_phase": list(map(int, library["rep_top5"].get(sid, {}).get("top5_demo_indices_phase", []))),
                    "metrics": metrics_prior_top5,
                },
            }

            # ---- Conditional ProMP (optional): RMSE ALL + top5 ----
            if args.use_cond_promp:
                cond_list = []
                Yref_cond_list = []
                cond_debug = []
                lengths = []

                for y_demo, di, (cs_idx, ce_idx) in zip(Y_list, used, csce_list):
                    y_start_xyz = y_demo[cs_idx, 0:3]
                    y_end_xyz = y_demo[ce_idx, 0:3]
                    L = contact_length_from_xyz(y_demo, cs_idx, ce_idx)

                    cpromp, dbg = condition_promp_multiple_viapoints_xyz_start_end(
                        promp=promp,
                        phase_grid=phase_grid,
                        cs_idx=cs_idx,
                        ce_idx=ce_idx,
                        y_start_xyz=y_start_xyz,
                        y_end_xyz=y_end_xyz,
                        xyz_cov=float(args.cond_xyz_cov),
                    )
                    y_cmean = np.asarray(cpromp.mean_trajectory(phase_grid), dtype=np.float64)  # (T,6)

                    cond_list.append({
                        "demo_index_phase": int(di),
                        "cs_idx": int(cs_idx),
                        "ce_idx": int(ce_idx),
                        "t_cs": float(phase_grid[cs_idx]),
                        "t_ce": float(phase_grid[ce_idx]),
                        "y_start_xyz": np.asarray(y_start_xyz, dtype=np.float64),
                        "y_end_xyz": np.asarray(y_end_xyz, dtype=np.float64),
                        "contact_len": float(L),
                        "cond_xyz_cov": float(args.cond_xyz_cov),
                        "y_cmean": y_cmean,
                    })
                    Yref_cond_list.append(y_cmean)
                    lengths.append(float(L))
                    cond_debug.append(dbg)

                metrics_cond_all = eval_rmse_vs_ref(Y_list=Y_list, Yref_list=Yref_cond_list)

                # top5 conditional metrics
                top5 = library["rep_top5"].get(sid, {}).get("top5_demo_indices_phase", [])
                Y_top5, U_top5, _ = subset_by_demo_indices(
                    Y_list=Y_list,
                    used=used,
                    csce_list=csce_list,
                    keep_demo_indices=top5,
                )
                Yref_top5 = []
                if len(Y_top5) > 0:
                    # build map from demo_index_phase -> conditioned mean
                    cmean_map = {int(it["demo_index_phase"]): np.asarray(it["y_cmean"], dtype=np.float64) for it in cond_list}
                    for di in U_top5:
                        if int(di) in cmean_map:
                            Yref_top5.append(cmean_map[int(di)])
                    # align lengths
                    if len(Yref_top5) != len(Y_top5):
                        # keep only matched
                        Y_top5_matched, Yref_top5_matched = [], []
                        for y_demo, di in zip(Y_top5, U_top5):
                            if int(di) in cmean_map:
                                Y_top5_matched.append(y_demo)
                                Yref_top5_matched.append(cmean_map[int(di)])
                        Y_top5 = Y_top5_matched
                        Yref_top5 = Yref_top5_matched

                metrics_cond_top5 = None
                if len(Y_top5) > 0 and len(Yref_top5) == len(Y_top5):
                    metrics_cond_top5 = eval_rmse_vs_ref(Y_list=Y_top5, Yref_list=Yref_top5)

                library["cond_promp"][sid] = {
                    "skill_id": int(sid),
                    "prior_promp": promp,
                    "used_demo_indices_phase": np.array(used, dtype=np.int32),
                    "cond_xyz_cov": float(args.cond_xyz_cov),
                    "pre_contact_steps": int(args.pre_contact_steps),
                    "post_contact_steps": int(args.post_contact_steps),
                    "conditioned_per_demo": cond_list,
                    "metrics_cond_all": metrics_cond_all,
                    "metrics_cond_top5": metrics_cond_top5,
                    "conditioning_debug": cond_debug,
                }

                stats["cond_promp"][sid] = {
                    "n_demos": int(len(cond_list)),
                    "cond_xyz_cov": float(args.cond_xyz_cov),
                    "contact_len_mean": float(np.mean(np.asarray(lengths))) if len(lengths) else None,
                    "contact_len_median": float(np.median(np.asarray(lengths))) if len(lengths) else None,
                    "rmse_all_demos": metrics_cond_all,
                    "rmse_top5": {
                        "top5_demo_indices_phase": list(map(int, top5)),
                        "metrics": metrics_cond_top5,
                    },
                }

    # --------------------
    # Save payload
    # --------------------
    payload = {
        "source_npz": str(npz_path),
        "dt": float(args.dt),
        "min_len": int(args.min_len),
        "model_choice": args.model,
        "phase_len": int(phase_len),
        "library": library,
        "stats": stats,
        "skill_ids_present": sorted(list(skill_to_demo_idxs.keys())),
        "n_phase_demos": int(Dc_phase),
        "n_phase_steps": int(Y_all.shape[0]),
        "drop_demos": list(map(int, args.drop_demos)),
        "cond_promp_enabled": bool(args.use_cond_promp),
        "cond_xyz_cov": float(args.cond_xyz_cov),
        "has_via_points": bool(_HAS_VIA_POINTS),
        "dmp_train_only": True,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n[saved] {out_path}")
    print(f"  skills_total_in_data: {len(skill_to_demo_idxs)}")
    print(f"  rep_top5 skills built: {len(library['rep_top5'])}")
    if args.model in ("dmp", "both"):
        print(f"  DMP skills built: {len(library['dmp'])}")
    if args.model in ("promp", "both"):
        print(f"  ProMP skills built: {len(library['promp'])}")
    if args.use_cond_promp:
        print(f"  Cond-ProMP skills built: {len(library['cond_promp'])}")
        print(f"  via_points available: {_HAS_VIA_POINTS}")

    # --------------------
    # Print concise RMSE summaries (ALL + top5) for both models
    # --------------------
    def _print_rmse_block(tag: str, sid: int, block: dict):
        # block expected: {rmse_pos:{mean,median,max}, rmse_rot:{...}, rmse_all:{...}}
        rp = block.get("rmse_pos", {})
        rr = block.get("rmse_rot", {})
        ra = block.get("rmse_all", {})
        if rp.get("mean", None) is None:
            print(f"  skill {sid:>2d} | {tag} | (no data)")
            return
        print(
            f"  skill {sid:>2d} | {tag} | "
            f"RMSE_pos mean/med/max = {rp['mean']:.6g}/{rp['median']:.6g}/{rp['max']:.6g} | "
            f"RMSE_rot mean/med/max = {rr['mean']:.6g}/{rr['median']:.6g}/{rr['max']:.6g} | "
            f"RMSE_all mean/med/max = {ra['mean']:.6g}/{ra['median']:.6g}/{ra['max']:.6g}"
        )

    if args.model in ("dmp", "both") and len(stats["dmp"]) > 0:
        print("\n[DMP RMSE summary] (ALL demos + top5)")
        for sid in sorted(stats["dmp"].keys()):
            _print_rmse_block("all", sid, stats["dmp"][sid]["rmse_all_demos"])
            _print_rmse_block("top5", sid, stats["dmp"][sid]["rmse_top5"])

    if args.model in ("promp", "both") and len(stats["promp"]) > 0:
        print("\n[ProMP RMSE summary] (prior mean; ALL demos + top5)")
        for sid in sorted(stats["promp"].keys()):
            allm = stats["promp"][sid].get("rmse_all_demos", None)
            topm = stats["promp"][sid].get("rmse_top5", {}).get("metrics", None)

            # Convert to the same printing format as DMP block
            def _as_block(m):
                if m is None:
                    return {"rmse_pos": {"mean": None, "median": None, "max": None},
                            "rmse_rot": {"mean": None, "median": None, "max": None},
                            "rmse_all": {"mean": None, "median": None, "max": None}}
                return {"rmse_pos": m["rmse_pos"], "rmse_rot": m["rmse_rot"], "rmse_all": m["rmse_all"]}

            _print_rmse_block("all", sid, _as_block(allm))
            _print_rmse_block("top5", sid, _as_block(topm))

    if args.use_cond_promp and len(stats["cond_promp"]) > 0:
        print("\n[Cond-ProMP RMSE summary] (conditioned mean; ALL demos + top5)")
        for sid in sorted(stats["cond_promp"].keys()):
            allm = stats["cond_promp"][sid].get("rmse_all_demos", None)
            topm = stats["cond_promp"][sid].get("rmse_top5", {}).get("metrics", None)

            def _as_block(m):
                if m is None:
                    return {"rmse_pos": {"mean": None, "median": None, "max": None},
                            "rmse_rot": {"mean": None, "median": None, "max": None},
                            "rmse_all": {"mean": None, "median": None, "max": None}}
                return {"rmse_pos": m["rmse_pos"], "rmse_rot": m["rmse_rot"], "rmse_all": m["rmse_all"]}

            _print_rmse_block("all", sid, _as_block(allm))
            _print_rmse_block("top5", sid, _as_block(topm))

    # --------------------
    # Plotting (optional)
    # --------------------
    if args.plot:
        i = int(args.plot_demo)
        if not (0 <= i < Dc_phase):
            raise ValueError(f"--plot_demo must be within [0, {Dc_phase-1}]")

        sp, ep = int(ptrp[i]), int(ptrp[i + 1])
        y = np.asarray(Y_all[sp:ep], dtype=np.float64)
        sid = int(demo_skill[i])

        overlays: list[tuple[str, np.ndarray]] = []
        vlines = {"phase0": 0, "phase1": (y.shape[0] - 1)}
        title = f"phase-demo={i} skill={sid} | model={args.model}"

        # ProMP prior mean overlay
        if sid in library["promp"]:
            y_mean = library["promp"][sid].get("y_mean", None)
            if y_mean is not None:
                overlays.append(("ProMP prior mean", y_mean))

        # Conditional overlay
        if args.use_cond_promp and sid in library["cond_promp"]:
            centry = library["cond_promp"][sid]
            y_cmean = None
            cs_idx = None
            ce_idx = None
            for item in centry["conditioned_per_demo"]:
                if int(item["demo_index_phase"]) == i:
                    y_cmean = item["y_cmean"]
                    cs_idx = int(item["cs_idx"])
                    ce_idx = int(item["ce_idx"])
                    break
            if y_cmean is not None:
                overlays.append((f"cProMP mean (cov={args.cond_xyz_cov:g})", y_cmean))
                vlines = {"phase0": 0, "cs": cs_idx, "ce": ce_idx, "phase1": (y.shape[0] - 1)}

        # DMP overlay for this demo (if exists)
        if args.model in ("dmp", "both") and sid in library["dmp"]:
            dmp_for_demo = None
            for item in library["dmp"][sid]["dmp_per_demo"]:
                if int(item["demo_index_phase"]) == i:
                    dmp_for_demo = item["dmp"]
                    break
            if dmp_for_demo is not None:
                y_hat = dmp_open_loop_y6(dmp_for_demo)
                overlays.append((f"DMP open_loop (w={args.dmp_n_weights})", y_hat))

        out_png = plot_dir / f"verify_phase_demo_{i:03d}_skill_{sid}_overlay.png"
        plot_overlay_6d(
            y=y,
            overlays=overlays,
            title=title,
            out_png=out_png,
            t=phase_grid,
            vlines=vlines,
        )
        print(f"[plot] {out_png}")

        # skills 1..8 prior means
        skill_means_prior = {}
        for sid2 in range(1, 9):
            if sid2 in library["promp"]:
                ym = library["promp"][sid2].get("y_mean", None)
                if ym is not None:
                    skill_means_prior[int(sid2)] = ym
        if len(skill_means_prior) > 0:
            out_png2 = plot_dir / "skills_01_to_08_promp_mean_xyz_w.png"
            plot_skill_means_1to8(
                skill_means=skill_means_prior,
                t=phase_grid,
                out_png=out_png2,
                title="ProMP prior means for skills 1~8 (xyz + wx wy wz)",
            )
            print(f"[plot] {out_png2}")

        # skills 1..8 conditional means (rep demo per skill)
        if args.use_cond_promp:
            skill_means_cond = {}
            for sid2 in range(1, 9):
                if sid2 not in library["cond_promp"]:
                    continue
                items = library["cond_promp"][sid2]["conditioned_per_demo"]
                if len(items) == 0:
                    continue

                rep_top5 = library["rep_top5"].get(sid2, None)
                rep_demo_idx = None
                if rep_top5 and len(rep_top5.get("top5_demo_indices_phase", [])) > 0:
                    rep_demo_idx = int(rep_top5["top5_demo_indices_phase"][0])

                y_rep = None
                if rep_demo_idx is not None:
                    for it in items:
                        if int(it["demo_index_phase"]) == rep_demo_idx:
                            y_rep = np.asarray(it["y_cmean"], dtype=np.float64)
                            break

                if y_rep is None:
                    lens = np.array([float(it["contact_len"]) for it in items], dtype=np.float64)
                    med = float(np.median(lens))
                    rep_i = int(np.argmin(np.abs(lens - med)))
                    y_rep = np.asarray(items[rep_i]["y_cmean"], dtype=np.float64)

                skill_means_cond[int(sid2)] = y_rep

            if len(skill_means_cond) > 0:
                out_png3 = plot_dir / "skills_01_to_08_cond_promp_mean_xyz_w.png"
                plot_skill_means_1to8(
                    skill_means=skill_means_cond,
                    t=phase_grid,
                    out_png=out_png3,
                    title="Conditional ProMP means (rep demo per skill; via points at contact start/end xyz)",
                )
                print(f"[plot] {out_png3}")
            else:
                print("[plot] No conditional skill means available for 1..8.")


if __name__ == "__main__":
    main()
