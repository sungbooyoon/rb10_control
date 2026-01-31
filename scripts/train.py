#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(1) Train & save skill library (phase-aligned) into a PKL.

Models:
- dmp   : CartesianDMP per demo (train-only)
- promp : ProMP per skill (prior)
- cpromp: ProMP prior per skill + store conditioning metadata (contact start/end xyz indices)
         (conditioning + RMSE/plot is handled in (2) evaluator)

Key design:
- Build skill->(Y_list, used, csce_list, per_demo_meta) via a common function.
- Compute rep_top5 per skill FIRST (based on contact length median + stability).
- Then train selected model (dmp/promp/cpromp) using FULL demos (NOT top5 only).

NPZ required:
  - X_phase_crop (N,3)
  - W_phase_crop (N,3)
  - demo_ptr_phase (D+1,)
  - demo_ptr_crop  (D+1,)
  - skill_id_crop (N_crop,) OR demo_skill_id_crop (D,)

NPZ optional:
  - contact_start_idx_crop (D,)
  - contact_end_idx_crop   (D,)
  - phase_grid (T,)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import numpy as np

from movement_primitives.dmp import CartesianDMP
from movement_primitives.promp import ProMP

import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr


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
    return infer_demo_skill_ids_from_skill_id_crop(npz["skill_id_crop"], ptr_crop)

# -----------------------------
# Conditional ProMP (multiple via points)
# -----------------------------
try:
    from movement_primitives.promp import via_points as promp_via_points  # type: ignore
    _HAS_VIA_POINTS = True
except Exception:
    promp_via_points = None
    _HAS_VIA_POINTS = False


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
    Constrain xyz at contact start/end with small covariance.
    Other dims (rotvec) are left almost unconstrained by using huge covariance,
    and their targets are set to the prior mean at those times.
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
        "t_cs": t_cs,
        "t_ce": t_ce,
        "cs_idx": int(cs_idx),
        "ce_idx": int(ce_idx),
        "D": int(D),
        "used_via_points": False,
        "method": None,
    }

    # ---- Try via_points first ----
    if _HAS_VIA_POINTS and (promp_via_points is not None):
        tries = [
            (
                "via_points_y(M,D)_cov(M,D)",
                dict(
                    y_cond=np.stack([y_cs_full, y_ce_full], axis=0),
                    y_conditional_cov=np.stack([cov_cs, cov_ce], axis=0),
                    ts=ts,
                )
            ),
            (
                "via_points_y(M*D)_cov(M*D)",
                dict(
                    y_cond=np.concatenate([y_cs_full, y_ce_full], axis=0),
                    y_conditional_cov=np.concatenate([cov_cs, cov_ce], axis=0),
                    ts=ts,
                )
            ),
        ]
        for name, kwargs in tries:
            try:
                cpromp = promp_via_points(promp=promp, **kwargs)  # type: ignore
                debug["used_via_points"] = True
                debug["method"] = name
                return cpromp, debug
            except Exception:
                pass

    # ---- Fallback: chain condition_position twice ----
    cpromp = promp.condition_position(y_cs_full, y_cov=cov_cs, t=t_cs, t_max=1.0)
    cpromp = cpromp.condition_position(y_ce_full, y_cov=cov_ce, t=t_ce, t_max=1.0)
    debug["used_via_points"] = False
    debug["method"] = "fallback_chain_condition_position"
    return cpromp, debug


# -----------------------------
# Resample (for ProMP internal use)
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


def fit_dmp_train_only(y: np.ndarray, dt: float, n_weights: int) -> dict:
    """
    Train-only CartesianDMP on y (T,6). No open_loop here.
    Returns:
      {"model": dmp, "T": Tn, "exec_time": exec_time}
    """
    y = np.asarray(y, dtype=np.float64)
    Tn, D = y.shape
    if D != 6 or Tn < 2:
        raise ValueError(f"bad y shape for DMP: {y.shape}")

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

    return {"model": dmp, "T": int(Tn), "exec_time": float(exec_time)}


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
    promp.imitate(Ts, Ys)

    # convenience (optional)
    y_mean = None
    try:
        y_mean = np.asarray(promp.mean_trajectory(t), dtype=np.float64)
    except Exception:
        y_mean = None

    return {"model": promp, "T_common": int(T_common), "t": t, "y_mean": y_mean}


# -----------------------------
# Contact phase indices
# -----------------------------
def get_contact_phase_indices_for_demo(
    data: dict,
    demo_i: int,
    ptrc: np.ndarray,
    phase_len: int,
    pre_steps: int,
    post_steps: int,
) -> tuple[int, int]:
    """
    Returns (cs_idx, ce_idx) in PHASE timeline.
    If per-demo crop indices exist, map crop-index -> phase-index by linear scaling.
    Else fallback to (pre_steps, phase_len-1-post_steps).
    """
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

    cs_idx = int(np.clip(pre_steps, 0, phase_len - 1))
    ce_idx = int(np.clip(phase_len - 1 - post_steps, 0, phase_len - 1))
    if ce_idx < cs_idx:
        ce_idx = cs_idx
    return cs_idx, ce_idx


def contact_length_from_xyz(y_demo: np.ndarray, cs_idx: int, ce_idx: int) -> float:
    p0 = np.asarray(y_demo[cs_idx, 0:3], dtype=np.float64)
    p1 = np.asarray(y_demo[ce_idx, 0:3], dtype=np.float64)
    return float(np.linalg.norm(p1 - p0))


def ee_stability_score(y_demo: np.ndarray, cs_idx: int, ce_idx: int) -> float:
    """
    Smaller is better.
    Heuristic stability within contact interval:
      - std of |Δpos|
      - std of |Δrotvec|
      + small mean terms to avoid degenerate cases
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

    return float(np.std(vpos) + 0.1 * np.mean(vpos) + np.std(vrot) + 0.1 * np.mean(vrot))


def pick_rep_top5_for_skill(
    Y_list: list[np.ndarray],
    used: list[int],
    csce_list: list[tuple[int, int]],
    topk: int = 5,
    near_median_keep: int | None = None,
) -> dict:
    """
    Policy:
      1) contact length L_i
      2) keep demos closest to median length (default: keep min(n, max(2*topk, 10)))
      3) among kept, select topk with lowest stability score
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

    keep_idx = np.argsort(dist)[:near_median_keep]
    keep_idx = keep_idx[np.argsort(stabs[keep_idx])]
    pick_idx = keep_idx[: min(topk, keep_idx.shape[0])]

    table = []
    for j in pick_idx.tolist():
        table.append({
            "demo_index_phase": int(used[j]),
            "contact_len": float(lengths[j]),
            "len_dist_to_median": float(dist[j]),
            "stability": float(stabs[j]),
            "cs_idx": int(csce_list[j][0]),
            "ce_idx": int(csce_list[j][1]),
        })

    return {
        "top5_demo_indices_phase": [int(used[j]) for j in pick_idx.tolist()],
        "median_contact_len": float(med),
        "table": table,
    }


# -----------------------------
# Demo filtering (drop demos)
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
# Common builder: skill -> (Y_list, used, csce_list, per_demo_meta)
# -----------------------------
def build_skill_demo_cache(
    Y_all: np.ndarray,
    ptrp: np.ndarray,
    ptrc: np.ndarray,
    demo_skill: np.ndarray,
    data_npz: dict,
    phase_len: int,
    min_len: int,
    pre_contact_steps: int,
    post_contact_steps: int,
) -> dict[int, dict]:
    """
    Returns:
      cache[sid] = {
        "Y_list": [np.ndarray(T,6), ...]    # phase-aligned demo trajectories
        "used": [demo_index_phase, ...]
        "csce_list": [(cs,ce), ...]
        "per_demo": {demo_index_phase: {"cs":..., "ce":...}, ...}
      }
    """
    D = int(ptrp.shape[0] - 1)

    # group indices by skill
    skill_to_demo_idxs: dict[int, list[int]] = {}
    for i in range(D):
        sid = int(demo_skill[i])
        if sid >= 0:
            skill_to_demo_idxs.setdefault(sid, []).append(i)

    cache: dict[int, dict] = {}
    for sid, demo_idxs in skill_to_demo_idxs.items():
        Y_list: list[np.ndarray] = []
        used: list[int] = []
        csce_list: list[tuple[int, int]] = []
        per_demo: dict[int, dict] = {}

        for i in demo_idxs:
            sp, ep = int(ptrp[i]), int(ptrp[i + 1])
            y = np.asarray(Y_all[sp:ep], dtype=np.float64)
            if y.shape[0] < min_len:
                continue

            cs, ce = get_contact_phase_indices_for_demo(
                data=data_npz,
                demo_i=i,
                ptrc=ptrc,
                phase_len=phase_len,
                pre_steps=pre_contact_steps,
                post_steps=post_contact_steps,
            )

            Y_list.append(y)
            used.append(i)
            csce_list.append((cs, ce))
            per_demo[int(i)] = {"cs_idx": int(cs), "ce_idx": int(ce)}

        if len(used) > 0:
            cache[int(sid)] = {
                "Y_list": Y_list,
                "used": used,
                "csce_list": csce_list,
                "per_demo": per_demo,
            }

    return cache


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/sungboo/rb10_control/dataset/demo_20260122_final.npz")
    ap.add_argument("--out", required=True)

    ap.add_argument("--model", choices=["dmp", "promp", "cpromp"], required=True)

    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--min_len", type=int, default=10)

    # DMP
    ap.add_argument("--dmp_n_weights", type=int, default=50)

    # ProMP
    ap.add_argument("--promp_n_basis", type=int, default=25)
    ap.add_argument("--promp_min_demos", type=int, default=3)

    # Contact -> phase mapping fallback
    ap.add_argument("--pre_contact_steps", type=int, default=10)
    ap.add_argument("--post_contact_steps", type=int, default=10)

    # Conditional ProMP
    ap.add_argument("--cond_xyz_cov", type=float, default=1e-4)
    ap.add_argument("--cond_loose_cov_other", type=float, default=1e6)


    # Drop
    ap.add_argument("--drop_demos", type=int, nargs="*", default=[36, 57, 98, 202])

    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_path = Path(args.out)

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

    # phase grid / len
    if "phase_grid" in data:
        phase_grid = np.asarray(data["phase_grid"], dtype=np.float64).reshape(-1)
        phase_len = int(phase_grid.shape[0])
    else:
        if Dc_phase <= 0:
            raise ValueError("No demos found.")
        phase_len = int(ptrp[1] - ptrp[0])
        phase_grid = np.linspace(0.0, 1.0, phase_len, dtype=np.float64)

    demo_skill = infer_demo_skill_ids_from_skill_id_crop(data["skill_id_crop"], ptrc)

    # Drop demos early (keeps ptr structure consistent)
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

    # Build 6D trajectory per step
    Y_all = np.concatenate([Xp, Wp], axis=1)  # (N,6)
    if Y_all.shape[1] != 6:
        raise ValueError(f"Expected 6D (xyz+rotvec), got {Y_all.shape}")

    # ---------- Common cache (per skill demos) ----------
    cache = build_skill_demo_cache(
        Y_all=Y_all,
        ptrp=ptrp,
        ptrc=ptrc,
        demo_skill=demo_skill,
        data_npz=data,
        phase_len=phase_len,
        min_len=int(args.min_len),
        pre_contact_steps=int(args.pre_contact_steps),
        post_contact_steps=int(args.post_contact_steps),
    )

    # Compute rep_top5 FIRST for all skills (based on full demos, not subset)
    rep_top5: dict[int, dict] = {}
    for sid, item in cache.items():
        Y_list = item["Y_list"]
        used = item["used"]
        csce_list = item["csce_list"]

        rep = pick_rep_top5_for_skill(Y_list=Y_list, used=used, csce_list=csce_list, topk=5)
        rep_top5[int(sid)] = rep

        # optional print
        if len(rep.get("top5_demo_indices_phase", [])) > 0:
            print(f"\n[rep-top5] skill {sid} | median_len={rep['median_contact_len']:.6g}")
            for rank, row in enumerate(rep["table"], start=1):
                print(
                    f"  #{rank:02d} demo={row['demo_index_phase']:>4d} | "
                    f"L={row['contact_len']:.6g} | |L-med|={row['len_dist_to_median']:.6g} | "
                    f"stab={row['stability']:.6g} | cs={row['cs_idx']} ce={row['ce_idx']}"
                )

    # ---------- Train requested model ----------
    library = {
        "model_type": args.model,
        "dmp": {},
        "promp": {},
        "cpromp": {},
        "rep_top5": rep_top5,            # always stored
        "per_demo_contact_phase": {},    # always stored (for evaluator)
    }
    stats = {
        "dmp": {},
        "promp": {},
        "cpromp": {},
        "rep_top5": {},
    }

    # store per-demo contact indices per skill
    for sid, item in cache.items():
        library["per_demo_contact_phase"][int(sid)] = item["per_demo"]

    if args.model == "dmp":
        for sid, item in cache.items():
            Y_list = item["Y_list"]
            used = item["used"]

            dmp_list = []
            for y_demo, di in zip(Y_list, used):
                fit = fit_dmp_train_only(y=y_demo, dt=float(args.dt), n_weights=int(args.dmp_n_weights))
                dmp_list.append({
                    "demo_index_phase": int(di),
                    "dmp": fit["model"],
                    "T": int(fit["T"]),
                    "exec_time": float(fit["exec_time"]),
                    "n_weights": int(args.dmp_n_weights),
                })

            if len(dmp_list) > 0:
                library["dmp"][int(sid)] = dmp_list
                stats["dmp"][int(sid)] = {"n_demos": int(len(dmp_list))}

    elif args.model == "promp":
        for sid, item in cache.items():
            Y_list = item["Y_list"]
            used = item["used"]
            if len(Y_list) < int(args.promp_min_demos):
                continue

            fit = fit_promp(Y_list=Y_list, n_basis=int(args.promp_n_basis))
            library["promp"][int(sid)] = {
                "skill_id": int(sid),
                "promp": fit["model"],
                "T_common": int(fit["T_common"]),
                "t": fit["t"],
                "y_mean": fit["y_mean"],  # optional convenience
                "n_basis": int(args.promp_n_basis),
                "used_demo_indices_phase": np.array(used, dtype=np.int32),
            }
            stats["promp"][int(sid)] = {"n_demos": int(len(used)), "T_common": int(fit["T_common"])}

    elif args.model == "cpromp":
        for sid, item in cache.items():
            Y_list = item["Y_list"]
            used = item["used"]
            csce_list = item["csce_list"]
            if len(Y_list) < int(args.promp_min_demos):
                continue

            # 1) prior ProMP 학습
            fit = fit_promp(Y_list=Y_list, n_basis=int(args.promp_n_basis))
            prior = fit["model"]

            # 2) per-demo conditioning 결과를 여기서 "저장"
            conditioned_per_demo = []
            conditioning_debug = []

            for y_demo, di, (cs_idx, ce_idx) in zip(Y_list, used, csce_list):
                y_start_xyz = np.asarray(y_demo[cs_idx, 0:3], dtype=np.float64)
                y_end_xyz = np.asarray(y_demo[ce_idx, 0:3], dtype=np.float64)

                cpromp, dbg = condition_promp_multiple_viapoints_xyz_start_end(
                    promp=prior,
                    phase_grid=phase_grid,
                    cs_idx=int(cs_idx),
                    ce_idx=int(ce_idx),
                    y_start_xyz=y_start_xyz,
                    y_end_xyz=y_end_xyz,
                    xyz_cov=float(args.cond_xyz_cov),
                    loose_cov_other=float(args.cond_loose_cov_other),
                )

                y_cmean = np.asarray(cpromp.mean_trajectory(phase_grid), dtype=np.float64)  # (T,6)

                conditioned_per_demo.append({
                    "demo_index_phase": int(di),
                    "cs_idx": int(cs_idx),
                    "ce_idx": int(ce_idx),
                    "t_cs": float(phase_grid[int(cs_idx)]),
                    "t_ce": float(phase_grid[int(ce_idx)]),
                    "y_start_xyz": y_start_xyz,
                    "y_end_xyz": y_end_xyz,
                    "cond_xyz_cov": float(args.cond_xyz_cov),
                    "cond_loose_cov_other": float(args.cond_loose_cov_other),
                    "y_cmean": y_cmean,
                })
                conditioning_debug.append(dbg)

            library["cpromp"][int(sid)] = {
                "skill_id": int(sid),

                # prior
                "prior_promp": prior,
                "T_common": int(fit["T_common"]),
                "t": fit["t"],
                "y_mean": fit["y_mean"],  # optional convenience
                "n_basis": int(args.promp_n_basis),
                "used_demo_indices_phase": np.array(used, dtype=np.int32),

                # conditioning saved (eval.py가 지금 찾는 키)
                "pre_contact_steps": int(args.pre_contact_steps),
                "post_contact_steps": int(args.post_contact_steps),
                "cond_xyz_cov": float(args.cond_xyz_cov),
                "cond_loose_cov_other": float(args.cond_loose_cov_other),
                "conditioned_per_demo": conditioned_per_demo,
                "conditioning_debug": conditioning_debug,
            }

            stats["cpromp"][int(sid)] = {
                "n_demos": int(len(used)),
                "T_common": int(fit["T_common"]),
                "used_via_points": bool(conditioning_debug[0].get("used_via_points", False)) if conditioning_debug else False,
                "method": conditioning_debug[0].get("method", None) if conditioning_debug else None,
            }


    else:
        raise ValueError(f"Unknown model: {args.model}")

    # rep_top5 stats
    for sid, rep in rep_top5.items():
        stats["rep_top5"][int(sid)] = {
            "median_contact_len": rep.get("median_contact_len", None),
            "top5_demo_indices_phase": rep.get("top5_demo_indices_phase", []),
        }

    payload = {
        "source_npz": str(npz_path),
        "model_type": str(args.model),
        "dt": float(args.dt),
        "min_len": int(args.min_len),

        "phase_len": int(phase_len),
        "phase_grid": phase_grid,  # store for evaluator convenience

        "drop_demos": list(map(int, args.drop_demos)),

        "promp_n_basis": int(args.promp_n_basis),
        "promp_min_demos": int(args.promp_min_demos),
        "dmp_n_weights": int(args.dmp_n_weights),

        "pre_contact_steps": int(args.pre_contact_steps),
        "post_contact_steps": int(args.post_contact_steps),

        "library": library,
        "stats": stats,
        "skill_ids_present": sorted(list(cache.keys())),
        "n_phase_demos": int(Dc_phase),
        "n_phase_steps": int(Y_all.shape[0]),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n[saved] {out_path}")
    print(f"  model_type: {args.model}")
    print(f"  skills_in_cache: {len(cache)}")
    if args.model == "dmp":
        print(f"  dmp_skills_built: {len(library['dmp'])}")
    elif args.model == "promp":
        print(f"  promp_skills_built: {len(library['promp'])}")
    elif args.model == "cpromp":
        print(f"  cpromp_skills_built: {len(library['cpromp'])}")


if __name__ == "__main__":
    main()
