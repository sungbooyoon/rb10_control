#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate & plot skill library from a saved PKL.

- Loads payload.pkl produced by training script (1).
- Loads source NPZ (default: payload["source_npz"]) and applies same drop_demos
  so that demo indices match the training-time "phase demo indices".
- Computes RMSE for each skill:
    * DMP: demo open_loop vs demo (overall + rep-top5)
    * ProMP prior: mean vs demo (overall + rep-top5)
    * cProMP: per-demo conditioned mean vs demo (overall + rep-top5)
- Optional plotting:
    * overlay plot for a chosen demo index: demo vs ProMP mean vs cProMP mean vs DMP open_loop
    * (NEW) skill 1~8: xyz+wxwywz mean in ONE figure (ProMP prior)
    * (NEW) skill 1~8: xyz+wxwywz mean in ONE figure (cProMP representative demo per skill)

Assumptions:
- NPZ contains X_phase_crop (N,3), W_phase_crop (N,3),
  demo_ptr_phase (D+1,), demo_ptr_crop (D+1,),
  and either skill_id_crop (N_crop,) OR demo_skill_id_crop (D,).
- Payload.pkl contains library with keys among ["dmp","promp","cpromp"] and (optionally) "rep_top5".
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from movement_primitives.dmp import CartesianDMP
from movement_primitives.promp import ProMP

import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr


# -----------------------------
# skill id helpers (same logic)
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
# resample / rmse
# -----------------------------
def resample(y: np.ndarray, Tnew: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    Told = int(y.shape[0])
    if Told == Tnew:
        return y
    x_old = np.linspace(0.0, 1.0, Told)
    x_new = np.linspace(0.0, 1.0, Tnew)
    out = np.zeros((Tnew, y.shape[1]), dtype=np.float64)
    for d in range(y.shape[1]):
        out[:, d] = np.interp(x_new, x_old, y[:, d])
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rmse_pos_rot_all(y: np.ndarray, yref: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    yref = np.asarray(yref, dtype=np.float64)
    if y.shape[0] != yref.shape[0]:
        yref = resample(yref, y.shape[0])
    return {
        "rmse_pos": _rmse(y[:, 0:3], yref[:, 0:3]),
        "rmse_rot": _rmse(y[:, 3:6], yref[:, 3:6]),
        "rmse_all": _rmse(y[:, 0:6], yref[:, 0:6]),
    }


def summarize(vals: List[float]) -> Dict[str, float]:
    x = np.asarray(vals, dtype=np.float64)
    if x.size == 0:
        return {"mean": np.nan, "median": np.nan, "max": np.nan}
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "max": float(np.max(x))}


# -----------------------------
# DMP helpers (open_loop -> y6)
# -----------------------------
def y6_to_pqs(y6: np.ndarray) -> np.ndarray:
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
    transforms = np.stack(transforms, axis=0)
    return np.asarray(ptr.pqs_from_transforms(transforms), dtype=np.float64)


def pqs_to_y6(pqs: np.ndarray) -> np.ndarray:
    pqs = np.asarray(pqs, dtype=np.float64)
    transforms = ptr.transforms_from_pqs(pqs)
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


def dmp_open_loop_y6(dmp: CartesianDMP) -> np.ndarray:
    out = dmp.open_loop()
    if isinstance(out, tuple) and len(out) == 2:
        _, Y_hat_pqs = out
    else:
        Y_hat_pqs = out
    return pqs_to_y6(np.asarray(Y_hat_pqs, dtype=np.float64))


# -----------------------------
# drop demos (must match training)
# -----------------------------
def filter_demos_by_index(
    X_phase: np.ndarray,
    W_phase: np.ndarray,
    ptr_phase: np.ndarray,
    ptr_crop: np.ndarray,
    demo_skill: np.ndarray,
    drop_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drop_ids = sorted(set(int(i) for i in drop_ids))
    D = int(ptr_phase.shape[0] - 1)
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
# rep-top5 selection (same heuristic)
# -----------------------------
def _get_contact_phase_indices_for_demo(
    data: dict,
    demo_i: int,
    ptrc: np.ndarray,
    phase_len: int,
    pre_steps: int,
    post_steps: int,
) -> Tuple[int, int]:
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
    y_demo = np.asarray(y_demo, dtype=np.float64)
    Tn = y_demo.shape[0]
    cs = int(np.clip(cs_idx, 0, Tn - 1))
    ce = int(np.clip(ce_idx, 0, Tn - 1))
    if ce <= cs:
        return 1e9
    seg = y_demo[cs : ce + 1]
    if seg.shape[0] < 3:
        return 1e9

    dpos = np.diff(seg[:, 0:3], axis=0)
    drot = np.diff(seg[:, 3:6], axis=0)
    vpos = np.linalg.norm(dpos, axis=1)
    vrot = np.linalg.norm(drot, axis=1)
    return float(np.std(vpos) + 0.1 * np.mean(vpos) + np.std(vrot) + 0.1 * np.mean(vrot))


def pick_rep_top5_demos_for_skill(
    Y_list: List[np.ndarray],
    used: List[int],
    csce_list: List[Tuple[int, int]],
    topk: int = 5,
    near_median_keep: Optional[int] = None,
) -> Dict:
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
    overlays: List[Tuple[str, np.ndarray]],
    title: str,
    out_png: Path,
    t: Optional[np.ndarray] = None,
    vlines: Optional[Dict[str, int]] = None,
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
    skill_means: Dict[int, np.ndarray],
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
# Core: build Y_list/used/csce_list
# -----------------------------
def build_skill_lists(
    sid: int,
    demo_idxs: List[int],
    Y_all: np.ndarray,
    ptrp: np.ndarray,
    npz_data: dict,
    ptrc: np.ndarray,
    phase_len: int,
    min_len: int,
    pre_contact_steps: int,
    post_contact_steps: int,
) -> Tuple[List[np.ndarray], List[int], List[Tuple[int, int]]]:
    Y_list: List[np.ndarray] = []
    used: List[int] = []
    csce_list: List[Tuple[int, int]] = []

    for i in demo_idxs:
        sp, ep = int(ptrp[i]), int(ptrp[i + 1])
        y = np.asarray(Y_all[sp:ep], dtype=np.float64)
        if y.shape[0] < min_len:
            continue
        Y_list.append(y)
        used.append(i)

        cs_idx, ce_idx = _get_contact_phase_indices_for_demo(
            data=npz_data,
            demo_i=i,
            ptrc=ptrc,
            phase_len=phase_len,
            pre_steps=pre_contact_steps,
            post_steps=post_contact_steps,
        )
        csce_list.append((cs_idx, ce_idx))

    return Y_list, used, csce_list


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="Skill library PKL from training script (1)")
    ap.add_argument("--npz", default=None, help="Override NPZ path (default: payload['source_npz'])")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_demo", type=int, default=0, help="phase-demo index (after dropping) to overlay plot")
    ap.add_argument("--plot_dir", default=None, help="Override plot output dir")
    ap.add_argument("--min_len", type=int, default=None, help="Override min_len (default: payload['min_len'])")
    ap.add_argument("--pre_contact_steps", type=int, default=None, help="Override (default: payload or 10)")
    ap.add_argument("--post_contact_steps", type=int, default=None, help="Override (default: payload or 10)")
    ap.add_argument("--drop_demos", type=int, nargs="*", default=None, help="Override drop_demos (default: payload['drop_demos'])")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    lib = payload.get("library", {})
    source_npz = Path(args.npz) if args.npz is not None else Path(payload["source_npz"])

    min_len = int(args.min_len) if args.min_len is not None else int(payload.get("min_len", 10))
    drop_demos = args.drop_demos if args.drop_demos is not None else list(map(int, payload.get("drop_demos", [])))

    # contact steps override (may not exist in payload)
    pre_steps = int(args.pre_contact_steps) if args.pre_contact_steps is not None else int(payload.get("pre_contact_steps", 10))
    post_steps = int(args.post_contact_steps) if args.post_contact_steps is not None else int(payload.get("post_contact_steps", 10))

    # plot dir
    plot_dir = Path(args.plot_dir) if args.plot_dir is not None else Path(payload.get("plot_dir", "./skill_library_eval_plots"))
    if args.plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # ---- load npz ----
    npz = np.load(source_npz, allow_pickle=True)
    required = ["X_phase_crop", "W_phase_crop", "demo_ptr_phase", "demo_ptr_crop"]
    for k in required:
        if k not in npz:
            raise KeyError(f"NPZ must contain key: {k}")

    Xp = np.asarray(npz["X_phase_crop"], dtype=np.float64)
    Wp = np.asarray(npz["W_phase_crop"], dtype=np.float64)
    ptrp = np.asarray(npz["demo_ptr_phase"], dtype=np.int64)
    ptrc = np.asarray(npz["demo_ptr_crop"], dtype=np.int64)
    D_phase = int(ptrp.shape[0] - 1)

    # phase_grid
    if "phase_grid" in npz:
        phase_grid = np.asarray(npz["phase_grid"], dtype=np.float64).reshape(-1)
        phase_len = int(phase_grid.shape[0])
    else:
        phase_len = int(ptrp[1] - ptrp[0]) if D_phase > 0 else int(payload.get("phase_len", 0))
        if phase_len <= 0:
            raise ValueError("Cannot infer phase_len.")
        phase_grid = np.linspace(0.0, 1.0, phase_len, dtype=np.float64)

    demo_skill = infer_demo_skill_ids(npz, ptrc)

    # ---- apply same dropping as training ----
    if drop_demos:
        bad = [i for i in drop_demos if i < 0 or i >= D_phase]
        if bad:
            raise ValueError(f"drop_demos out-of-range: {bad} (valid 0..{D_phase-1})")

        Xp, Wp, ptrp, ptrc, demo_skill = filter_demos_by_index(
            X_phase=Xp, W_phase=Wp, ptr_phase=ptrp, ptr_crop=ptrc, demo_skill=demo_skill, drop_ids=drop_demos
        )
        D_phase = int(ptrp.shape[0] - 1)

    Y_all = np.concatenate([Xp, Wp], axis=1)
    if Y_all.shape[1] != 6:
        raise ValueError(f"Expected 6D (xyz+rotvec), got {Y_all.shape}")

    # ---- group by skill (post-drop indices) ----
    skill_to_demo_idxs: Dict[int, List[int]] = {}
    for i in range(D_phase):
        sid = int(demo_skill[i])
        if sid >= 0:
            skill_to_demo_idxs.setdefault(sid, []).append(i)

    # ---- rep_top5 from pkl (if present); else computed on the fly per skill ----
    rep_top5_map = lib.get("rep_top5", {}) if isinstance(lib.get("rep_top5", {}), dict) else {}
    if not rep_top5_map:
        rep_top5_map = {}

    # We'll evaluate for skills that appear in ANY model library
    skills_eval = set()
    for k in ("dmp", "promp", "cpromp"):
        if isinstance(lib.get(k, {}), dict):
            skills_eval |= set(int(s) for s in lib[k].keys())
    skills_eval = sorted(skills_eval)

    print(f"[info] pkl: {pkl_path}")
    print(f"[info] npz: {source_npz}")
    print(f"[info] demos after drop: D={D_phase}, drop_demos={drop_demos}")
    print(f"[info] skills to eval (trained): {skills_eval}")

    # -----------------------
    # Evaluate per skill
    # -----------------------
    for sid in skills_eval:
        demo_idxs = skill_to_demo_idxs.get(int(sid), [])
        if not demo_idxs:
            print(f"\n[skill {sid}] no demos in NPZ after drop -> skip")
            continue

        Y_list, used, csce_list = build_skill_lists(
            sid=sid,
            demo_idxs=demo_idxs,
            Y_all=Y_all,
            ptrp=ptrp,
            npz_data=npz,
            ptrc=ptrc,
            phase_len=phase_len,
            min_len=min_len,
            pre_contact_steps=pre_steps,
            post_contact_steps=post_steps,
        )
        if not used:
            print(f"\n[skill {sid}] all demos filtered by min_len={min_len} -> skip")
            continue

        # rep-top5: pkl first, else recompute
        rep = rep_top5_map.get(int(sid), None)
        if not rep or (len(rep.get("top5_demo_indices_phase", [])) == 0):
            rep = pick_rep_top5_demos_for_skill(Y_list=Y_list, used=used, csce_list=csce_list, topk=5)
            rep_top5_map[int(sid)] = rep

        top5 = [int(x) for x in rep.get("top5_demo_indices_phase", [])]
        top5_set = set(top5)

        print(f"\n=== skill {sid} ===")
        if top5:
            print(f"[rep-top5] {top5} | median_contact_len={rep.get('median_contact_len', None)}")
        else:
            print("[rep-top5] (empty) -> top5 metrics will be nan")

        # ---- DMP metrics ----
        if int(sid) in lib.get("dmp", {}):
            dmp_entry = lib["dmp"][int(sid)]

            # NEW: handle both formats (backward compatible)
            if isinstance(dmp_entry, list):
                # old per-demo format (legacy)
                dmp_map = {int(it["demo_index_phase"]): it["dmp"] for it in dmp_entry}
                get_dmp_for_demo = lambda di: dmp_map.get(int(di), None)
            elif isinstance(dmp_entry, dict):
                # new prototype format (mean-w)
                dmp_proto = dmp_entry.get("dmp", None)
                if dmp_proto is None:
                    print("[DMP] entry has no 'dmp' -> skip")
                    get_dmp_for_demo = lambda di: None
                else:
                    get_dmp_for_demo = lambda di: dmp_proto
            else:
                print("[DMP] unexpected entry type -> skip")
                get_dmp_for_demo = lambda di: None

            rmse_pos_all, rmse_rot_all, rmse_all_all = [], [], []
            rmse_pos_top, rmse_rot_top, rmse_all_top = [], [], []

            # IMPORTANT:
            # If your CartesianDMP open_loop length differs from demo length,
            # rmse_pos_rot_all() already resamples yref to y length.
            for di in used:
                sp, ep = int(ptrp[di]), int(ptrp[di + 1])
                y_demo = np.asarray(Y_all[sp:ep], dtype=np.float64)

                dmp_use = get_dmp_for_demo(di)
                if dmp_use is None:
                    continue

                # reset before rollout to avoid state carryover between demos
                try:
                    dmp_use.reset()
                except Exception:
                    pass

                y_hat = dmp_open_loop_y6(dmp_use)
                m = rmse_pos_rot_all(y_demo, y_hat)

                rmse_pos_all.append(m["rmse_pos"])
                rmse_rot_all.append(m["rmse_rot"])
                rmse_all_all.append(m["rmse_all"])

                if di in top5_set:
                    rmse_pos_top.append(m["rmse_pos"])
                    rmse_rot_top.append(m["rmse_rot"])
                    rmse_all_top.append(m["rmse_all"])

            s_all_pos = summarize(rmse_pos_all)
            s_all_rot = summarize(rmse_rot_all)
            s_all_all = summarize(rmse_all_all)
            s_top_pos = summarize(rmse_pos_top)
            s_top_rot = summarize(rmse_rot_top)
            s_top_all = summarize(rmse_all_top)

            print("[DMP RMSE overall]")
            print(f"  pos mean/med/max = {s_all_pos['mean']:.6g}/{s_all_pos['median']:.6g}/{s_all_pos['max']:.6g}")
            print(f"  rot mean/med/max = {s_all_rot['mean']:.6g}/{s_all_rot['median']:.6g}/{s_all_rot['max']:.6g}")
            print(f"  all mean/med/max = {s_all_all['mean']:.6g}/{s_all_all['median']:.6g}/{s_all_all['max']:.6g}")

            print("[DMP RMSE rep-top5]")
            print(f"  pos mean/med/max = {s_top_pos['mean']:.6g}/{s_top_pos['median']:.6g}/{s_top_pos['max']:.6g}")
            print(f"  rot mean/med/max = {s_top_rot['mean']:.6g}/{s_top_rot['median']:.6g}/{s_top_rot['max']:.6g}")
            print(f"  all mean/med/max = {s_top_all['mean']:.6g}/{s_top_all['median']:.6g}/{s_top_all['max']:.6g}")
        else:
            print("[DMP] not in pkl")


        # ---- ProMP prior metrics ----
        if int(sid) in lib.get("promp", {}):
            entry = lib["promp"][int(sid)]
            y_mean = entry.get("y_mean", None)
            if y_mean is None:
                print("[ProMP prior] y_mean missing -> skip")
            else:
                y_mean = np.asarray(y_mean, dtype=np.float64)

                rmse_pos_all, rmse_rot_all, rmse_all_all = [], [], []
                rmse_pos_top, rmse_rot_top, rmse_all_top = [], [], []

                for y_demo, di in zip(Y_list, used):
                    m = rmse_pos_rot_all(y_demo, y_mean)
                    rmse_pos_all.append(m["rmse_pos"])
                    rmse_rot_all.append(m["rmse_rot"])
                    rmse_all_all.append(m["rmse_all"])
                    if di in top5_set:
                        rmse_pos_top.append(m["rmse_pos"])
                        rmse_rot_top.append(m["rmse_rot"])
                        rmse_all_top.append(m["rmse_all"])

                s_all_pos = summarize(rmse_pos_all)
                s_all_rot = summarize(rmse_rot_all)
                s_all_all = summarize(rmse_all_all)
                s_top_pos = summarize(rmse_pos_top)
                s_top_rot = summarize(rmse_rot_top)
                s_top_all = summarize(rmse_all_top)

                print("[ProMP prior RMSE overall]")
                print(f"  pos mean/med/max = {s_all_pos['mean']:.6g}/{s_all_pos['median']:.6g}/{s_all_pos['max']:.6g}")
                print(f"  rot mean/med/max = {s_all_rot['mean']:.6g}/{s_all_rot['median']:.6g}/{s_all_rot['max']:.6g}")
                print(f"  all mean/med/max = {s_all_all['mean']:.6g}/{s_all_all['median']:.6g}/{s_all_all['max']:.6g}")

                print("[ProMP prior RMSE rep-top5]")
                print(f"  pos mean/med/max = {s_top_pos['mean']:.6g}/{s_top_pos['median']:.6g}/{s_top_pos['max']:.6g}")
                print(f"  rot mean/med/max = {s_top_rot['mean']:.6g}/{s_top_rot['median']:.6g}/{s_top_rot['max']:.6g}")
                print(f"  all mean/med/max = {s_top_all['mean']:.6g}/{s_top_all['median']:.6g}/{s_top_all['max']:.6g}")
        else:
            print("[ProMP prior] not in pkl")

        # ---- cProMP metrics ----
        if int(sid) in lib.get("cpromp", {}):
            centry = lib["cpromp"][int(sid)]
            per_demo = centry.get("conditioned_per_demo", [])
            cmap = {int(it["demo_index_phase"]): np.asarray(it["y_cmean"], dtype=np.float64) for it in per_demo}

            rmse_pos_all, rmse_rot_all, rmse_all_all = [], [], []
            rmse_pos_top, rmse_rot_top, rmse_all_top = [], [], []

            for y_demo, di in zip(Y_list, used):
                if di not in cmap:
                    continue
                m = rmse_pos_rot_all(y_demo, cmap[di])
                rmse_pos_all.append(m["rmse_pos"])
                rmse_rot_all.append(m["rmse_rot"])
                rmse_all_all.append(m["rmse_all"])
                if di in top5_set:
                    rmse_pos_top.append(m["rmse_pos"])
                    rmse_rot_top.append(m["rmse_rot"])
                    rmse_all_top.append(m["rmse_all"])

            s_all_pos = summarize(rmse_pos_all)
            s_all_rot = summarize(rmse_rot_all)
            s_all_all = summarize(rmse_all_all)
            s_top_pos = summarize(rmse_pos_top)
            s_top_rot = summarize(rmse_rot_top)
            s_top_all = summarize(rmse_all_top)

            print("[cProMP RMSE overall]")
            print(f"  pos mean/med/max = {s_all_pos['mean']:.6g}/{s_all_pos['median']:.6g}/{s_all_pos['max']:.6g}")
            print(f"  rot mean/med/max = {s_all_rot['mean']:.6g}/{s_all_rot['median']:.6g}/{s_all_rot['max']:.6g}")
            print(f"  all mean/med/max = {s_all_all['mean']:.6g}/{s_all_all['median']:.6g}/{s_all_all['max']:.6g}")

            print("[cProMP RMSE rep-top5]")
            print(f"  pos mean/med/max = {s_top_pos['mean']:.6g}/{s_top_pos['median']:.6g}/{s_top_pos['max']:.6g}")
            print(f"  rot mean/med/max = {s_top_rot['mean']:.6g}/{s_top_rot['median']:.6g}/{s_top_rot['max']:.6g}")
            print(f"  all mean/med/max = {s_top_all['mean']:.6g}/{s_top_all['median']:.6g}/{s_top_all['max']:.6g}")
        else:
            print("[cProMP] not in pkl")

    # -----------------------
    # Plot overlay for one demo
    # -----------------------
    if args.plot:
        i = int(args.plot_demo)
        if not (0 <= i < D_phase):
            raise ValueError(f"--plot_demo must be within [0, {D_phase-1}]")

        sp, ep = int(ptrp[i]), int(ptrp[i + 1])
        y = np.asarray(Y_all[sp:ep], dtype=np.float64)
        sid = int(demo_skill[i])

        overlays: List[Tuple[str, np.ndarray]] = []
        vlines = {"phase0": 0, "phase1": y.shape[0] - 1}
        title = f"phase-demo={i} skill={sid}"

        # ProMP prior mean
        if int(sid) in lib.get("promp", {}):
            ym = lib["promp"][int(sid)].get("y_mean", None)
            if ym is not None:
                overlays.append(("ProMP prior mean", np.asarray(ym, dtype=np.float64)))

        # cProMP mean for this demo
        if int(sid) in lib.get("cpromp", {}):
            items = lib["cpromp"][int(sid)].get("conditioned_per_demo", [])
            for it in items:
                if int(it["demo_index_phase"]) == i:
                    overlays.append(("cProMP mean", np.asarray(it["y_cmean"], dtype=np.float64)))
                    cs_idx = int(it.get("cs_idx", 0))
                    ce_idx = int(it.get("ce_idx", y.shape[0] - 1))
                    vlines = {"phase0": 0, "cs": cs_idx, "ce": ce_idx, "phase1": y.shape[0] - 1}
                    break

        # DMP open_loop for this demo (NEW: prototype or legacy)
        if int(sid) in lib.get("dmp", {}):
            dmp_entry = lib["dmp"][int(sid)]
            dmp_use = None

            if isinstance(dmp_entry, dict):
                dmp_use = dmp_entry.get("dmp", None)
            elif isinstance(dmp_entry, list):
                for it in dmp_entry:
                    if int(it.get("demo_index_phase", -1)) == i:
                        dmp_use = it.get("dmp", None)
                        break

            if dmp_use is not None:
                try:
                    dmp_use.reset()
                except Exception:
                    pass
                overlays.append(("DMP open_loop", dmp_open_loop_y6(dmp_use)))


        out_png = plot_dir / f"overlay_demo_{i:03d}_skill_{sid}.png"
        plot_overlay_6d(y=y, overlays=overlays, title=title, out_png=out_png, t=phase_grid, vlines=vlines)
        print(f"\n[plot] {out_png}")

        # -----------------------
        # (NEW) Skill 1~8: ProMP prior means in one figure
        # -----------------------
        skill_means_prior: Dict[int, np.ndarray] = {}
        for sid2 in range(1, 9):
            if sid2 in lib.get("promp", {}):
                ym = lib["promp"][sid2].get("y_mean", None)
                if ym is not None:
                    skill_means_prior[sid2] = np.asarray(ym, dtype=np.float64)

        if len(skill_means_prior) > 0:
            out_png2 = plot_dir / "skills_01_to_08_promp_mean_xyz_w.png"
            plot_skill_means_1to8(
                skill_means=skill_means_prior,
                t=phase_grid,
                out_png=out_png2,
                title="ProMP prior means for skills 1~8 (xyz + wx wy wz)",
            )
            print(f"[plot] {out_png2}")
        else:
            print("[plot] No ProMP prior means available for skills 1..8.")

        # -----------------------
        # (NEW) Skill 1~8: cProMP representative means in one figure
        # -----------------------
        skill_means_cond: Dict[int, np.ndarray] = {}
        for sid2 in range(1, 9):
            if sid2 not in lib.get("cpromp", {}):
                continue
            items = lib["cpromp"][sid2].get("conditioned_per_demo", [])
            if not items:
                continue

            # pick representative demo:
            # 1) rep-top5 rank #1 if exists in rep_top5_map
            # 2) else median contact_len among items (needs contact_len stored)
            rep_demo_idx: Optional[int] = None
            rep = rep_top5_map.get(sid2, None)
            if rep and len(rep.get("top5_demo_indices_phase", [])) > 0:
                rep_demo_idx = int(rep["top5_demo_indices_phase"][0])

            y_rep: Optional[np.ndarray] = None
            if rep_demo_idx is not None:
                for it in items:
                    if int(it.get("demo_index_phase", -1)) == rep_demo_idx:
                        y_rep = np.asarray(it["y_cmean"], dtype=np.float64)
                        break

            if y_rep is None:
                # fallback: choose closest-to-median contact_len
                lens = []
                for it in items:
                    if "contact_len" in it:
                        lens.append(float(it["contact_len"]))
                    else:
                        lens.append(np.nan)
                lens_arr = np.asarray(lens, dtype=np.float64)
                if np.all(np.isnan(lens_arr)):
                    # last fallback: just take first
                    y_rep = np.asarray(items[0]["y_cmean"], dtype=np.float64)
                else:
                    med = float(np.nanmedian(lens_arr))
                    rep_i = int(np.nanargmin(np.abs(lens_arr - med)))
                    y_rep = np.asarray(items[rep_i]["y_cmean"], dtype=np.float64)

            skill_means_cond[sid2] = y_rep

        if len(skill_means_cond) > 0:
            out_png3 = plot_dir / "skills_01_to_08_cpromp_rep_mean_xyz_w.png"
            plot_skill_means_1to8(
                skill_means=skill_means_cond,
                t=phase_grid,
                out_png=out_png3,
                title="cProMP representative means for skills 1~8 (xyz + wx wy wz)",
            )
            print(f"[plot] {out_png3}")
        else:
            print("[plot] No cProMP representative means available for skills 1..8.")


if __name__ == "__main__":
    main()
