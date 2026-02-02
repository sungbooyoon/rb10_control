#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_spromp.py (MODE A ONLY = OPTION-1)

MODE A (OPTION-1):
- Evaluate on the SAME demo set used for BGMM style assignment:
    style_pkl["used_demo_indices_original"] (and corresponding "labels")
- This gives "one RMSE per demo" and makes N comparable across models.

Evaluate + plot style-conditioned ProMP (sProMP) on FULL PHASE trajectories.
- Window-after-contact (L) is used ONLY for BGMM style assignment (already done in style_pkl),
  and is NOT used for evaluation or plotting.

This script:
- Loads spromp.pkl (style-conditioned ProMP library; expected trained on full phase).
- Loads style_pkl (BGMM result) to get:
    * used_demo_indices_original (ORIGINAL ids)
    * labels (style id per used demo)
    * optional drop_demos (not used for MODE A evaluation set; only reported)
- Loads NPZ trajectories (phase-aligned) and evaluates per-style:
    * ProMP mean vs demos belonging to that style (FULL PHASE)
    * RMSE: pos / rot / all
- Plot per-style:
    * overlay plot for one representative demo: demo vs style ProMP mean (FULL PHASE)
    * style demo mean ± std (across demos in that style, FULL PHASE)
    * style ProMP mean ± conf (from saved y_var if present, FULL PHASE)

NPZ subset handling (important):
- BGMM indices are "ORIGINAL demo indices".
- If NPZ is a filtered subset, it SHOULD contain:
    kept_orig_demo_index : (D0,) mapping NPZ local demo idx -> ORIGINAL demo idx
  If absent, assume local==original.

Assumptions:
- NPZ has: X_phase_crop (N,3), W_phase_crop (N,3), demo_ptr_phase (D+1,)
- style_pkl (BGMM) has:
    - used_demo_indices_original : (N_used,)
    - labels : (N_used,)
- spromp.pkl has:
    - library: dict {style_id: entry}, each entry contains y_mean (T,6) and optional y_var (T,6)
    - source_npz path (optional; can override with --npz)

Example:
  python3 eval_spromp.py \
    --pkl /home/sungboo/rb10_control/dataset/spromp_multi_exp3.pkl \
    --style_pkl /home/sungboo/rb10_control/dataset/test_bgmm_exp3.pkl \
    --plot \
    --plot_dir /home/sungboo/rb10_control/images/demo_20260122_exp3/spromp_multi_modeA
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# basic utils
# -----------------------------
def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def resample(y: np.ndarray, Tnew: int) -> np.ndarray:
    """Linear resample (T,D) -> (Tnew,D)."""
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


def _finite(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(x))))


def rmse_pos_rot_all(y: np.ndarray, yref: np.ndarray) -> Dict[str, float]:
    """
    Compare demo y (T,6) to reference yref (Tref,6); resample yref to y if needed.
    """
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
        return {"mean": np.nan, "median": np.nan, "max": np.nan, "n": 0}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
        "n": int(x.size),
    }


def summarize_mean_std(vals: List[float]) -> Dict[str, float]:
    x = np.asarray(vals, dtype=np.float64)
    if x.size == 0:
        return {"mean": np.nan, "std": np.nan, "n": 0}
    std = float(np.std(x, ddof=1)) if x.size >= 2 else 0.0
    return {"mean": float(np.mean(x)), "std": std, "n": int(x.size)}


# -----------------------------
# plots (FULL PHASE)
# -----------------------------
def plot_overlay_6d(
    y: np.ndarray,          # (T,6) demo
    y_mean: np.ndarray,     # (Tref,6) model mean
    title: str,
    out_png: Path,
):
    y = np.asarray(y, dtype=np.float64)
    y_mean = np.asarray(y_mean, dtype=np.float64)

    Tn = int(y.shape[0])
    t = np.linspace(0.0, 1.0, Tn, dtype=np.float64)

    if y_mean.shape[0] != Tn:
        y_mean = resample(y_mean, Tn)

    labels = ["x", "y", "z", "wx", "wy", "wz"]
    plt.figure(figsize=(12, 14))
    for k in range(6):
        ax = plt.subplot(6, 1, k + 1)
        ax.plot(t, y[:, k], label="demo")
        ax.plot(t, y_mean[:, k], label="style ProMP mean")
        ax.set_ylabel(labels[k])
        if k == 0:
            ax.set_title(title)
            ax.legend(loc="upper right", fontsize=9)
        if k == 5:
            ax.set_xlabel("t (0..1)")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_demo_mean_std_6d(
    demos: List[np.ndarray],  # list of (T,6), assumed same T
    t: np.ndarray,
    out_png: Path,
    title: str,
):
    if len(demos) == 0:
        return
    Y = np.stack(demos, axis=0)  # (N,T,6)
    mu = np.mean(Y, axis=0)
    sd = np.std(Y, axis=0, ddof=1) if Y.shape[0] >= 2 else np.zeros_like(mu)

    labels = ["x", "y", "z", "wx", "wy", "wz"]
    t = np.asarray(t, dtype=np.float64).reshape(-1)

    plt.figure(figsize=(12, 14))
    for k in range(6):
        ax = plt.subplot(6, 1, k + 1)
        ax.plot(t, mu[:, k], label="demo mean")
        ax.fill_between(
            t,
            (mu[:, k] - sd[:, k]).ravel(),
            (mu[:, k] + sd[:, k]).ravel(),
            alpha=0.25,
            label="±1 std" if k == 0 else None,
        )
        ax.set_ylabel(labels[k])
        if k == 0:
            ax.set_title(title)
            ax.legend(loc="upper right", fontsize=9)
        if k == 5:
            ax.set_xlabel("t (0..1)")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_mean_conf_6d(
    t: np.ndarray,
    y_mean: np.ndarray,
    y_var: np.ndarray,
    out_png: Path,
    title: str,
    conf_scale: float = 1.96,
):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y_mean = np.asarray(y_mean, dtype=np.float64)
    y_var = np.asarray(y_var, dtype=np.float64)

    if y_mean.shape[0] != t.shape[0]:
        y_mean = resample(y_mean, t.shape[0])
    if y_var.shape[0] != t.shape[0]:
        y_var = resample(y_var, t.shape[0])

    conf = conf_scale * np.sqrt(np.maximum(y_var, 0.0))

    labels = ["x", "y", "z", "wx", "wy", "wz"]
    plt.figure(figsize=(12, 14))
    for k in range(6):
        ax = plt.subplot(6, 1, k + 1)
        ax.plot(t, y_mean[:, k], label="mean")
        ax.fill_between(
            t,
            (y_mean[:, k] - conf[:, k]).ravel(),
            (y_mean[:, k] + conf[:, k]).ravel(),
            alpha=0.30,
            label="±conf" if k == 0 else None,
        )
        ax.set_ylabel(labels[k])
        if k == 0:
            ax.set_title(title)
            ax.legend(loc="upper right", fontsize=9)
        if k == 5:
            ax.set_xlabel("t (0..1)")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


# -----------------------------
# payload parsing helpers
# -----------------------------
def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Pickle must contain a dict, got {type(obj)}")
    return obj


def _extract_style_table_from_spromp(sp: dict) -> Dict[int, dict]:
    """
    spromp.pkl expected: payload['library'] = {style_id: entry}
    where style_id are ints (0..K-1) or str-digits.
    """
    lib = sp.get("library", None)
    if not isinstance(lib, dict) or len(lib) == 0:
        raise ValueError("spromp.pkl: payload['library'] must be a non-empty dict.")

    table: Dict[int, dict] = {}
    for k, v in lib.items():
        try:
            kk = int(k)
        except Exception:
            continue
        if isinstance(v, dict):
            table[kk] = v
    if len(table) == 0:
        raise ValueError(f"spromp.pkl: library has no int-like keys. keys={list(lib.keys())[:20]}")
    return table


def _get_bgmm_used_and_labels(style_bgmm: dict) -> Tuple[np.ndarray, np.ndarray]:
    if "used_demo_indices_original" not in style_bgmm or "labels" not in style_bgmm:
        raise ValueError("style_pkl must contain used_demo_indices_original and labels.")
    used_orig = np.asarray(style_bgmm["used_demo_indices_original"], dtype=np.int64).reshape(-1)
    labels = np.asarray(style_bgmm["labels"], dtype=np.int64).reshape(-1)
    if used_orig.shape[0] != labels.shape[0]:
        raise ValueError("style_pkl: used_demo_indices_original and labels length mismatch.")
    return used_orig, labels


def _get_drop_set(style_bgmm: dict) -> set:
    dd = style_bgmm.get("drop_demos", [])
    if isinstance(dd, (list, tuple, np.ndarray)):
        return set(int(x) for x in list(dd))
    return set()


def _extract_demo_full_phase_local(
    Xp: np.ndarray,
    Wp: np.ndarray,
    ptrp: np.ndarray,
    demo_local: int,
) -> np.ndarray:
    """Return y_full (T,6) for NPZ LOCAL demo index demo_local."""
    sp, ep = int(ptrp[demo_local]), int(ptrp[demo_local + 1])
    xyz = np.asarray(Xp[sp:ep], dtype=np.float64)
    rot = np.asarray(Wp[sp:ep], dtype=np.float64)
    if xyz.shape[0] != rot.shape[0]:
        raise ValueError(f"X/W length mismatch for demo_local {demo_local}: {xyz.shape[0]} vs {rot.shape[0]}")
    return np.concatenate([xyz, rot], axis=1)


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="spromp.pkl (style-conditioned ProMP)")
    ap.add_argument("--style_pkl", required=True, help="BGMM style assignment pkl (discover_styles_bgmm.py output)")
    ap.add_argument("--npz", default=None, help="Override NPZ path (default: spromp payload['source_npz'])")

    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_dir", default=None)
    ap.add_argument(
        "--plot_demo",
        type=int,
        default=None,
        help="Optional: ORIGINAL demo index to overlay. Must be in BGMM used set AND present in this NPZ subset.",
    )

    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--conf_scale", type=float, default=1.96)
    ap.add_argument(
        "--strict_T_match",
        action="store_true",
        help="If set, require y_mean length == demo length; otherwise resample y_mean/y_var to demo length.",
    )
    args = ap.parse_args()

    sp_path = Path(args.pkl)
    style_path = Path(args.style_pkl)

    sp = _load_pickle(sp_path)
    style_bgmm = _load_pickle(style_path)

    style_table = _extract_style_table_from_spromp(sp)  # style_id -> entry
    used_orig_all, labels_all = _get_bgmm_used_and_labels(style_bgmm)  # ORIGINAL ids + labels
    drop_set = _get_drop_set(style_bgmm)

    # NPZ path
    npz_path = Path(args.npz) if args.npz is not None else Path(sp.get("source_npz", ""))
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    npz = np.load(npz_path, allow_pickle=True)
    for k in ("X_phase_crop", "W_phase_crop", "demo_ptr_phase"):
        if k not in npz:
            raise KeyError(f"NPZ must contain key: {k}")

    Xp = np.asarray(npz["X_phase_crop"], dtype=np.float64)
    Wp = np.asarray(npz["W_phase_crop"], dtype=np.float64)
    ptrp = np.asarray(npz["demo_ptr_phase"], dtype=np.int64)

    D0 = int(ptrp.shape[0] - 1)
    if D0 <= 0:
        raise ValueError("No demos in NPZ (demo_ptr_phase too short).")

    # nominal phase length
    T_phase_nom = int(ptrp[1] - ptrp[0])
    if T_phase_nom <= 1:
        raise ValueError(f"Bad nominal phase length T={T_phase_nom}")

    # local->orig mapping (NPZ subset support)
    if "kept_orig_demo_index" in npz:
        local_to_orig = np.asarray(npz["kept_orig_demo_index"], dtype=np.int64).reshape(-1)
        if local_to_orig.shape[0] != D0:
            raise ValueError(
                f"NPZ kept_orig_demo_index length mismatch: {local_to_orig.shape[0]} vs D0={D0}"
            )
    else:
        local_to_orig = np.arange(D0, dtype=np.int64)

    # build orig->local map (keep FIRST occurrence if duplicates)
    orig_to_local: Dict[int, int] = {}
    for di in range(D0):
        o = int(local_to_orig[di])
        if o not in orig_to_local:
            orig_to_local[o] = int(di)

    # MODE A: evaluation demo set = BGMM used demos ∩ NPZ-present demos
    used_pairs: List[Tuple[int, int]] = []  # (demo_local, style_id)
    n_missing_in_npz = 0
    n_style_not_in_model = 0
    for o, sid in zip(used_orig_all.tolist(), labels_all.tolist()):
        o = int(o)
        sid = int(sid)
        if o not in orig_to_local:
            n_missing_in_npz += 1
            continue
        if sid not in style_table:
            n_style_not_in_model += 1
            continue
        used_pairs.append((orig_to_local[o], sid))

    used_pairs = list(used_pairs)
    N_eval = len(used_pairs)

    # group LOCAL demos by style
    per_style_demos_local: Dict[int, List[int]] = {sid: [] for sid in style_table.keys()}
    for di_local, sid in used_pairs:
        per_style_demos_local[sid].append(int(di_local))

    print(f"[info] spromp.pkl: {sp_path}")
    print(f"[info] style_pkl: {style_path}")
    print(f"[info] npz: {npz_path}")
    print(f"[info] npz demos D0={D0}, nominal T_phase={T_phase_nom}")
    print(f"[info] styles_in_spromp={sorted(style_table.keys())}")
    print(f"[info] BGMM used demos (original) N_all={int(used_orig_all.size)} | labels N={int(labels_all.size)}")
    if "kept_orig_demo_index" in npz:
        o_min = int(np.min(local_to_orig)) if D0 > 0 else -1
        o_max = int(np.max(local_to_orig)) if D0 > 0 else -1
        print(f"[info] NPZ subset mapping present: kept_orig_demo_index (orig range in NPZ: {o_min}..{o_max})")
    else:
        print("[info] NPZ subset mapping NOT present: assuming local==original")

    print(f"[info] MODE A eval demo set = (BGMM used) ∩ (present in NPZ) ∩ (style in model)")
    print(f"[info]   -> N_eval={N_eval}")
    if n_missing_in_npz > 0:
        print(f"[warn] {n_missing_in_npz} BGMM-used demos missing in this NPZ subset -> ignored")
    if n_style_not_in_model > 0:
        print(f"[warn] {n_style_not_in_model} BGMM-used demos have style not in spromp library -> ignored")
    if len(drop_set) > 0:
        print(f"[note] style_pkl drop_demos exists (len={len(drop_set)}), but MODE A does NOT use drop_demos to define eval set.")

    results: Dict[int, dict] = {}

    # global accumulators (ALL used demos across ALL styles; one RMSE per demo)
    g_pos: List[float] = []
    g_rot: List[float] = []
    g_all: List[float] = []

    for style_id in sorted(style_table.keys()):
        entry = style_table[style_id]
        y_mean = entry.get("y_mean", None)
        if y_mean is None:
            print(f"\n[style {style_id}] entry has no y_mean -> skip")
            continue
        y_mean = np.asarray(y_mean, dtype=np.float64)

        y_var = entry.get("y_var", None)
        y_var = None if y_var is None else np.asarray(y_var, dtype=np.float64)

        demos_local = per_style_demos_local.get(style_id, [])
        if len(demos_local) == 0:
            print(f"\n[style {style_id}] no demos assigned (MODE A) -> skip")
            continue

        rmse_pos: List[float] = []
        rmse_rot: List[float] = []
        rmse_all: List[float] = []
        demos_full: List[np.ndarray] = []
        used_demos_orig: List[int] = []
        used_demos_local: List[int] = []
        drops: Dict[str, int] = {}

        for di_local in demos_local:
            y_full = _extract_demo_full_phase_local(Xp, Wp, ptrp, demo_local=di_local)

            if y_full.shape[0] < int(args.min_len):
                drops["min_len"] = drops.get("min_len", 0) + 1
                continue
            if not _finite(y_full):
                drops["nonfinite_full"] = drops.get("nonfinite_full", 0) + 1
                continue

            if args.strict_T_match and (y_mean.shape[0] != y_full.shape[0]):
                drops["T_mismatch_strict"] = drops.get("T_mismatch_strict", 0) + 1
                continue

            m = rmse_pos_rot_all(y_full, y_mean)
            rmse_pos.append(m["rmse_pos"])
            rmse_rot.append(m["rmse_rot"])
            rmse_all.append(m["rmse_all"])

            # global accumulators
            g_pos.append(m["rmse_pos"])
            g_rot.append(m["rmse_rot"])
            g_all.append(m["rmse_all"])

            demos_full.append(y_full)
            used_demos_local.append(int(di_local))
            used_demos_orig.append(int(local_to_orig[di_local]))

        s_pos = summarize(rmse_pos)
        s_rot = summarize(rmse_rot)
        s_all = summarize(rmse_all)

        print(f"\n=== style {style_id} ===")
        print(f"[assigned demos] {len(demos_local)}  | [used] {s_all['n']}")
        if drops:
            drops_sorted = sorted(drops.items(), key=lambda x: -x[1])
            msg = ", ".join([f"{k}:{v}" for k, v in drops_sorted])
            print(f"[dropped reasons] {msg}")

        print("[ProMP RMSE] (FULL PHASE, MODE A: 1 RMSE per demo)")
        print(f"  pos mean/med/max = {s_pos['mean']:.6g}/{s_pos['median']:.6g}/{s_pos['max']:.6g}")
        print(f"  rot mean/med/max = {s_rot['mean']:.6g}/{s_rot['median']:.6g}/{s_rot['max']:.6g}")
        print(f"  all mean/med/max = {s_all['mean']:.6g}/{s_all['median']:.6g}/{s_all['max']:.6g}")

        results[int(style_id)] = {
            "style_id": int(style_id),
            "n_assigned": int(len(demos_local)),
            "n_used": int(s_all["n"]),
            "drops": dict(drops),
            "rmse_pos": s_pos,
            "rmse_rot": s_rot,
            "rmse_all": s_all,
            "used_demo_indices_local": used_demos_local,
            "used_demo_indices_original": used_demos_orig,
        }

        # -----------------------
        # plotting per-style (FULL PHASE)
        # -----------------------
        if args.plot and s_all["n"] > 0:
            plot_dir = Path(args.plot_dir) if args.plot_dir is not None else Path("./spromp_eval_modeA_plots")
            plot_dir.mkdir(parents=True, exist_ok=True)

            # choose overlay demo (ORIGINAL id constraint)
            overlay_local: Optional[int] = None
            overlay_orig: Optional[int] = None

            if args.plot_demo is not None:
                cand_orig = int(args.plot_demo)
                if cand_orig in orig_to_local:
                    cand_local = orig_to_local[cand_orig]
                    # must be USED (after drops) in this style:
                    if cand_local in used_demos_local:
                        overlay_local = int(cand_local)
                        overlay_orig = int(cand_orig)

            if overlay_local is None:
                overlay_local = used_demos_local[0]
                overlay_orig = int(local_to_orig[overlay_local])

            y_overlay = _extract_demo_full_phase_local(Xp, Wp, ptrp, demo_local=overlay_local)

            out1 = plot_dir / f"style_{style_id:02d}_overlay_demo_orig_{overlay_orig:03d}.png"
            y_mean_ov = y_mean if y_mean.shape[0] == y_overlay.shape[0] else resample(y_mean, y_overlay.shape[0])
            plot_overlay_6d(
                y=y_overlay,
                y_mean=y_mean_ov,
                title=f"style {style_id} | demo_orig={overlay_orig} | FULL PHASE (MODE A)",
                out_png=out1,
            )
            print(f"[plot] {out1}")

            # demo mean ± std: resample demos to common T (use y_mean length)
            T_ref = int(y_mean.shape[0])
            demos_rs = [resample(d, T_ref) if d.shape[0] != T_ref else d for d in demos_full]
            out2 = plot_dir / f"style_{style_id:02d}_demo_mean_std_fullphase.png"
            plot_demo_mean_std_6d(
                demos=demos_rs,
                t=np.linspace(0.0, 1.0, T_ref, dtype=np.float64),
                out_png=out2,
                title=f"style {style_id}: demo mean ± std (FULL PHASE, MODE A, resampled to T={T_ref})",
            )
            print(f"[plot] {out2}")

            # promp mean ± conf (if y_var exists)
            if y_var is not None:
                out3 = plot_dir / f"style_{style_id:02d}_promp_mean_conf_fullphase.png"
                plot_mean_conf_6d(
                    t=np.linspace(0.0, 1.0, int(y_mean.shape[0]), dtype=np.float64),
                    y_mean=y_mean,
                    y_var=y_var,
                    out_png=out3,
                    title=f"style {style_id}: ProMP mean ± {args.conf_scale}*std (FULL PHASE, MODE A)",
                    conf_scale=float(args.conf_scale),
                )
                print(f"[plot] {out3}")

    # per-style summary
    print("\n[summary] styles evaluated:")
    for sid in sorted(results.keys()):
        r = results[sid]
        print(
            f"  style {sid:02d}: used {r['n_used']}/{r['n_assigned']} | "
            f"rmse_all mean/med/max={r['rmse_all']['mean']:.4g}/{r['rmse_all']['median']:.4g}/{r['rmse_all']['max']:.4g}"
        )

    # global overall RMSE (ALL used demos across ALL styles)
    print("\n==============================")
    print("[GLOBAL RMSE over ALL used demos (all styles)]")
    print("==============================")

    if len(g_all) > 0:
        s_pos_g = summarize_mean_std(g_pos)
        s_rot_g = summarize_mean_std(g_rot)
        s_all_g = summarize_mean_std(g_all)
        print(f"[sProMP-multi] N={s_all_g['n']}")
        print(f"  pos mean±std = {s_pos_g['mean']:.6g} ± {s_pos_g['std']:.6g}")
        print(f"  rot mean±std = {s_rot_g['mean']:.6g} ± {s_rot_g['std']:.6g}")
        print(f"  all mean±std = {s_all_g['mean']:.6g} ± {s_all_g['std']:.6g}")
    else:
        print("(no samples)")

    # extra sanity print: expected N_eval vs actual used count
    used_total = int(sum(results[sid]["n_used"] for sid in results.keys()))
    if used_total != N_eval:
        print(f"[warn] Used count after per-demo dropping ({used_total}) != N_eval before drops ({N_eval}). "
              f"This is expected if min_len/nonfinite/strict_T_match dropped some demos.")


if __name__ == "__main__":
    main()
