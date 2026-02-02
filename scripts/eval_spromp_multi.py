#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_spromp.py (MODE A ONLY)

Evaluate + plot style-conditioned ProMP (sProMP) on FULL PHASE trajectories.
- Window-after-contact (L) is used ONLY for BGMM style assignment (already done in style_pkl),
  and is NOT used for evaluation or plotting.

This script:
- Loads spromp.pkl (trained style-conditioned ProMP library; expected to be trained on full phase).
- Loads style_pkl (BGMM result) to get per-demo style labels and (optionally) drop_demos.
- Loads NPZ trajectories (phase-aligned) and evaluates per-style:
    * ProMP mean vs demos belonging to that style (FULL PHASE)
    * RMSE: pos / rot / all
- Plot per-style:
    * overlay plot for one representative demo: demo vs style ProMP mean (FULL PHASE)
    * style demo mean ± std (across demos in that style, FULL PHASE)
    * style ProMP mean ± conf (from saved y_var if present, FULL PHASE)

Assumptions:
- NPZ has: X_phase_crop (N,3), W_phase_crop (N,3), demo_ptr_phase (D+1,)
- style_pkl (BGMM) has:
    - labels for used demos, and mapping to original demo indices
    - optional drop_demos (list of orig indices dropped during BGMM training)
- spromp.pkl has:
    - library: dict {style_id: entry}, each entry contains y_mean (T,6) and optional y_var (T,6)
    - source_npz path (optional; can override with --npz)

Example:
  python3 eval_spromp.py \
    --pkl /home/sungboo/rb10_control/dataset/spromp.pkl \
    --style_pkl /home/sungboo/rb10_control/dataset/test_bgmm.pkl \
    --plot \
    --plot_dir /home/sungboo/rb10_control/images/spromp_eval_full
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

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
    t: Optional[np.ndarray] = None,
):
    y = np.asarray(y, dtype=np.float64)
    y_mean = np.asarray(y_mean, dtype=np.float64)

    Tn = int(y.shape[0])
    tt = np.arange(Tn) if t is None else np.asarray(t, dtype=np.float64).reshape(-1)

    if y_mean.shape[0] != Tn:
        y_mean = resample(y_mean, Tn)

    labels = ["x", "y", "z", "wx", "wy", "wz"]
    plt.figure(figsize=(12, 14))
    for k in range(6):
        ax = plt.subplot(6, 1, k + 1)
        ax.plot(tt, y[:, k], label="demo")
        ax.plot(tt, y_mean[:, k], label="style ProMP mean")
        ax.set_ylabel(labels[k])
        if k == 0:
            ax.set_title(title)
            ax.legend(loc="upper right", fontsize=9)
        if k == 5:
            ax.set_xlabel("phase_idx" if t is None else "t (0..1)")

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


def _build_orig_to_style_map(style_bgmm: dict) -> Dict[int, int]:
    """
    BGMM payload should have:
      - used_demo_indices_original : (N_used,)
      - labels : (N_used,)
    """
    if "used_demo_indices_original" not in style_bgmm or "labels" not in style_bgmm:
        raise ValueError("style_pkl must contain used_demo_indices_original and labels.")
    used_orig = np.asarray(style_bgmm["used_demo_indices_original"], dtype=np.int64).reshape(-1)
    labels = np.asarray(style_bgmm["labels"], dtype=np.int64).reshape(-1)
    if used_orig.shape[0] != labels.shape[0]:
        raise ValueError("style_pkl: used_demo_indices_original and labels length mismatch.")
    return {int(o): int(l) for o, l in zip(used_orig.tolist(), labels.tolist())}


def _get_drop_set(style_bgmm: dict) -> set:
    dd = style_bgmm.get("drop_demos", [])
    if isinstance(dd, (list, tuple, np.ndarray)):
        return set(int(x) for x in list(dd))
    return set()


def _extract_demo_full_phase(
    Xp: np.ndarray,
    Wp: np.ndarray,
    ptrp: np.ndarray,
    demo_orig: int,
) -> np.ndarray:
    """Return y_full (T,6) for original demo index demo_orig."""
    sp, ep = int(ptrp[demo_orig]), int(ptrp[demo_orig + 1])
    xyz = np.asarray(Xp[sp:ep], dtype=np.float64)
    rot = np.asarray(Wp[sp:ep], dtype=np.float64)
    if xyz.shape[0] != rot.shape[0]:
        raise ValueError(f"X/W length mismatch for demo {demo_orig}: {xyz.shape[0]} vs {rot.shape[0]}")
    return np.concatenate([xyz, rot], axis=1)


def _entry_time_grid(entry: dict, T_fallback: int) -> np.ndarray:
    """
    Prefer entry['t'] if present; else make linspace(0,1,T).
    """
    if isinstance(entry, dict) and ("t" in entry):
        try:
            t = np.asarray(entry["t"], dtype=np.float64).reshape(-1)
            if t.size >= 2:
                return t
        except Exception:
            pass
    return np.linspace(0.0, 1.0, int(T_fallback), dtype=np.float64)


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
    ap.add_argument("--plot_demo", type=int, default=None,
                    help="Optional: ORIGINAL demo index to overlay. If not given, pick first used demo in each style.")

    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--conf_scale", type=float, default=1.96)
    ap.add_argument("--strict_T_match", action="store_true",
                    help="If set, require y_mean length == T_phase. Otherwise resample y_mean/y_var to demo length.")
    args = ap.parse_args()

    sp_path = Path(args.pkl)
    style_path = Path(args.style_pkl)

    sp = _load_pickle(sp_path)
    style_bgmm = _load_pickle(style_path)

    style_table = _extract_style_table_from_spromp(sp)      # style_id -> entry
    orig_to_style = _build_orig_to_style_map(style_bgmm)    # orig_demo_idx -> style_id
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

    # infer a nominal T_phase from demo_0
    T_phase_nom = int(ptrp[1] - ptrp[0])
    if T_phase_nom <= 1:
        raise ValueError(f"Bad nominal phase length T={T_phase_nom}")

    kept_original = [i for i in range(D0) if i not in drop_set]
    print(f"[info] spromp.pkl: {sp_path}")
    print(f"[info] style_pkl: {style_path}")
    print(f"[info] npz: {npz_path}")
    print(f"[info] demos original D0={D0}, drop={sorted(drop_set)}, kept={len(kept_original)}")
    print(f"[info] nominal T_phase={T_phase_nom}")
    print(f"[info] styles_in_spromp={sorted(style_table.keys())}")
    print(f"[info] eval mode: FULL_PHASE (window NOT used)")

    # group original demos by style (only those that are kept and appear in BGMM mapping)
    per_style_demos_orig: Dict[int, List[int]] = {sid: [] for sid in style_table.keys()}
    n_unlabeled = 0
    for orig in kept_original:
        sid = int(orig_to_style.get(int(orig), -1))
        if sid in per_style_demos_orig:
            per_style_demos_orig[sid].append(int(orig))
        else:
            n_unlabeled += 1

    if n_unlabeled > 0:
        print(f"[warn] {n_unlabeled} kept demos have no style label (or style not in spromp library) -> ignored")

    results: Dict[int, dict] = {}
    # -----------------------
    # Global RMSE accumulators (ALL used demos across ALL styles)
    # -----------------------
    g_pos, g_rot, g_all = [], [], []


    for style_id in sorted(style_table.keys()):
        entry = style_table[style_id]
        y_mean = entry.get("y_mean", None)
        if y_mean is None:
            print(f"\n[style {style_id}] entry has no y_mean -> skip")
            continue
        y_mean = np.asarray(y_mean, dtype=np.float64)

        y_var = entry.get("y_var", None)
        y_var = None if y_var is None else np.asarray(y_var, dtype=np.float64)

        demos_orig = per_style_demos_orig.get(style_id, [])
        if len(demos_orig) == 0:
            print(f"\n[style {style_id}] no demos assigned -> skip")
            continue

        rmse_pos, rmse_rot, rmse_all = [], [], []
        demos_full: List[np.ndarray] = []
        used_demos_orig: List[int] = []
        drops: Dict[str, int] = {}

        # time grid for plots (prefer entry['t'])
        t_style = _entry_time_grid(entry, T_fallback=int(y_mean.shape[0]))

        for orig in demos_orig:
            y_full = _extract_demo_full_phase(Xp, Wp, ptrp, demo_orig=orig)

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

            # accumulate global (only USED demos)
            g_pos.extend(rmse_pos)
            g_rot.extend(rmse_rot)
            g_all.extend(rmse_all)


            demos_full.append(y_full)
            used_demos_orig.append(int(orig))

        s_pos = summarize(rmse_pos)
        s_rot = summarize(rmse_rot)
        s_all = summarize(rmse_all)

        print(f"\n=== style {style_id} ===")
        print(f"[assigned demos] {len(demos_orig)}  | [used] {s_all['n']}")
        if drops:
            drops_sorted = sorted(drops.items(), key=lambda x: -x[1])
            msg = ", ".join([f"{k}:{v}" for k, v in drops_sorted])
            print(f"[dropped reasons] {msg}")

        print("[ProMP RMSE] (FULL PHASE)")
        print(f"  pos mean/med/max = {s_pos['mean']:.6g}/{s_pos['median']:.6g}/{s_pos['max']:.6g}")
        print(f"  rot mean/med/max = {s_rot['mean']:.6g}/{s_rot['median']:.6g}/{s_rot['max']:.6g}")
        print(f"  all mean/med/max = {s_all['mean']:.6g}/{s_all['median']:.6g}/{s_all['max']:.6g}")

        results[int(style_id)] = {
            "style_id": int(style_id),
            "n_assigned": int(len(demos_orig)),
            "n_used": int(s_all["n"]),
            "drops": dict(drops),
            "rmse_pos": s_pos,
            "rmse_rot": s_rot,
            "rmse_all": s_all,
            "used_demo_indices_original": used_demos_orig,
        }

        # -----------------------
        # plotting per-style (FULL PHASE)
        # -----------------------
        if args.plot and s_all["n"] > 0:
            plot_dir = Path(args.plot_dir) if args.plot_dir is not None else Path("./spromp_eval_full_plots")
            plot_dir.mkdir(parents=True, exist_ok=True)

            # choose overlay demo:
            overlay_orig = None
            if args.plot_demo is not None:
                cand = int(args.plot_demo)
                if cand in used_demos_orig:
                    overlay_orig = cand
            if overlay_orig is None:
                overlay_orig = used_demos_orig[0]

            y_overlay = _extract_demo_full_phase(Xp, Wp, ptrp, demo_orig=overlay_orig)

            out1 = plot_dir / f"style_{style_id:02d}_overlay_demo_orig_{overlay_orig:03d}.png"
            plot_overlay_6d(
                y=y_overlay,
                y_mean=y_mean,
                title=f"style {style_id} | demo_orig={overlay_orig} | FULL PHASE",
                out_png=out1,
                t=np.linspace(0.0, 1.0, y_overlay.shape[0], dtype=np.float64),
            )
            print(f"[plot] {out1}")

            # demo mean ± std: need common T; resample demos to same length (use y_mean length as reference)
            T_ref = int(y_mean.shape[0])
            demos_rs = [resample(d, T_ref) if d.shape[0] != T_ref else d for d in demos_full]
            out2 = plot_dir / f"style_{style_id:02d}_demo_mean_std_fullphase.png"
            plot_demo_mean_std_6d(
                demos=demos_rs,
                t=np.linspace(0.0, 1.0, T_ref, dtype=np.float64),
                out_png=out2,
                title=f"style {style_id}: demo mean ± std (FULL PHASE, resampled to T={T_ref})",
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
                    title=f"style {style_id}: ProMP mean ± {args.conf_scale}*std (FULL PHASE)",
                    conf_scale=float(args.conf_scale),
                )
                print(f"[plot] {out3}")

    # summary
    print("\n[summary] styles evaluated:")
    for sid in sorted(results.keys()):
        r = results[sid]
        print(
            f"  style {sid:02d}: used {r['n_used']}/{r['n_assigned']} | "
            f"rmse_all mean/med/max={r['rmse_all']['mean']:.4g}/{r['rmse_all']['median']:.4g}/{r['rmse_all']['max']:.4g}"
        )

    # -----------------------
    # Global overall RMSE (ALL used demos across ALL styles)
    # -----------------------
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
    

if __name__ == "__main__":
    main()