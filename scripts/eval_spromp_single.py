#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_spromp_single.py (MODE A ONLY, robust)

Evaluate + plot style-conditioned ProMP on FULL PHASE trajectories.
- Window-after-contact is used ONLY for BGMM style assignment (already done in style_pkl),
  and is NOT used for evaluation or plotting.

Supports multiple spromp_single.pkl formats:
- per-style entry contains:
    - "promp" (pickled ProMP object), OR
    - "weight_mean" + "weight_cov" (reconstruct via from_weight_distribution), OR
    - "y_mean" (+ optional "y_var") precomputed.

Example:
  python3 eval_spromp_single.py \
    --pkl /home/sungboo/rb10_control/dataset/spromp_single.pkl \
    --style_pkl /home/sungboo/rb10_control/dataset/test_bgmm.pkl \
    --plot --plot_dir /home/sungboo/rb10_control/images/spromp_eval_single
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

from movement_primitives.promp import ProMP


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
    demos: List[np.ndarray],  # list of (T,6) (already aligned to same T)
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
# pickle helpers
# -----------------------------
def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Pickle must contain a dict, got {type(obj)}")
    return obj


def _get_drop_set(style_bgmm: dict) -> set:
    dd = style_bgmm.get("drop_demos", [])
    if isinstance(dd, (list, tuple, np.ndarray)):
        return set(int(x) for x in list(dd))
    return set()


def _build_orig_to_style_map(style_bgmm: dict) -> Dict[int, int]:
    if "used_demo_indices_original" not in style_bgmm or "labels" not in style_bgmm:
        raise ValueError("style_pkl must contain used_demo_indices_original and labels.")
    used_orig = np.asarray(style_bgmm["used_demo_indices_original"], dtype=np.int64).reshape(-1)
    labels = np.asarray(style_bgmm["labels"], dtype=np.int64).reshape(-1)
    if used_orig.shape[0] != labels.shape[0]:
        raise ValueError("style_pkl: used_demo_indices_original and labels length mismatch.")
    return {int(o): int(l) for o, l in zip(used_orig.tolist(), labels.tolist())}


# -----------------------------
# robust style table extraction
# -----------------------------
def _looks_like_style_entry(v: Any) -> bool:
    if not isinstance(v, dict):
        return False
    # style entry could have any of these
    keys = set(v.keys())
    if "promp" in keys:
        return True
    if ("weight_mean" in keys and "weight_cov" in keys):
        return True
    if "y_mean" in keys:
        return True
    return False


def _coerce_style_table(sp: dict) -> Dict[int, dict]:
    """
    Find {style_id: entry} somewhere in sp payload.
    - First try common keys.
    - Then search one level deep.
    - Then search two levels deep.
    """
    cand_keys = ["library", "styles", "style_models", "models", "style_table"]
    for ck in cand_keys:
        if ck in sp and isinstance(sp[ck], dict) and len(sp[ck]) > 0:
            table = {}
            ok = True
            for k, v in sp[ck].items():
                try:
                    kk = int(k)
                except Exception:
                    ok = False
                    break
                if not isinstance(v, dict):
                    ok = False
                    break
                table[kk] = v
            if ok and len(table) > 0:
                return table

    # search 1-level deep
    for _, v in sp.items():
        if isinstance(v, dict) and len(v) > 0:
            # if keys are int-like and values look like style entries
            table = {}
            for k2, v2 in v.items():
                try:
                    kk = int(k2)
                except Exception:
                    table = {}
                    break
                if not _looks_like_style_entry(v2):
                    table = {}
                    break
                table[kk] = v2
            if len(table) > 0:
                return table

    # search 2-level deep
    for _, v in sp.items():
        if isinstance(v, dict):
            for _, v2 in v.items():
                if isinstance(v2, dict) and len(v2) > 0:
                    table = {}
                    for k3, v3 in v2.items():
                        try:
                            kk = int(k3)
                        except Exception:
                            table = {}
                            break
                        if not _looks_like_style_entry(v3):
                            table = {}
                            break
                        table[kk] = v3
                    if len(table) > 0:
                        return table

    raise ValueError(
        "spromp.pkl has no style table. "
        f"Top-level keys={list(sp.keys())[:50]} "
        f"(tried common keys={cand_keys} + nested scan)"
    )


def _infer_base_promp(sp: dict, style_table: Dict[int, dict]) -> Optional[ProMP]:
    """
    If train saved a base ProMP somewhere, use it for n_dims/n_basis.
    Otherwise infer from first available style entry.
    """
    for k in ["base_promp", "base", "global_promp", "promp"]:
        if k in sp and isinstance(sp[k], ProMP):
            return sp[k]
        if k in sp and isinstance(sp[k], dict) and "promp" in sp[k] and isinstance(sp[k]["promp"], ProMP):
            return sp[k]["promp"]

    # infer from style entry
    for _, ent in style_table.items():
        if isinstance(ent, dict) and "promp" in ent and isinstance(ent["promp"], ProMP):
            return ent["promp"]
        if isinstance(ent, dict) and ("weight_mean" in ent):
            wm = np.asarray(ent["weight_mean"], dtype=np.float64).reshape(-1)
            # assume 6D if divisible by 6; else fallback to 6 anyway
            if wm.size % 6 == 0:
                n_basis = wm.size // 6
                base = ProMP(n_dims=6, n_weights_per_dim=int(n_basis))
                return base
    return None


def _entry_to_mean_var(
    entry: dict,
    T: int,
    base_promp: Optional[ProMP],
    conf_mc_samples: int = 0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Return (y_mean(T,6), y_var(T,6) or None) for this style entry.
    Priority:
      1) y_mean/y_var in entry
      2) promp object in entry
      3) weight_mean/weight_cov in entry (reconstruct ProMP)
    """
    t = np.linspace(0.0, 1.0, int(T), dtype=np.float64)

    # 1) already computed
    if "y_mean" in entry:
        y_mean = np.asarray(entry["y_mean"], dtype=np.float64)
        if y_mean.shape[0] != T:
            y_mean = resample(y_mean, T)
        y_var = None
        if "y_var" in entry and entry["y_var"] is not None:
            y_var = np.asarray(entry["y_var"], dtype=np.float64)
            if y_var.shape[0] != T:
                y_var = resample(y_var, T)
        return y_mean, y_var

    # 2) promp object
    if "promp" in entry and isinstance(entry["promp"], ProMP):
        p: ProMP = entry["promp"]
        y_mean = np.asarray(p.mean_trajectory(t), dtype=np.float64)
        y_var = None
        try:
            y_var = np.asarray(p.var_trajectory(t), dtype=np.float64)
        except Exception:
            y_var = None
        return y_mean, y_var

    # 3) weight distribution -> reconstruct
    if ("weight_mean" in entry) and ("weight_cov" in entry):
        wm = np.asarray(entry["weight_mean"], dtype=np.float64).reshape(-1)
        wc = np.asarray(entry["weight_cov"], dtype=np.float64)

        if base_promp is not None:
            n_dims = int(base_promp.n_dims)
            n_basis = int(base_promp.n_weights_per_dim)
        else:
            n_dims = 6
            if wm.size % n_dims != 0:
                raise ValueError(f"Cannot infer n_basis: weight_mean len={wm.size} not divisible by n_dims={n_dims}")
            n_basis = wm.size // n_dims

        p = ProMP(n_dims=n_dims, n_weights_per_dim=n_basis).from_weight_distribution(wm, wc)
        y_mean = np.asarray(p.mean_trajectory(t), dtype=np.float64)
        y_var = None
        try:
            y_var = np.asarray(p.var_trajectory(t), dtype=np.float64)
        except Exception:
            y_var = None
        return y_mean, y_var

    raise ValueError(f"Style entry has no usable fields: keys={list(entry.keys())}")


# -----------------------------
# NPZ helpers
# -----------------------------
def _extract_demo_full_phase(
    Xp: np.ndarray,
    Wp: np.ndarray,
    ptrp: np.ndarray,
    demo_orig: int,
) -> np.ndarray:
    sp, ep = int(ptrp[demo_orig]), int(ptrp[demo_orig + 1])
    xyz = np.asarray(Xp[sp:ep], dtype=np.float64)
    rot = np.asarray(Wp[sp:ep], dtype=np.float64)
    if xyz.shape[0] != rot.shape[0]:
        raise ValueError(f"X/W length mismatch for demo {demo_orig}: {xyz.shape[0]} vs {rot.shape[0]}")
    return np.concatenate([xyz, rot], axis=1)


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="spromp_single.pkl")
    ap.add_argument("--style_pkl", required=True, help="BGMM style assignment pkl")
    ap.add_argument("--npz", default=None, help="Override NPZ path (default: spromp payload['source_npz'] if exists)")

    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_dir", default=None)
    ap.add_argument("--plot_demo", type=int, default=None,
                    help="Optional: ORIGINAL demo index to overlay (must belong to that style).")

    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--conf_scale", type=float, default=1.96)
    ap.add_argument("--strict_T_match", action="store_true",
                    help="If set, require model mean length == demo length; otherwise resample model mean/var to demo length.")
    args = ap.parse_args()

    sp_path = Path(args.pkl)
    style_path = Path(args.style_pkl)

    sp = _load_pickle(sp_path)
    style_bgmm = _load_pickle(style_path)

    style_table = _coerce_style_table(sp)
    base_promp = _infer_base_promp(sp, style_table)

    orig_to_style = _build_orig_to_style_map(style_bgmm)
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

    kept_original = [i for i in range(D0) if i not in drop_set]

    # nominal phase length
    T_phase_nom = int(ptrp[1] - ptrp[0])
    if T_phase_nom <= 1:
        raise ValueError(f"Bad nominal phase length T={T_phase_nom}")

    print(f"[info] spromp_single.pkl: {sp_path}")
    print(f"[info] style_pkl:        {style_path}")
    print(f"[info] npz:              {npz_path}")
    print(f"[info] demos original D0={D0}, drop={sorted(drop_set)}, kept={len(kept_original)}")
    print(f"[info] nominal T_phase={T_phase_nom}")
    print(f"[info] styles_in_model={sorted(style_table.keys())}")
    print(f"[info] eval mode: FULL_PHASE (window NOT used)")
    if base_promp is not None:
        print(f"[info] base promp inferred: n_dims={base_promp.n_dims}, n_basis={base_promp.n_weights_per_dim}")
    else:
        print("[warn] base promp not found; will infer n_basis from style weights when needed")

    # group demos by style (only those kept + labeled)
    per_style_demos_orig: Dict[int, List[int]] = {sid: [] for sid in style_table.keys()}
    n_unlabeled = 0
    for orig in kept_original:
        sid = int(orig_to_style.get(int(orig), -1))
        if sid in per_style_demos_orig:
            per_style_demos_orig[sid].append(int(orig))
        else:
            n_unlabeled += 1
    if n_unlabeled > 0:
        print(f"[warn] {n_unlabeled} kept demos have no style label (or style not in model) -> ignored")

    results: Dict[int, dict] = {}
	# -----------------------
    # Global RMSE accumulators (ALL used demos across ALL styles)
    # -----------------------
    g_pos, g_rot, g_all = [], [], []


    for style_id in sorted(style_table.keys()):
        entry = style_table[style_id]
        demos_orig = per_style_demos_orig.get(style_id, [])
        if len(demos_orig) == 0:
            print(f"\n[style {style_id}] no demos assigned -> skip")
            continue

        rmse_pos, rmse_rot, rmse_all = [], [], []
        demos_full: List[np.ndarray] = []
        used_demos_orig: List[int] = []
        drops: Dict[str, int] = {}

        # compute model mean/var at nominal T first (for mean/std plots)
        y_mean_nom, y_var_nom = _entry_to_mean_var(entry, T=T_phase_nom, base_promp=base_promp)

        for orig in demos_orig:
            y_full = _extract_demo_full_phase(Xp, Wp, ptrp, demo_orig=orig)

            if y_full.shape[0] < int(args.min_len):
                drops["min_len"] = drops.get("min_len", 0) + 1
                continue
            if not _finite(y_full):
                drops["nonfinite_full"] = drops.get("nonfinite_full", 0) + 1
                continue

            # model mean for this demo length
            if args.strict_T_match and (y_mean_nom.shape[0] != y_full.shape[0]):
                drops["T_mismatch_strict"] = drops.get("T_mismatch_strict", 0) + 1
                continue

            y_mean = y_mean_nom if y_mean_nom.shape[0] == y_full.shape[0] else resample(y_mean_nom, y_full.shape[0])

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
            plot_dir = Path(args.plot_dir) if args.plot_dir is not None else Path("./spromp_eval_single_full_plots")
            plot_dir.mkdir(parents=True, exist_ok=True)

            # choose overlay demo (must be used)
            overlay_orig = None
            if args.plot_demo is not None and int(args.plot_demo) in used_demos_orig:
                overlay_orig = int(args.plot_demo)
            if overlay_orig is None:
                overlay_orig = used_demos_orig[0]

            y_overlay = _extract_demo_full_phase(Xp, Wp, ptrp, demo_orig=overlay_orig)
            # model mean at overlay length
            y_mean_ov = y_mean_nom if y_mean_nom.shape[0] == y_overlay.shape[0] else resample(y_mean_nom, y_overlay.shape[0])

            out1 = plot_dir / f"style_{style_id:02d}_overlay_demo_orig_{overlay_orig:03d}.png"
            plot_overlay_6d(
                y=y_overlay,
                y_mean=y_mean_ov,
                title=f"style {style_id} | demo_orig={overlay_orig} | FULL PHASE",
                out_png=out1,
            )
            print(f"[plot] {out1}")

            # demo mean ± std: resample demos to nominal T for aggregate plot
            T_ref = int(y_mean_nom.shape[0])
            t_ref = np.linspace(0.0, 1.0, T_ref, dtype=np.float64)
            demos_rs = [resample(d, T_ref) if d.shape[0] != T_ref else d for d in demos_full]

            out2 = plot_dir / f"style_{style_id:02d}_demo_mean_std_fullphase.png"
            plot_demo_mean_std_6d(
                demos=demos_rs,
                t=t_ref,
                out_png=out2,
                title=f"style {style_id}: demo mean ± std (FULL PHASE, resampled to T={T_ref})",
            )
            print(f"[plot] {out2}")

            # model mean ± conf (if var available)
            if y_var_nom is not None:
                out3 = plot_dir / f"style_{style_id:02d}_promp_mean_conf_fullphase.png"
                plot_mean_conf_6d(
                    t=t_ref,
                    y_mean=y_mean_nom,
                    y_var=y_var_nom,
                    out_png=out3,
                    title=f"style {style_id}: ProMP mean ± {args.conf_scale}*std (FULL PHASE)",
                    conf_scale=float(args.conf_scale),
                )
                print(f"[plot] {out3}")
	
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
        print(f"[sProMP-single] N={s_all_g['n']}")
        print(f"  pos mean±std = {s_pos_g['mean']:.6g} ± {s_pos_g['std']:.6g}")
        print(f"  rot mean±std = {s_rot_g['mean']:.6g} ± {s_rot_g['std']:.6g}")
        print(f"  all mean±std = {s_all_g['mean']:.6g} ± {s_all_g['std']:.6g}")
    else:
        print("(no samples)")

if __name__ == "__main__":
    main()