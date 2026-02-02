#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_spromp.py (STYLE OVERLAY ON SINGLE ProMP SPACE)

Goal (your requested change):
- Do NOT train separate ProMPs per style via imitate().
- Instead:
    1) Use a single shared ProMP representation (same basis; same n_dims/n_basis).
    2) For each demo, compute weight vector w_i via linear regression: ProMP.weights(t, y_i).
    3) For each style, estimate a weight-space distribution p(w|style)=N(mu_s, Sigma_s)
       (default: mu_s style-specific, Sigma_shared pooled within-style).
    4) Optionally, instantiate style-conditioned ProMP objects via from_weight_distribution
       (but main artifact is mu/cov per style; you can reconstruct any time).

Inputs:
- NPZ (phase-aligned):
    X_phase_crop: (N_phase, 3)
    W_phase_crop: (N_phase, 3)
    demo_ptr_phase: (D+1,)
  Optional (for contact-start mapping if you ever want window features; not used for training by default):
    crop_s, crop_e: (D,) raw->crop mapping
    contact_start_idx: (D,) raw indices
- BGMM PKL from discover_styles_bgmm.py:
    labels: (N_used,)
    used_demo_indices_original: (N_used,)  # indices w.r.t original demos in NPZ
    args.window_after_contact exists but is used only for style discovery (not required here)

Outputs:
- PKL containing:
    - base ProMP hyperparams (n_dims, n_basis, centers)
    - style -> (mu_w, Sigma_w or Sigma_shared)
    - (optional cached) style -> y_mean/y_var for convenience
    - metadata / drops

Example:
  python3 train_spromp.py \
    --npz /home/sungboo/rb10_control/dataset/demo_20260122_final.npz \
    --style_pkl /home/sungboo/rb10_control/dataset/test_bgmm.pkl \
    --out /home/sungboo/rb10_control/dataset/spromp.pkl \
    --n_basis 25 --min_demos 5 --cov_mode pooled --ridge 1e-6 --cache_traj
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from movement_primitives.promp import ProMP


# -----------------------------
# utils
# -----------------------------
def _finite(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(x))))


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


def _as_diag_var(v: np.ndarray) -> np.ndarray:
    """
    Accept:
      - (T,D) diag var
      - (T,D,D) full cov -> diag
      - (D,) -> (1,D)
      - (D,D) -> (1,D) diag
    Return (T,D).
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        return v.reshape(1, -1)
    if v.ndim == 2:
        if v.shape[0] == v.shape[1]:
            return np.diag(v).reshape(1, -1)
        return v
    if v.ndim == 3:
        return np.einsum("tii->ti", v)
    raise ValueError(f"Unsupported var/cov shape: {v.shape}")


def trajectory_mean_and_var(promp: ProMP, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This ProMP implementation supports var_trajectory directly.
    Return:
      y_mean: (T,D)
      y_var : (T,D) diag var
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y_mean = np.asarray(promp.mean_trajectory(t), dtype=np.float64)
    y_var = np.asarray(promp.var_trajectory(t), dtype=np.float64)
    y_var = _as_diag_var(y_var)
    if y_var.shape[0] == 1 and y_mean.shape[0] > 1:
        y_var = np.tile(y_var, (y_mean.shape[0], 1))
    return y_mean, y_var


def _extract_demo_phase_traj(
    Xp: np.ndarray,
    Wp: np.ndarray,
    ptrp: np.ndarray,
    demo_orig: int,
    T_phase: int,
) -> np.ndarray:
    """Return y(T,6) for original demo index."""
    sp, ep = int(ptrp[demo_orig]), int(ptrp[demo_orig + 1])
    xyz = np.asarray(Xp[sp:ep], dtype=np.float64)
    rot = np.asarray(Wp[sp:ep], dtype=np.float64)
    if xyz.shape[0] != T_phase or rot.shape[0] != T_phase:
        y = np.concatenate([xyz, rot], axis=1)
        y = resample(y, T_phase)
        return y
    return np.concatenate([xyz, rot], axis=1)


def _cov_shrink_to_diag(S: np.ndarray, alpha: float) -> np.ndarray:
    """(1-alpha)*S + alpha*diag(S). alpha in [0,1]."""
    S = np.asarray(S, dtype=np.float64)
    if alpha <= 0.0:
        return S
    d = np.diag(np.diag(S))
    return (1.0 - float(alpha)) * S + float(alpha) * d


def _ensure_psd(S: np.ndarray, eps: float) -> np.ndarray:
    """Make covariance numerically PSD by adding eps*I."""
    S = np.asarray(S, dtype=np.float64)
    S = 0.5 * (S + S.T)
    S = S + float(eps) * np.eye(S.shape[0], dtype=np.float64)
    return S


def _pooled_within_cov(W: np.ndarray, labels: np.ndarray, unique_styles: List[int]) -> Tuple[np.ndarray, int]:
    """
    Compute pooled within-style covariance:
      Sum_s Sum_{i in s} (w_i - mu_s)(w_i - mu_s)^T / N_total
    Returns (Sigma_within, N_total_used_for_cov)
    """
    W = np.asarray(W, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    Wdim = int(W.shape[1])

    Ssum = np.zeros((Wdim, Wdim), dtype=np.float64)
    Ntot = 0

    for s in unique_styles:
        idx = np.where(labels == int(s))[0]
        if idx.size <= 1:
            continue
        Ws = W[idx]
        mu = np.mean(Ws, axis=0, keepdims=True)
        C = Ws - mu
        Ssum += C.T @ C
        Ntot += int(Ws.shape[0])

    if Ntot <= 0:
        # fallback: overall covariance
        C = W - np.mean(W, axis=0, keepdims=True)
        Ssum = C.T @ C
        Ntot = int(W.shape[0])

    Sigma = Ssum / max(Ntot, 1)
    return Sigma, Ntot


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", default="/home/sungboo/rb10_control/dataset/demo_20260122_final.npz")
    ap.add_argument("--style_pkl", required=True, help="BGMM output pkl from discover_styles_bgmm.py")
    ap.add_argument("--out", required=True)

    ap.add_argument("--n_basis", type=int, default=25)
    ap.add_argument("--min_demos", type=int, default=5)

    # weight regression regularization (ProMP.weights)
    ap.add_argument("--ridge", type=float, default=1e-6, help="Ridge lambda for ProMP.weights() regression per demo.")

    # covariance strategy in weight space
    ap.add_argument(
        "--cov_mode",
        choices=["pooled", "per_style"],
        default="pooled",
        help="pooled: mu per style + Sigma_shared pooled within-style (recommended). "
             "per_style: estimate Sigma_s per style (requires enough demos).",
    )
    ap.add_argument("--shrink_diag", type=float, default=0.0,
                    help="Shrink covariance towards diagonal: S <- (1-a)S + a*diag(S). a in [0,1].")
    ap.add_argument("--cov_eps", type=float, default=1e-6, help="Add eps*I for numerical PSD.")

    # optional cached trajectories for convenience (not needed for execution)
    ap.add_argument("--cache_traj", action="store_true",
                    help="Cache y_mean/y_var per style in payload (computed from mu/cov).")
    ap.add_argument("--standardize_var", action="store_true",
                    help="If caching y_var, normalize it by per-dim median for nicer plots (diagnostics only).")

    args = ap.parse_args()

    npz_path = Path(args.npz)
    style_pkl_path = Path(args.style_pkl)
    out_path = Path(args.out)

    data = np.load(npz_path, allow_pickle=True)
    for k in ("X_phase_crop", "W_phase_crop", "demo_ptr_phase"):
        if k not in data:
            raise KeyError(f"NPZ must contain key: {k}")

    Xp = np.asarray(data["X_phase_crop"], dtype=np.float64)
    Wp = np.asarray(data["W_phase_crop"], dtype=np.float64)
    ptrp = np.asarray(data["demo_ptr_phase"], dtype=np.int64)

    D = int(ptrp.shape[0] - 1)
    if D <= 0:
        raise ValueError("No demos in NPZ.")
    T_phase = int(ptrp[1] - ptrp[0])
    if T_phase <= 1:
        raise ValueError(f"Bad phase length T={T_phase}")

    # load BGMM payload
    with open(style_pkl_path, "rb") as f:
        bg = pickle.load(f)

    if "labels" not in bg or "used_demo_indices_original" not in bg:
        raise KeyError("style_pkl must contain: labels, used_demo_indices_original")

    labels_used = np.asarray(bg["labels"], dtype=np.int64).reshape(-1)
    used_orig = np.asarray(bg["used_demo_indices_original"], dtype=np.int64).reshape(-1)
    if labels_used.shape[0] != used_orig.shape[0]:
        raise ValueError(f"labels len {labels_used.shape[0]} != used_demo_indices_original len {used_orig.shape[0]}")

    unique_styles = sorted(np.unique(labels_used).tolist())
    n_styles = len(unique_styles)

    # -------------------------
    # Collect demo trajectories + weights (single shared ProMP space)
    # -------------------------
    n_dims = 6
    promp_base = ProMP(n_dims=n_dims, n_weights_per_dim=int(args.n_basis))
    t = np.linspace(0.0, 1.0, T_phase, dtype=np.float64)

    W_list: List[np.ndarray] = []
    S_list: List[int] = []
    used_demo_list: List[int] = []
    dropped: List[Dict] = []

    for s, demo_orig in zip(labels_used.tolist(), used_orig.tolist()):
        demo_orig = int(demo_orig)
        s = int(s)

        if demo_orig < 0 or demo_orig >= D:
            dropped.append(dict(demo_orig=demo_orig, style_id=s, reason="demo_index_oob"))
            continue

        y_full = _extract_demo_phase_traj(Xp, Wp, ptrp, demo_orig=demo_orig, T_phase=T_phase)
        if not _finite(y_full):
            dropped.append(dict(demo_orig=demo_orig, style_id=s, reason="nonfinite_y"))
            continue
        if y_full.shape != (T_phase, n_dims):
            dropped.append(dict(demo_orig=demo_orig, style_id=s, reason="bad_shape", got=list(y_full.shape)))
            continue

        # weights regression in the shared basis space
        w = promp_base.weights(t, y_full, lmbda=float(args.ridge))
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if w.shape[0] != int(promp_base.n_weights):
            dropped.append(dict(demo_orig=demo_orig, style_id=s, reason="bad_w_dim", got=int(w.shape[0])))
            continue
        if not _finite(w):
            dropped.append(dict(demo_orig=demo_orig, style_id=s, reason="nonfinite_w"))
            continue

        W_list.append(w)
        S_list.append(s)
        used_demo_list.append(demo_orig)

    if len(W_list) == 0:
        raise RuntimeError("No usable demos after filtering.")

    W_all = np.stack(W_list, axis=0)  # (N_used_eff, Wdim)
    S_all = np.asarray(S_list, dtype=np.int64).reshape(-1)
    used_demo_arr = np.asarray(used_demo_list, dtype=np.int64).reshape(-1)

    # -------------------------
    # Enforce min_demos per style (filter styles and samples)
    # -------------------------
    style_counts: Dict[int, int] = {int(s): int(np.sum(S_all == int(s))) for s in unique_styles}
    active_styles = [int(s) for s in unique_styles if style_counts.get(int(s), 0) >= int(args.min_demos)]
    skipped_styles = [int(s) for s in unique_styles if int(s) not in set(active_styles)]

    if len(active_styles) == 0:
        raise RuntimeError(f"No styles have >= min_demos={args.min_demos}. counts={style_counts}")

    # filter samples to active styles only
    keep_mask = np.isin(S_all, np.asarray(active_styles, dtype=np.int64))
    W = W_all[keep_mask]
    S = S_all[keep_mask]
    used_demo_arr = used_demo_arr[keep_mask]

    # -------------------------
    # Estimate style-conditioned weight distributions
    # -------------------------
    Wdim = int(W.shape[1])

    # style means
    mu_style: Dict[int, np.ndarray] = {}
    Sigma_style: Dict[int, np.ndarray] = {}  # only for per_style mode
    per_style_used_demos: Dict[int, np.ndarray] = {}
    per_style_n: Dict[int, int] = {}

    for s in active_styles:
        idx = np.where(S == int(s))[0]
        Ws = W[idx]
        mu = np.mean(Ws, axis=0)
        mu_style[int(s)] = mu.astype(np.float64, copy=False)
        per_style_used_demos[int(s)] = used_demo_arr[idx].astype(np.int32, copy=False)
        per_style_n[int(s)] = int(Ws.shape[0])

    Sigma_shared = None
    n_cov_samples = 0

    if args.cov_mode == "pooled":
        Sigma_shared, n_cov_samples = _pooled_within_cov(W, S, active_styles)
        Sigma_shared = _cov_shrink_to_diag(Sigma_shared, float(args.shrink_diag))
        Sigma_shared = _ensure_psd(Sigma_shared, float(args.cov_eps))
    else:
        for s in active_styles:
            idx = np.where(S == int(s))[0]
            Ws = W[idx]
            if Ws.shape[0] < 2:
                # fallback to tiny diag
                S_s = float(args.cov_eps) * np.eye(Wdim, dtype=np.float64)
            else:
                C = Ws - np.mean(Ws, axis=0, keepdims=True)
                S_s = (C.T @ C) / max(int(Ws.shape[0]), 1)
                S_s = _cov_shrink_to_diag(S_s, float(args.shrink_diag))
                S_s = _ensure_psd(S_s, float(args.cov_eps))
            Sigma_style[int(s)] = S_s
        n_cov_samples = int(W.shape[0])

    # -------------------------
    # Optional cache: style y_mean/y_var for quick plots
    # -------------------------
    cache: Dict[int, Dict] = {}
    if args.cache_traj:
        for s in active_styles:
            cov = Sigma_shared if Sigma_shared is not None else Sigma_style[int(s)]
            promp_s = ProMP(n_dims=n_dims, n_weights_per_dim=int(args.n_basis)).from_weight_distribution(
                mu_style[int(s)], cov
            )
            y_mean, y_var = trajectory_mean_and_var(promp_s, t)

            if args.standardize_var and (y_var is not None):
                med = np.median(np.maximum(y_var, 1e-12), axis=0)
                y_var = y_var / np.maximum(med[None, :], 1e-12)

            cache[int(s)] = dict(
                t=t,
                y_mean=np.asarray(y_mean, dtype=np.float64),
                y_var=np.asarray(y_var, dtype=np.float64),
            )

    # -------------------------
    # Payload
    # -------------------------
    payload = dict(
        source_npz=str(npz_path),
        source_style_pkl=str(style_pkl_path),
        bgmm_args=bg.get("args", None),
        bgmm_active_clusters=bg.get("active_clusters", None),
        bgmm_cluster_counts=bg.get("cluster_counts", None),

        # shared ProMP space
        n_dims=int(n_dims),
        n_basis=int(args.n_basis),
        n_weights=int(promp_base.n_weights),
        centers=np.asarray(promp_base.centers, dtype=np.float64),
        T_phase=int(T_phase),
        t=np.asarray(t, dtype=np.float64),

        # style assignment (samples used)
        styles_present=unique_styles,
        styles_active=active_styles,
        styles_skipped=skipped_styles,
        style_counts_original=style_counts,
        used_demo_indices_original=np.asarray(used_demo_arr, dtype=np.int32),
        used_labels=np.asarray(S, dtype=np.int32),

        # learned style overlay in weight space
        cov_mode=str(args.cov_mode),
        ridge=float(args.ridge),
        shrink_diag=float(args.shrink_diag),
        cov_eps=float(args.cov_eps),

        mu_style={int(s): np.asarray(mu_style[int(s)], dtype=np.float64) for s in active_styles},
        Sigma_shared=None if Sigma_shared is None else np.asarray(Sigma_shared, dtype=np.float64),
        Sigma_style=None if args.cov_mode != "per_style" else {int(s): np.asarray(Sigma_style[int(s)], dtype=np.float64) for s in active_styles},

        per_style_used_demo_indices_original={int(s): per_style_used_demos[int(s)] for s in active_styles},
        per_style_n=per_style_n,

        # optional cached trajectories for plotting
        cache_traj=bool(args.cache_traj),
        traj_cache=cache if args.cache_traj else None,

        # bookkeeping
        dropped=dropped,
        args=vars(args),
        n_cov_samples=int(n_cov_samples),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    # -------------------------
    # Summary print
    # -------------------------
    print(f"[info] npz: {npz_path}")
    print(f"[info] style_pkl: {style_pkl_path}")
    print(f"[info] demos in npz: D={D}, T_phase={T_phase}")
    print(f"[info] bgmm used demos: {used_orig.shape[0]}")
    print(f"[info] usable demos after filtering: N={W.shape[0]} | dropped={len(dropped)}")
    print(f"[info] styles (unique labels in bgmm): {unique_styles}")
    print(f"[info] active styles (>=min_demos={args.min_demos}): {active_styles} | skipped={skipped_styles}")
    print(f"[info] cov_mode={args.cov_mode} | ridge={args.ridge} | shrink_diag={args.shrink_diag} | cov_eps={args.cov_eps}")
    if Sigma_shared is not None:
        print(f"[info] Sigma_shared built from pooled within-style cov | n_cov_samples={n_cov_samples}")
    if args.cache_traj:
        print(f"[info] cached trajectories for {len(cache)} styles")

    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()