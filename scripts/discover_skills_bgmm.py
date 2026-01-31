#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BGMM clustering on phase-aligned demos without using skill_id.

- Load NPZ: X_phase_crop (N,3), W_phase_crop (N,3), demo_ptr_phase (D+1,)
- Optional: drop demos (keep consistent with your training pipeline)
- Build per-demo Y(t) = [xyz, rotvec] (T,6) from phase-aligned arrays
- Flatten to x_i in R^(6T)
- (Optional) standardize features
- Fit BayesianGaussianMixture (Dirichlet Process-style)
- Save labels + responsibilities

Example:
  python cluster_bgmm_flatten.py \
    --npz /home/sungboo/rb10_control/dataset/demo_20260122_final.npz \
    --out /home/sungboo/rb10_control/dataset/bgmm_cluster.npz \
    --drop_demos 36 57 98 202 \
    --n_components 20 \
    --standardize
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required. Install it, e.g.\n"
        "  pip install scikit-learn\n"
        f"ImportError: {e}"
    )


def filter_phase_demos_by_index(
    X_phase: np.ndarray,
    W_phase: np.ndarray,
    ptr_phase: np.ndarray,
    drop_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop demos by phase-demo index, rebuild concatenated arrays and new ptr.
    Returns:
      Xp_new, Wp_new, ptrp_new, kept_demo_indices (original indices kept)
    """
    drop_ids = sorted(set(int(i) for i in drop_ids))
    D = int(ptr_phase.shape[0] - 1)
    keep = [i for i in range(D) if i not in drop_ids]
    kept_demo_indices = np.asarray(keep, dtype=np.int64)

    if len(keep) == D:
        return X_phase, W_phase, ptr_phase, kept_demo_indices

    Xp_new, Wp_new = [], []
    ptrp_new = [0]

    for i in keep:
        sp, ep = int(ptr_phase[i]), int(ptr_phase[i + 1])
        Xp_new.append(X_phase[sp:ep])
        Wp_new.append(W_phase[sp:ep])
        ptrp_new.append(ptrp_new[-1] + (ep - sp))

    Xp_new = np.concatenate(Xp_new, axis=0) if len(Xp_new) else np.zeros((0, 3), dtype=np.float64)
    Wp_new = np.concatenate(Wp_new, axis=0) if len(Wp_new) else np.zeros((0, 3), dtype=np.float64)

    return Xp_new, Wp_new, np.asarray(ptrp_new, dtype=np.int64), kept_demo_indices


def build_flattened_matrix(
    X_phase: np.ndarray,
    W_phase: np.ndarray,
    ptr_phase: np.ndarray,
    min_len: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build:
      X_flat: (D_used, 6*T) flattened [xyz, rotvec] per demo
      used_demo_indices_phase: (D_used,) indices in the *current* ptr_phase indexing
    Assumes all phase demos have same T; otherwise raises.
    """
    X_phase = np.asarray(X_phase, dtype=np.float64)
    W_phase = np.asarray(W_phase, dtype=np.float64)
    ptr_phase = np.asarray(ptr_phase, dtype=np.int64)

    D = int(ptr_phase.shape[0] - 1)
    if D <= 0:
        raise ValueError("No demos found (D<=0).")

    lens = np.array([int(ptr_phase[i + 1] - ptr_phase[i]) for i in range(D)], dtype=np.int64)
    # enforce equal length for phase-aligned
    T = int(lens[0])
    if not np.all(lens == T):
        # if you want to allow mismatch, you'd need resampling per demo BEFORE flattening.
        bad = np.where(lens != T)[0][:20]
        raise ValueError(
            f"Phase demos must share the same length T. Got varying lengths.\n"
            f"First T={T}, mismatched demo indices (first20)={bad.tolist()}, lens(first20)={lens[bad].tolist()}"
        )

    if T < min_len:
        raise ValueError(f"All demos are shorter than min_len={min_len}. T={T}")

    X_flat_list = []
    used = []

    for i in range(D):
        sp, ep = int(ptr_phase[i]), int(ptr_phase[i + 1])
        y = np.concatenate([X_phase[sp:ep], W_phase[sp:ep]], axis=1)  # (T,6)
        if y.shape[0] < min_len:
            continue
        X_flat_list.append(y.reshape(-1))  # (6T,)
        used.append(i)

    if len(X_flat_list) == 0:
        raise ValueError("No demos remained after min_len filtering.")

    X_flat = np.stack(X_flat_list, axis=0)
    used_demo_indices_phase = np.asarray(used, dtype=np.int64)
    return X_flat, used_demo_indices_phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/sungboo/rb10_control/dataset/demo_20260122_final.npz")
    ap.add_argument("--out", required=True)

    # keep consistent with your pipeline
    ap.add_argument("--drop_demos", type=int, nargs="*", default=[36, 57, 98, 202])

    ap.add_argument("--min_len", type=int, default=10)

    # BGMM params
    ap.add_argument("--n_components", type=int, default=8, help="Upper bound on clusters (truncation level).")
    ap.add_argument("--covariance_type", choices=["full", "diag", "tied", "spherical"], default="full")
    ap.add_argument("--weight_concentration_prior_type", choices=["dirichlet_process", "dirichlet_distribution"],
                    default="dirichlet_process")
    ap.add_argument("--weight_concentration_prior", type=float, default=None,
                    help="If None, sklearn default is used. Smaller => fewer active clusters (DP).")

    ap.add_argument("--max_iter", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--reg_covar", type=float, default=1e-6)

    ap.add_argument("--standardize", action="store_true", help="Strongly recommended for flatten features.")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_path = Path(args.out)

    data = np.load(npz_path, allow_pickle=True)
    for k in ("X_phase_crop", "W_phase_crop", "demo_ptr_phase"):
        if k not in data:
            raise KeyError(f"NPZ must contain key: {k}")

    Xp = np.asarray(data["X_phase_crop"], dtype=np.float64)
    Wp = np.asarray(data["W_phase_crop"], dtype=np.float64)
    ptrp = np.asarray(data["demo_ptr_phase"], dtype=np.int64)

    D0 = int(ptrp.shape[0] - 1)
    drop_demos = sorted(set(int(i) for i in args.drop_demos))

    # apply drop (phase-only)
    kept_original = np.arange(D0, dtype=np.int64)
    if drop_demos:
        bad = [i for i in drop_demos if i < 0 or i >= D0]
        if bad:
            raise ValueError(f"--drop_demos out-of-range: {bad} (valid 0..{D0-1})")

        Xp, Wp, ptrp, kept_original = filter_phase_demos_by_index(Xp, Wp, ptrp, drop_demos)

    # build flatten features
    X_flat, used_demo_indices_phase = build_flattened_matrix(
        X_phase=Xp,
        W_phase=Wp,
        ptr_phase=ptrp,
        min_len=int(args.min_len),
    )

    # used indices mapped back to *original phase-demo indices*
    used_demo_indices_original = kept_original[used_demo_indices_phase]

    # standardize
    scaler = None
    X_fit = X_flat
    if args.standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_fit = scaler.fit_transform(X_flat)

    # fit BGMM
    bgmm = BayesianGaussianMixture(
        n_components=int(args.n_components),
        covariance_type=str(args.covariance_type),
        weight_concentration_prior_type=str(args.weight_concentration_prior_type),
        weight_concentration_prior=args.weight_concentration_prior,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        reg_covar=float(args.reg_covar),
        init_params="kmeans",
        random_state=int(args.seed),
    )
    bgmm.fit(X_fit)

    probs = bgmm.predict_proba(X_fit)  # (N, K)
    labels = np.argmax(probs, axis=1).astype(np.int64)

    # summary
    K = probs.shape[1]
    counts = np.bincount(labels, minlength=K)
    active = np.where(counts > 0)[0].tolist()

    print(f"[info] npz: {npz_path}")
    print(f"[info] demos (original): {D0}, dropped: {drop_demos}, kept: {len(kept_original)}")
    print(f"[info] used demos (after min_len): {X_fit.shape[0]}")
    print(f"[info] feature dim: {X_fit.shape[1]} (= 6*T)")
    print(f"[BGMM] n_components={K}, active_clusters={len(active)}")
    for k in active:
        print(f"  cluster {k:02d}: n={int(counts[k])}")

    # save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        source_npz=str(npz_path),
        drop_demos=np.asarray(drop_demos, dtype=np.int64),
        used_demo_indices_phase=used_demo_indices_phase,              # index in post-drop ptrp
        used_demo_indices_original=used_demo_indices_original,        # original phase-demo index before drop
        X_flat=X_flat.astype(np.float32),
        standardize=bool(args.standardize),
        labels=labels,
        probs=probs.astype(np.float32),
        # bgmm params to reproduce
        n_components=int(args.n_components),
        covariance_type=str(args.covariance_type),
        weight_concentration_prior_type=str(args.weight_concentration_prior_type),
        weight_concentration_prior=(np.nan if args.weight_concentration_prior is None else float(args.weight_concentration_prior)),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        reg_covar=float(args.reg_covar),
        seed=int(args.seed),
    )
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
