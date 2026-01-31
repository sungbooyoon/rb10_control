#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BP-AR-HMM style clustering/segmentation on phase-aligned demos.

Input NPZ:
  - X_phase_crop (N,3)
  - W_phase_crop (N,3)
  - demo_ptr_phase (D+1,)
Optional:
  - phase_grid (T,) (not required)

What it does:
  1) Load phase-aligned demos as sequences: y_i = (T_i, 6) = [xyz, rotvec]
  2) (Optional) drop demos by index (same as your training pipeline)
  3) Fit an AR-HMM style model across multiple sequences:
     - backend=pyhsmm: sticky HDP-AR-HMM-ish (if available)
     - backend=ssm: ARHMM with fixed K (fallback)
  4) Outputs:
     - per-demo inferred state sequence z_i (length T_i)
     - per-demo "dominant state" label (argmax usage)
     - per-demo state-usage histogram (for secondary clustering if desired)

Notes:
  - "True BP-AR-HMM" (Beta Process AR-HMM) is not commonly packaged.
    In practice, sticky HDP-AR-HMM or large-K ARHMM is the typical substitute.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import List, Dict, Tuple, Optional

import numpy as np


# -----------------------------
# Utilities
# -----------------------------
def load_phase_demos(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    data = np.load(npz_path, allow_pickle=True)
    for k in ["X_phase_crop", "W_phase_crop", "demo_ptr_phase"]:
        if k not in data:
            raise KeyError(f"NPZ must contain key: {k}")

    Xp = np.asarray(data["X_phase_crop"], dtype=np.float64)
    Wp = np.asarray(data["W_phase_crop"], dtype=np.float64)
    ptrp = np.asarray(data["demo_ptr_phase"], dtype=np.int64).reshape(-1)
    phase_grid = np.asarray(data["phase_grid"], dtype=np.float64).reshape(-1) if "phase_grid" in data else None

    if Xp.shape != Wp.shape:
        raise ValueError(f"X_phase_crop shape {Xp.shape} != W_phase_crop shape {Wp.shape}")
    if Xp.shape[1] != 3:
        raise ValueError(f"Expected X_phase_crop (N,3), got {Xp.shape}")
    if ptrp.ndim != 1 or ptrp.shape[0] < 2:
        raise ValueError(f"demo_ptr_phase must be (D+1,), got {ptrp.shape}")
    if int(ptrp[-1]) != int(Xp.shape[0]):
        raise ValueError(f"ptrp[-1]={int(ptrp[-1])} must equal N={Xp.shape[0]}")

    Y_all = np.concatenate([Xp, Wp], axis=1)  # (N,6)
    return Y_all, Xp, ptrp, phase_grid


def filter_demos_by_index(
    Y_all: np.ndarray,
    ptrp: np.ndarray,
    drop_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Drop demos by index and rebuild concatenated Y_all + ptrp."""
    drop_ids = sorted(set(int(i) for i in drop_ids))
    D = int(ptrp.shape[0] - 1)

    bad = [i for i in drop_ids if i < 0 or i >= D]
    if bad:
        raise ValueError(f"drop_demos out-of-range: {bad} (valid 0..{D-1})")

    keep = [i for i in range(D) if i not in drop_ids]
    if len(keep) == D:
        return Y_all, ptrp, keep

    Ys_new = []
    ptr_new = [0]
    for i in keep:
        s, e = int(ptrp[i]), int(ptrp[i + 1])
        seg = np.asarray(Y_all[s:e], dtype=np.float64)
        Ys_new.append(seg)
        ptr_new.append(ptr_new[-1] + (e - s))

    Y_new = np.concatenate(Ys_new, axis=0) if len(Ys_new) > 0 else np.zeros((0, 6), dtype=np.float64)
    return Y_new, np.asarray(ptr_new, dtype=np.int64), keep


def split_sequences(Y_all: np.ndarray, ptrp: np.ndarray, min_len: int = 5) -> Tuple[List[np.ndarray], List[int]]:
    """Return list of (T_i,6) sequences and their demo indices (phase-demo indices)."""
    D = int(ptrp.shape[0] - 1)
    seqs: List[np.ndarray] = []
    demo_ids: List[int] = []
    for i in range(D):
        s, e = int(ptrp[i]), int(ptrp[i + 1])
        y = np.asarray(Y_all[s:e], dtype=np.float64)
        if y.shape[0] < min_len:
            continue
        seqs.append(y)
        demo_ids.append(i)
    return seqs, demo_ids


def state_histogram(z: np.ndarray, K: int) -> np.ndarray:
    h = np.bincount(z.astype(np.int64), minlength=K).astype(np.float64)
    if h.sum() > 0:
        h /= h.sum()
    return h


# -----------------------------
# Backend: pyhsmm (if available)
# -----------------------------
def fit_arhmm_pyhsmm(
    seqs: List[np.ndarray],
    n_iters: int,
    K_max: int,
    seed: int,
    kappa: float,
) -> Dict:
    """
    Attempt a sticky HDP-AR-HMM-ish fit using pyhsmm if installed.
    WARNING: pyhsmm availability varies; this is best-effort.
    """
    rng = np.random.default_rng(seed)

    try:
        import pyhsmm  # type: ignore
        import pyhsmm.basic.distributions as distn  # type: ignore
        import pyhsmm.models as models  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pyhsmm backend requested but pyhsmm import failed.\n"
            "Install pyhsmm (may require older numpy/scipy) or use --backend ssm.\n"
            f"Import error: {repr(e)}"
        )

    # AR emissions: y_t = A y_{t-1} + eps
    # Use an AutoRegression distribution if available; otherwise approximate with Gaussian (not ideal).
    # Many forks differ; we try a couple names.
    ARDist = None
    for name in ["AutoRegression", "Regression", "AR"]:
        if hasattr(distn, name):
            ARDist = getattr(distn, name)
            break
    if ARDist is None:
        raise RuntimeError(
            "pyhsmm imported, but no AR distribution class found in pyhsmm.basic.distributions.\n"
            "Your pyhsmm fork may differ. Use --backend ssm as fallback."
        )

    D_obs = int(seqs[0].shape[1])

    # Priors (rough defaults; tune later)
    # NOTE: pyhsmm AR distribution constructor signatures vary across forks.
    # We'll try a tolerant creation approach.
    def make_ardist():
        # generic-ish prior
        # Some versions: AutoRegression(nu_0, S_0, M_0, K_0, affine, ...) etc.
        # We'll try common patterns.
        for kwargs in (
            dict(nu_0=D_obs + 2, S_0=np.eye(D_obs), M_0=np.zeros((D_obs, D_obs)), K_0=np.eye(D_obs)),
            dict(nu_0=D_obs + 2, S_0=np.eye(D_obs)),
            dict(),
        ):
            try:
                return ARDist(**kwargs)
            except TypeError:
                continue
        raise RuntimeError("Could not construct AR distribution in this pyhsmm fork.")

    obs_dists = [make_ardist() for _ in range(K_max)]

    # Sticky HDP-HMM-ish: many pyhsmm APIs expose WeakLimitStickyHDPHMM
    ModelCls = None
    for name in ["WeakLimitStickyHDPHMM", "WeakLimitHDPHMM"]:
        if hasattr(models, name):
            ModelCls = getattr(models, name)
            break
    if ModelCls is None:
        raise RuntimeError("pyhsmm models does not expose WeakLimitStickyHDPHMM/WeakLimitHDPHMM in this fork.")

    # Instantiate model
    # Typical args: obs_distns, alpha, gamma, kappa
    # We'll try sticky if possible.
    m = None
    for kwargs in (
        dict(obs_distns=obs_dists, alpha=5.0, gamma=5.0, kappa=float(kappa)),
        dict(obs_distns=obs_dists, alpha=5.0, gamma=5.0),
        dict(obs_distns=obs_dists),
    ):
        try:
            m = ModelCls(**kwargs)
            break
        except TypeError:
            continue
    if m is None:
        raise RuntimeError("Could not instantiate pyhsmm HDP-HMM model (signature mismatch).")

    # Add sequences
    for y in seqs:
        m.add_data(y)

    # Gibbs sampling / resampling
    for it in range(n_iters):
        m.resample_model()
        if (it + 1) % max(1, n_iters // 10) == 0:
            print(f"[pyhsmm] iter {it+1}/{n_iters}")

    # Collect inferred z
    z_list = [np.asarray(s.stateseq, dtype=np.int64) for s in m.states_list]

    # Estimate effective K = number of used states
    used_states = sorted(set(int(z) for zs in z_list for z in zs.tolist()))
    K_eff = len(used_states)

    return {
        "backend": "pyhsmm",
        "model": m,
        "z_list": z_list,
        "K_max": int(K_max),
        "K_eff": int(K_eff),
        "used_states": used_states,
    }


# -----------------------------
# Backend: ssm (fallback)
# -----------------------------
def fit_arhmm_ssm(
    seqs: List[np.ndarray],
    n_iters: int,
    K: int,
    seed: int,
    sticky: float,
) -> Dict:
    """
    Fit ARHMM with fixed K using ssm (Linderman).
    This is a practical fallback when true BNP BP-AR-HMM code isn't available.
    """
    try:
        import ssm  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "ssm backend requested but ssm import failed.\n"
            "Install ssm (pip install ssm) or try --backend pyhsmm.\n"
            f"Import error: {repr(e)}"
        )

    D_obs = int(seqs[0].shape[1])

    # Build ARHMM
    # Most common:
    #   ssm.HMM(K, D, observations="ar")
    # or
    #   ssm.ARHMM(K, D)
    hmm = None
    for ctor in (
        lambda: ssm.HMM(K, D_obs, observations="ar"),
        lambda: getattr(ssm, "ARHMM")(K, D_obs),
    ):
        try:
            hmm = ctor()
            break
        except Exception:
            continue
    if hmm is None:
        raise RuntimeError("Could not construct an ARHMM with ssm in this environment.")

    # Optional sticky prior: bump diagonal of transition matrix
    if sticky is not None and sticky > 0:
        try:
            P = hmm.transitions.log_Ps  # (K,K) in log space in some versions
            # safer: work in probs if available
            Ps = np.exp(P - P.max(axis=1, keepdims=True))
            Ps = Ps / Ps.sum(axis=1, keepdims=True)
            Ps = (1 - sticky) * Ps + sticky * np.eye(K)
            Ps = Ps / Ps.sum(axis=1, keepdims=True)
            hmm.transitions.log_Ps = np.log(Ps + 1e-16)
        except Exception:
            pass

    # Fit via EM
    # hmm.fit expects list of arrays
    hmm.fit(seqs, method="em", num_iters=n_iters, init_method="kmeans", verbose=2)

    # Viterbi states
    z_list = [np.asarray(hmm.most_likely_states(y), dtype=np.int64) for y in seqs]

    return {
        "backend": "ssm",
        "model": hmm,
        "z_list": z_list,
        "K": int(K),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/sungboo/rb10_control/dataset/demo_20260122_final.npz")
    ap.add_argument("--out", required=True, help="Output PKL with clustering results")

    ap.add_argument("--drop_demos", type=int, nargs="*", default=[36, 57, 98, 202])
    ap.add_argument("--min_len", type=int, default=10)

    ap.add_argument("--backend", choices=["pyhsmm", "ssm"], default="ssm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=100)

    # pyhsmm params (BNP-ish)
    ap.add_argument("--K_max", type=int, default=30, help="Truncation level for pyhsmm weak-limit model")
    ap.add_argument("--kappa", type=float, default=50.0, help="Sticky strength (pyhsmm)")

    # ssm params (fixed-K fallback)
    ap.add_argument("--K", type=int, default=20, help="Number of states for ssm ARHMM")
    ap.add_argument("--sticky", type=float, default=0.2, help="Diagonal bias for ssm transitions (0..1)")

    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_path = Path(args.out)

    print(f"[info] npz: {npz_path}")
    Y_all, _, ptrp, phase_grid = load_phase_demos(npz_path)

    keep_map = None
    if args.drop_demos:
        print(f"[info] drop_demos: {sorted(set(args.drop_demos))}")
        Y_all, ptrp, keep_map = filter_demos_by_index(Y_all, ptrp, args.drop_demos)

    seqs, demo_ids = split_sequences(Y_all, ptrp, min_len=int(args.min_len))
    if len(seqs) == 0:
        raise RuntimeError("No sequences left after filtering (min_len / drop_demos).")

    print(f"[info] sequences: {len(seqs)} demos (after drop & min_len={args.min_len})")
    print(f"[info] each seq dim: {seqs[0].shape[1]} (expect 6)")

    # Fit model
    if args.backend == "pyhsmm":
        fit = fit_arhmm_pyhsmm(
            seqs=seqs,
            n_iters=int(args.iters),
            K_max=int(args.K_max),
            seed=int(args.seed),
            kappa=float(args.kappa),
        )
        z_list = fit["z_list"]
        K_for_hist = int(fit["K_max"])
    else:
        fit = fit_arhmm_ssm(
            seqs=seqs,
            n_iters=int(args.iters),
            K=int(args.K),
            seed=int(args.seed),
            sticky=float(args.sticky),
        )
        z_list = fit["z_list"]
        K_for_hist = int(fit["K"])

    # Summaries per demo
    per_demo = {}
    for demo_i, z in zip(demo_ids, z_list):
        z = np.asarray(z, dtype=np.int64)
        K_hist = K_for_hist
        h = state_histogram(z, K_hist)
        dom = int(np.argmax(h)) if np.isfinite(h).all() and h.sum() > 0 else -1
        per_demo[int(demo_i)] = {
            "z": z,
            "dominant_state": dom,
            "hist": h,
            "T": int(z.shape[0]),
        }

    used_states = sorted(set(int(s) for v in per_demo.values() for s in v["z"].tolist()))
    print(f"[info] used states (unique): {used_states[:50]}{'...' if len(used_states) > 50 else ''}")
    print(f"[info] n_used_states: {len(used_states)}")

    payload = {
        "npz": str(npz_path),
        "drop_demos": list(map(int, args.drop_demos)),
        "min_len": int(args.min_len),
        "phase_grid": phase_grid,  # might be None
        "backend": args.backend,
        "fit": {k: v for k, v in fit.items() if k not in ["model", "z_list"]},  # keep light
        "per_demo": per_demo,  # demo_index_phase -> {z, dominant_state, hist, T}
        "demo_ids_used": demo_ids,
        "K_hist": int(K_for_hist),
        "note": (
            "This is AR-HMM style segmentation across demos. "
            "If you need true Beta-Process BP-AR-HMM, you'll likely need a custom implementation "
            "or a specific research codebase."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
