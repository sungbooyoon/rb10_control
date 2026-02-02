#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_spromp.py

Train style-conditioned ProMP library (Mixture of MPs):
  style_id (BGMM label per demo) -> ProMP trained on phase-aligned trajectories.

Inputs:
- NPZ (phase-aligned):
    X_phase_crop: (N_phase, 3)
    W_phase_crop: (N_phase, 3)
    demo_ptr_phase: (D+1,)
  Optional (for contact-start mapping):
    crop_s, crop_e: (D,) raw->crop mapping
    contact_start_idx: (D,) raw indices
- BGMM PKL from discover_styles_bgmm.py:
    labels: (N_used,)
    used_demo_indices_original: (N_used,)  # indices w.r.t original demos in NPZ
    args: includes window_after_contact, n_components, etc. (best-effort)

Outputs:
- PKL containing style->promp, mean/var, and metadata.

Typical:
  python3 home/sungboo/rb10_control/scripts/train_spromp.py \
    --npz /home/sungboo/rb10_control/dataset/demo_20260122_final.npz \
    --style_pkl /home/sungboo/rb10_control/dataset/test_bgmm.pkl \
    --out /home/sungboo/rb10_control/dataset/spromp.pkl \
    --n_basis 25 --min_demos 5 --standardize_var
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
    return bool(np.all(np.isfinite(x)))


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


def trajectory_mean_and_var(promp: ProMP, t: np.ndarray, mc_samples: int = 200) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Return:
      y_mean: (T,D)
      y_var : (T,D) diag var if possible else Monte Carlo estimate; else None
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y_mean = np.asarray(promp.mean_trajectory(t), dtype=np.float64)

    # try built-in variance
    for attr in ("var_trajectory", "variance_trajectory", "cov_trajectory", "covariance_trajectory"):
        if hasattr(promp, attr):
            try:
                v = getattr(promp, attr)(t)
                v = _as_diag_var(v)
                if v.shape[0] == 1 and y_mean.shape[0] > 1:
                    # some impl returns single-time; expand if needed
                    v = np.tile(v, (y_mean.shape[0], 1))
                return y_mean, np.asarray(v, dtype=np.float64)
            except Exception:
                pass

    # fallback: MC variance from samples if available
    if hasattr(promp, "sample_trajectories"):
        try:
            Ys = np.asarray(promp.sample_trajectories(t, n_samples=int(mc_samples)), dtype=np.float64)
            # expected shapes: (S,T,D) or (T,S,D) depending on impl; handle common cases
            if Ys.ndim != 3:
                return y_mean, None
            if Ys.shape[1] == y_mean.shape[0]:
                # (S,T,D)
                y_var = np.var(Ys, axis=0)
                return y_mean, np.asarray(y_var, dtype=np.float64)
            if Ys.shape[0] == y_mean.shape[0]:
                # (T,S,D)
                y_var = np.var(Ys, axis=1)
                return y_mean, np.asarray(y_var, dtype=np.float64)
        except Exception:
            pass

    return y_mean, None


def _raw_to_phase_index(data_npz: dict, demo_orig: int, raw_idx: int, T_phase: int) -> Optional[int]:
    """raw index -> crop idx -> phase idx."""
    if "crop_s" not in data_npz or "crop_e" not in data_npz:
        return None
    crop_s = int(np.asarray(data_npz["crop_s"], dtype=np.int64).reshape(-1)[demo_orig])
    crop_e = int(np.asarray(data_npz["crop_e"], dtype=np.int64).reshape(-1)[demo_orig])
    crop_len = int(crop_e - crop_s)
    if crop_len <= 1:
        return None
    crop_idx = int(np.clip(raw_idx - crop_s, 0, crop_len - 1))
    phase_idx = int(np.round(crop_idx / (crop_len - 1) * (T_phase - 1)))
    return int(np.clip(phase_idx, 0, T_phase - 1))


def _contact_start_phase(data_npz: dict, demo_orig: int, T_phase: int) -> Optional[int]:
    if "contact_start_idx" not in data_npz:
        return None
    cs_raw = int(np.asarray(data_npz["contact_start_idx"], dtype=np.int64).reshape(-1)[demo_orig])
    return _raw_to_phase_index(data_npz, demo_orig, cs_raw, T_phase)


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
        # still allow but resample to T_phase
        y = np.concatenate([xyz, rot], axis=1)
        y = resample(y, T_phase)
        return y
    return np.concatenate([xyz, rot], axis=1)


# -----------------------------
# ProMP fit per style
# -----------------------------
def fit_style_promp(
    Y_list: List[np.ndarray],
    n_basis: int,
    T_common: int,
    mc_var_samples: int = 200,
) -> Dict:
    """
    Fit one ProMP to Y_list (N, T, D).
    Return dict with model + mean/var.
    """
    if len(Y_list) == 0:
        raise ValueError("Empty Y_list for style.")

    # enforce common length
    Y_rs = [resample(np.asarray(y, dtype=np.float64), T_common) for y in Y_list]
    Ys = np.stack(Y_rs, axis=0)  # (N,T,D)
    N, T, D = Ys.shape

    t = np.linspace(0.0, 1.0, T_common, dtype=np.float64)
    Ts = np.tile(t[None, :], (N, 1))  # (N,T)

    promp = ProMP(n_dims=int(D), n_weights_per_dim=int(n_basis))
    promp.imitate(Ts, Ys)

    y_mean, y_var = trajectory_mean_and_var(promp, t, mc_samples=int(mc_var_samples))
    return dict(
        promp=promp,
        T_common=int(T_common),
        t=t,
        y_mean=np.asarray(y_mean, dtype=np.float64),
        y_var=None if y_var is None else np.asarray(y_var, dtype=np.float64),
        n_var_mc_samples=int(mc_var_samples),
        n_basis=int(n_basis),
        n_demos=int(N),
    )


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
    ap.add_argument("--mc_var_samples", type=int, default=200)

    # training trajectory choice
    ap.add_argument("--fit_window_from_bgmm", action="store_true", help="Fit ProMP on the SAME post-contact window used in BGMM (window_after_contact).")
    ap.add_argument("--fit_full_phase", action="store_true", help="Force fit on full phase even if BGMM used a window. (default behavior)")

    # if using window, how to handle short remainder (near end)
    ap.add_argument("--pad_short_window", action="store_true", help="If cs+L exceeds T, pad by repeating last frame to keep fixed length.")
    ap.add_argument("--drop_short_window", action="store_true", help="If cs+L exceeds T, drop that demo for training that style.")

    # optional: scale variance for nicer plotting (not affecting model)
    ap.add_argument("--standardize_var", action="store_true", help="If y_var exists, normalize it by per-dim median to stabilize scale in plots.")
    

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

    labels = np.asarray(bg["labels"], dtype=np.int64).reshape(-1)
    used_orig = np.asarray(bg["used_demo_indices_original"], dtype=np.int64).reshape(-1)
    if labels.shape[0] != used_orig.shape[0]:
        raise ValueError(f"labels len {labels.shape[0]} != used_demo_indices_original len {used_orig.shape[0]}")

    # determine style count / active styles
    unique_styles = sorted(np.unique(labels).tolist())
    # some BGMM returns labels in [0..K-1], okay.

    # decide training length
    bg_args = bg.get("args", {})
    L_bgmm = None
    if isinstance(bg_args, dict) and ("window_after_contact" in bg_args):
        L_bgmm = bg_args.get("window_after_contact", None)
    # fallback if stored differently
    if L_bgmm is None and "window_after_contact" in bg:
        L_bgmm = bg["window_after_contact"]

    use_window = bool(args.fit_window_from_bgmm) and not bool(args.fit_full_phase)
    if use_window:
        if L_bgmm is None:
            raise ValueError("--fit_window_from_bgmm was set, but BGMM pkl has no window_after_contact.")
        L = int(L_bgmm)
        if L <= 1:
            raise ValueError(f"Bad window_after_contact={L}")
        T_common = L
    else:
        T_common = T_phase

    if args.drop_short_window and args.pad_short_window:
        raise ValueError("Choose only one of --drop_short_window / --pad_short_window")

    # group demos by style
    style_to_demos: Dict[int, List[int]] = {int(s): [] for s in unique_styles}
    for s, di in zip(labels.tolist(), used_orig.tolist()):
        style_to_demos[int(s)].append(int(di))

    # train per-style ProMP
    library: Dict[int, Dict] = {}
    stats: Dict[int, Dict] = {}
    dropped_by_style: Dict[int, List[Dict]] = {int(s): [] for s in unique_styles}

    for s in unique_styles:
        demo_list = style_to_demos.get(int(s), [])
        if len(demo_list) < int(args.min_demos):
            stats[int(s)] = dict(n_demos=int(len(demo_list)), skipped=True, reason="too_few_demos")
            continue

        Y_list: List[np.ndarray] = []

        for demo_orig in demo_list:
            if demo_orig < 0 or demo_orig >= D:
                dropped_by_style[int(s)].append(dict(demo_orig=int(demo_orig), reason="demo_index_oob"))
                continue

            y_full = _extract_demo_phase_traj(Xp, Wp, ptrp, demo_orig=demo_orig, T_phase=T_phase)
            if not _finite(y_full):
                dropped_by_style[int(s)].append(dict(demo_orig=int(demo_orig), reason="nonfinite_full"))
                continue

            if use_window:
                cs = _contact_start_phase(data, demo_orig=demo_orig, T_phase=T_phase)
                if cs is None:
                    dropped_by_style[int(s)].append(dict(demo_orig=int(demo_orig), reason="no_contact_start_mapping"))
                    continue
                w0 = int(cs)
                w1 = int(cs + T_common)

                if w1 <= T_phase:
                    y = y_full[w0:w1]
                else:
                    if args.drop_short_window:
                        dropped_by_style[int(s)].append(dict(
                            demo_orig=int(demo_orig),
                            reason="window_exceeds_T(drop)",
                            cs=int(cs),
                            w1=int(w1),
                            T=int(T_phase),
                        ))
                        continue
                    if args.pad_short_window:
                        # pad by repeating last frame
                        y_part = y_full[w0:T_phase]
                        if y_part.shape[0] <= 0:
                            dropped_by_style[int(s)].append(dict(
                                demo_orig=int(demo_orig),
                                reason="empty_window_after_cs",
                                cs=int(cs),
                            ))
                            continue
                        pad_n = int(w1 - T_phase)
                        pad = np.tile(y_part[-1:, :], (pad_n, 1))
                        y = np.concatenate([y_part, pad], axis=0)
                    else:
                        # default: clamp window end (will change length -> not allowed)
                        dropped_by_style[int(s)].append(dict(
                            demo_orig=int(demo_orig),
                            reason="window_exceeds_T(no_pad_no_drop)",
                            cs=int(cs),
                            w1=int(w1),
                            T=int(T_phase),
                        ))
                        continue

                if y.shape[0] != T_common:
                    dropped_by_style[int(s)].append(dict(
                        demo_orig=int(demo_orig),
                        reason="window_len_mismatch",
                        got=int(y.shape[0]),
                        expect=int(T_common),
                    ))
                    continue
            else:
                y = y_full

            if y.shape[0] < 2:
                dropped_by_style[int(s)].append(dict(demo_orig=int(demo_orig), reason="too_short"))
                continue

            Y_list.append(y)

        if len(Y_list) < int(args.min_demos):
            stats[int(s)] = dict(
                n_demos=int(len(Y_list)),
                skipped=True,
                reason="too_few_after_drop",
                dropped=len(dropped_by_style[int(s)]),
            )
            continue

        fit = fit_style_promp(
            Y_list=Y_list,
            n_basis=int(args.n_basis),
            T_common=int(T_common),
            mc_var_samples=int(args.mc_var_samples),
        )

        y_var = fit["y_var"]
        if args.standardize_var and (y_var is not None):
            # normalize variance scale for plotting/diagnostics
            med = np.median(np.maximum(y_var, 1e-12), axis=0)
            y_var = y_var / np.maximum(med[None, :], 1e-12)
            fit["y_var"] = y_var

        library[int(s)] = dict(
            style_id=int(s),
            promp=fit["promp"],
            T_common=int(fit["T_common"]),
            t=fit["t"],
            y_mean=fit["y_mean"],
            y_var=fit["y_var"],
            n_var_mc_samples=int(fit["n_var_mc_samples"]),
            n_basis=int(fit["n_basis"]),
            used_demo_indices_original=np.asarray([int(d) for d in demo_list], dtype=np.int32),
            used_demos_after_filter=np.asarray(
                [int(d) for d in demo_list if all(dd.get("demo_orig") != int(d) for dd in dropped_by_style[int(s)])],
                dtype=np.int32
            ),
        )

        stats[int(s)] = dict(
            skipped=False,
            n_demos_bgmm=int(len(demo_list)),
            n_demos_used=int(len(Y_list)),
            T_common=int(T_common),
            has_y_var=bool(fit["y_var"] is not None),
            dropped=int(len(dropped_by_style[int(s)])),
        )

    payload = dict(
        source_npz=str(npz_path),
        source_style_pkl=str(style_pkl_path),
        bgmm_args=bg.get("args", None),
        bgmm_active_clusters=bg.get("active_clusters", None),
        bgmm_cluster_counts=bg.get("cluster_counts", None),

        fit_window_from_bgmm=bool(use_window),
        T_phase=int(T_phase),
        T_common=int(T_common),

        n_basis=int(args.n_basis),
        min_demos=int(args.min_demos),
        mc_var_samples=int(args.mc_var_samples),

        styles_present=unique_styles,
        library=library,
        stats=stats,
        dropped_by_style=dropped_by_style,
        args=vars(args),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    # summary print
    print(f"[info] npz: {npz_path}")
    print(f"[info] style_pkl: {style_pkl_path}")
    print(f"[info] demos in npz: D={D}, T_phase={T_phase}")
    print(f"[info] used demos from bgmm: {used_orig.shape[0]}")
    print(f"[info] training mode: {'window_after_contact' if use_window else 'full_phase'} | T_common={T_common}")
    print(f"[info] styles (unique labels): {unique_styles}")

    built = sorted(library.keys())
    print(f"[result] built styles: {built} (/{len(unique_styles)})")
    for s in unique_styles:
        st = stats.get(int(s), {})
        if st.get("skipped", False):
            print(f"  style {int(s):02d}: SKIP | reason={st.get('reason')} | n={st.get('n_demos', st.get('n_demos_bgmm', 0))}")
        else:
            print(f"  style {int(s):02d}: OK   | n_used={st.get('n_demos_used')} | dropped={st.get('dropped')} | has_var={st.get('has_y_var')}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()