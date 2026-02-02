#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
discover_styles_bgmm.py

BGMM clustering on phase-aligned demos (optionally rotation-only),
with robust SO(3) relative-rotation features (NO NaNs).

Key features:
- Use scipy Rotation to avoid NaNs for zero-rotvec
- Optional windowing on phase timeline
- Optional relative rotation features (R_ref^T R_t -> logmap rotvec)
- Optional report vs. existing skill_id (confusion-like stats)

NPZ expected keys (from your inspect_npz.py):
- X_phase_crop: (N_phase, 3)
- W_phase_crop: (N_phase, 3)   # rotvec in local/task frame
- demo_ptr_phase: (D+1,)
Optional (for contact_start mapping):
- crop_s: (D,), crop_e: (D,)
- contact_start_idx: (D,) or first_contact_index: (D,) or stable_index: (D,) etc.
- skill_id_crop: (N_crop,) and demo_ptr_crop: (D+1,) for report_vs_skill majority vote

Example:
  python3 discover_styles_bgmm.py \
    --npz /home/sungboo/rb10_control/dataset/demo_20260122_final.npz \
    --out /home/sungboo/rb10_control/dataset/test_bgmm.pkl \
    --standardize --n_components 8 \
    --window 0 100 --use_relative --relative_ref contact_start --report_vs_skill
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required.\n"
        "  pip install scikit-learn\n"
        f"ImportError: {e}"
    )

# --- robust SO(3) ---
try:
    from scipy.spatial.transform import Rotation as R
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    R = None


def _finite(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(x)))


def _rotvec_rel_scientific(rotvec: np.ndarray, ref_rotvec: np.ndarray) -> np.ndarray:
    """
    Compute relative rotation rotvec_rel = log( R(ref)^T * R(rotvec) ).
    Uses scipy Rotation (robust for zero rotvec).
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for --use_relative. Install: pip install scipy")

    rotvec = np.asarray(rotvec, dtype=np.float64)
    ref_rotvec = np.asarray(ref_rotvec, dtype=np.float64)

    R_t = R.from_rotvec(rotvec)
    R_ref = R.from_rotvec(ref_rotvec)
    R_rel = R_ref.inv() * R_t
    return R_rel.as_rotvec()


def filter_phase_demos_by_index(
    X_phase: np.ndarray,
    W_phase: np.ndarray,
    ptr_phase: np.ndarray,
    drop_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop demos by phase-demo index, rebuild concatenated arrays and new ptr.
    Returns: Xp_new, Wp_new, ptrp_new, kept_demo_indices_original
    """
    drop_ids = sorted(set(int(i) for i in drop_ids))
    D = int(ptr_phase.shape[0] - 1)
    keep = [i for i in range(D) if i not in drop_ids]
    kept_demo_indices_original = np.asarray(keep, dtype=np.int64)

    if len(keep) == D:
        return X_phase, W_phase, ptr_phase, kept_demo_indices_original

    Xp_new, Wp_new = [], []
    ptrp_new = [0]
    for i in keep:
        sp, ep = int(ptr_phase[i]), int(ptr_phase[i + 1])
        Xp_new.append(X_phase[sp:ep])
        Wp_new.append(W_phase[sp:ep])
        ptrp_new.append(ptrp_new[-1] + (ep - sp))

    Xp_new = np.concatenate(Xp_new, axis=0) if Xp_new else np.zeros((0, 3), dtype=np.float64)
    Wp_new = np.concatenate(Wp_new, axis=0) if Wp_new else np.zeros((0, 3), dtype=np.float64)
    return Xp_new, Wp_new, np.asarray(ptrp_new, dtype=np.int64), kept_demo_indices_original


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


def _get_contact_start_phase_index(data: dict, demo_i: int, T_phase: int) -> Optional[int]:
    """
    Map a contact-start-like index to phase index [0..T_phase-1].

    Uses (preferred):
      - contact_start_idx (raw index in original demo timeline)
      - else first_contact_index
      - else stable_index
      - else chosen_index
    Requires crop_s/crop_e to map raw->crop, then crop->phase.

    Returns None if not possible.
    """
    if "crop_s" not in data or "crop_e" not in data:
        return None

    crop_s = int(np.asarray(data["crop_s"], dtype=np.int64).reshape(-1)[demo_i])
    crop_e = int(np.asarray(data["crop_e"], dtype=np.int64).reshape(-1)[demo_i])
    crop_len = int(crop_e - crop_s)
    if crop_len <= 1:
        return None

    # pick best available start index in raw timeline
    cand_keys = ["contact_start_idx", "first_contact_index", "stable_index", "chosen_index"]
    raw_idx = None
    for k in cand_keys:
        if k in data:
            arr = np.asarray(data[k], dtype=np.int64).reshape(-1)
            if demo_i < arr.shape[0]:
                raw_idx = int(arr[demo_i])
                break
    if raw_idx is None:
        return None

    # raw->crop
    crop_idx = raw_idx - crop_s
    crop_idx = int(np.clip(crop_idx, 0, crop_len - 1))

    # crop->phase
    phase_idx = int(np.round(crop_idx / (crop_len - 1) * (T_phase - 1)))
    phase_idx = int(np.clip(phase_idx, 0, T_phase - 1))
    return phase_idx


def build_feature_matrix(
    X_phase: np.ndarray,
    W_phase: np.ndarray,
    ptr_phase: np.ndarray,
    *,
    use_xyz: bool,
    use_rot: bool,
    window: Optional[Tuple[int, int]],
    use_relative: bool,
    relative_ref: str,
    min_len: int,
    data_npz: dict,
    verbose_drop: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Build flattened feature matrix:
      per demo -> slice window -> (T_window, D_feat) -> flatten

    Returns:
      X_flat: (N_used, D_flat)
      used_demo_indices_phase: (N_used,) indices in post-drop ptr_phase indexing
      dropped_reason: {demo_idx: reason}
    """
    X_phase = np.asarray(X_phase, dtype=np.float64)
    W_phase = np.asarray(W_phase, dtype=np.float64)
    ptr_phase = np.asarray(ptr_phase, dtype=np.int64)

    D = int(ptr_phase.shape[0] - 1)
    if D <= 0:
        raise ValueError("No demos found (D<=0).")

    lens = np.array([int(ptr_phase[i + 1] - ptr_phase[i]) for i in range(D)], dtype=np.int64)
    T = int(lens[0])
    if not np.all(lens == T):
        bad = np.where(lens != T)[0][:20]
        raise ValueError(
            f"Phase demos must share the same length T.\n"
            f"First T={T}, mismatched demos(first20)={bad.tolist()}, lens={lens[bad].tolist()}"
        )
    if T < min_len:
        raise ValueError(f"Phase length T={T} < min_len={min_len}")

    w0, w1 = 0, T
    if window is not None:
        w0, w1 = int(window[0]), int(window[1])
        w0 = max(0, min(T, w0))
        w1 = max(0, min(T, w1))
        if w1 <= w0:
            raise ValueError(f"Bad --window {window}, after clamp -> [{w0},{w1})")

    # feature dims
    dims = 0
    if use_xyz:
        dims += 3
    if use_rot:
        dims += 3
    if dims == 0:
        raise ValueError("At least one of --use_xyz / --use_rot must be enabled.")

    X_list = []
    used = []
    dropped_reason: Dict[int, str] = {}

    for i in range(D):
        sp, ep = int(ptr_phase[i]), int(ptr_phase[i + 1])

        xyz = X_phase[sp:ep]
        rot = W_phase[sp:ep]

        # window
        xyz_w = xyz[w0:w1]
        rot_w = rot[w0:w1]

        if xyz_w.shape[0] < min_len or rot_w.shape[0] < min_len:
            dropped_reason[i] = f"too_short_after_window(len={xyz_w.shape[0]})"
            continue

        # relative rotation
        if use_relative:
            if not use_rot:
                dropped_reason[i] = "use_relative_requires_use_rot"
                continue

            if relative_ref == "contact_start":
                ref_idx = _get_contact_start_phase_index(data_npz, i, T_phase=T)
                if ref_idx is None:
                    dropped_reason[i] = "no_contact_start_mapping_keys"
                    continue
                ref_rotvec = rot[ref_idx]  # use full-phase ref, not windowed
            elif relative_ref == "phase0":
                ref_rotvec = rot[0]
            else:
                dropped_reason[i] = f"unknown_relative_ref({relative_ref})"
                continue

            # compute per-step relative rotvec in the window
            rel = np.zeros_like(rot_w, dtype=np.float64)
            ok = True
            for t in range(rot_w.shape[0]):
                rv = rot_w[t]
                if not (_finite(rv) and _finite(ref_rotvec)):
                    ok = False
                    break
                rel[t] = _rotvec_rel_scientific(rv, ref_rotvec)

            if not ok or not _finite(rel):
                dropped_reason[i] = "nonfinite_after_relative"
                continue
            rot_w = rel

        # assemble features
        feat_parts = []
        if use_xyz:
            if not _finite(xyz_w):
                dropped_reason[i] = "xyz_nonfinite"
                continue
            feat_parts.append(xyz_w)
        if use_rot:
            if not _finite(rot_w):
                dropped_reason[i] = "rot_nonfinite"
                continue
            feat_parts.append(rot_w)

        y = np.concatenate(feat_parts, axis=1)  # (Tw, dims)
        if not _finite(y):
            dropped_reason[i] = "y_nonfinite"
            continue

        X_list.append(y.reshape(-1))
        used.append(i)

    if len(X_list) == 0:
        # print a few reasons
        if verbose_drop and dropped_reason:
            print("[drop-reasons] (first 20)")
            for k in sorted(dropped_reason.keys())[:20]:
                print(f"  demo {k}: {dropped_reason[k]}")
        raise ValueError("No demos remained after building features (all dropped).")

    X_flat = np.stack(X_list, axis=0)
    used_demo_indices_phase = np.asarray(used, dtype=np.int64)
    return X_flat, used_demo_indices_phase, dropped_reason


def _confusion_counts(labels: np.ndarray, demo_skill: np.ndarray, K: int) -> Dict[int, Dict[int, int]]:
    out: Dict[int, Dict[int, int]] = {}
    for k in range(K):
        out[k] = {}
    for y, s in zip(labels.tolist(), demo_skill.tolist()):
        out.setdefault(int(y), {})
        out[int(y)][int(s)] = out[int(y)].get(int(s), 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", default="/home/sungboo/rb10_control/dataset/demo_20260122_final.npz")
    ap.add_argument("--out", required=True)

    # keep consistent with pipeline
    ap.add_argument("--drop_demos", type=int, nargs="*", default=[36, 57, 98, 202])
    ap.add_argument("--min_len", type=int, default=10)

    # which signals to cluster on
    ap.add_argument("--use_xyz", action="store_true", help="Include xyz in feature.")
    ap.add_argument("--use_rot", action="store_true", help="Include rotvec in feature.")
    ap.add_argument("--rot_only", action="store_true", help="Shortcut: use_rot=True, use_xyz=False")

    # window on phase indices [start, end)
    ap.add_argument("--window", type=int, nargs=2, default=None, metavar=("START", "END"),
                    help="Phase window [START, END). Example: --window 0 100")

    # relative rotation options
    ap.add_argument("--use_relative", action="store_true",
                    help="Use relative rotation logmap: log(R_ref^T R_t). Requires scipy.")
    ap.add_argument("--relative_ref", choices=["contact_start", "phase0"], default="contact_start")

    # BGMM params
    ap.add_argument("--n_components", type=int, default=8)
    ap.add_argument("--covariance_type", choices=["full", "diag", "tied", "spherical"], default="full")
    ap.add_argument("--weight_concentration_prior_type",
                    choices=["dirichlet_process", "dirichlet_distribution"],
                    default="dirichlet_process")
    ap.add_argument("--weight_concentration_prior", type=float, default=None)
    ap.add_argument("--max_iter", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--reg_covar", type=float, default=1e-6)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    # report
    ap.add_argument("--report_vs_skill", action="store_true",
                    help="If skill_id_crop exists, print cluster->skill composition.")

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

    kept_original = np.arange(D0, dtype=np.int64)
    if drop_demos:
        bad = [i for i in drop_demos if i < 0 or i >= D0]
        if bad:
            raise ValueError(f"--drop_demos out-of-range: {bad} (valid 0..{D0-1})")
        Xp, Wp, ptrp, kept_original = filter_phase_demos_by_index(Xp, Wp, ptrp, drop_demos)

    # signal selection
    use_xyz = bool(args.use_xyz)
    use_rot = bool(args.use_rot)
    if args.rot_only:
        use_xyz = False
        use_rot = True
    if (not use_xyz) and (not use_rot):
        # default to rot-only for your use case
        use_xyz = False
        use_rot = True

    # build features
    X_flat, used_demo_indices_phase, dropped_reason = build_feature_matrix(
        X_phase=Xp,
        W_phase=Wp,
        ptr_phase=ptrp,
        use_xyz=use_xyz,
        use_rot=use_rot,
        window=tuple(args.window) if args.window is not None else None,
        use_relative=bool(args.use_relative),
        relative_ref=str(args.relative_ref),
        min_len=int(args.min_len),
        data_npz=data,
        verbose_drop=True,
    )
    used_demo_indices_original = kept_original[used_demo_indices_phase]

    # standardize
    scaler = None
    X_fit = X_flat
    if args.standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_fit = scaler.fit_transform(X_flat)

    # final guard
    if not np.all(np.isfinite(X_fit)):
        bad_rows = np.where(~np.isfinite(X_fit).all(axis=1))[0][:20]
        raise RuntimeError(f"Non-finite remained in X_fit. bad_rows(first20)={bad_rows.tolist()}")

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

    probs = bgmm.predict_proba(X_fit)
    labels = np.argmax(probs, axis=1).astype(np.int64)

    K = probs.shape[1]
    counts = np.bincount(labels, minlength=K)
    active = np.where(counts > 0)[0].tolist()

    # summary
    print(f"[info] npz: {npz_path}")
    print(f"[info] demos (original): {D0}, dropped: {drop_demos}, kept: {len(kept_original)}")
    print(f"[info] used demos (after feature build): {X_fit.shape[0]}, dropped_in_build: {len(dropped_reason)}")
    print(f"[info] feature dim: {X_fit.shape[1]}")
    print(f"[BGMM] n_components={K}, active_clusters={len(active)}")
    for k in active:
        print(f"  cluster {k:02d}: n={int(counts[k])}")

    # optional report vs skill
    report = None
    if args.report_vs_skill and ("skill_id_crop" in data) and ("demo_ptr_crop" in data):
        demo_skill = infer_demo_skill_ids_from_skill_id_crop(data["skill_id_crop"], data["demo_ptr_crop"])
        # map to used original indices
        demo_skill_used = demo_skill[used_demo_indices_original]
        report = _confusion_counts(labels=labels, demo_skill=demo_skill_used, K=K)

        print("\n[report_vs_skill] cluster -> skill composition (top5 per cluster)")
        for k in active:
            comp = report.get(int(k), {})
            items = sorted(comp.items(), key=lambda x: -x[1])[:5]
            s = ", ".join([f"skill{sid}:{cnt}" for sid, cnt in items])
            print(f"  cluster {k:02d}: {s}")

    # save pkl (so you can load scaler+bgmm later)
    payload = dict(
        source_npz=str(npz_path),
        drop_demos=drop_demos,
        used_demo_indices_phase=used_demo_indices_phase,
        used_demo_indices_original=used_demo_indices_original,

        use_xyz=use_xyz,
        use_rot=use_rot,
        window=None if args.window is None else [int(args.window[0]), int(args.window[1])],
        use_relative=bool(args.use_relative),
        relative_ref=str(args.relative_ref),

        standardize=bool(args.standardize),
        scaler=scaler,
        bgmm=bgmm,

        X_flat=X_flat.astype(np.float32),
        labels=labels,
        probs=probs.astype(np.float32),
        cluster_counts=counts.astype(np.int64),
        active_clusters=active,
        report_vs_skill=report,
        dropped_reason=dropped_reason,
        args=vars(args),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()