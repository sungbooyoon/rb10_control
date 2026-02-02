#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
discover_styles_bgmm.py

BGMM clustering on phase-aligned demos using post-contact window ONLY.

Key points (fixed):
- phase0 is crop start (NOT contact start) -> we do NOT support absolute phase windowing.
- window is always defined by contact_start_idx mapped into phase timeline:
    window = [contact_start_phase, contact_start_phase + L)
- supports robust SO(3) relative-rotation features using scipy Rotation (NO NaNs).
- drop_demos is handled correctly by mapping phase-demo index -> original demo index (kept_original).
- avoids UnboundLocalError by always defining xyz_w/rot_w before use.

NPZ expected keys:
- X_phase_crop: (N_phase, 3)
- W_phase_crop: (N_phase, 3)   # rotvec in local/task frame
- demo_ptr_phase: (D+1,)

For contact mapping (raw->crop->phase):
- crop_s: (D,), crop_e: (D,)
- contact_start_idx: (D,)  # raw index in original timeline
- (optional) contact_end_idx: (D,)  # unused in this script

For report_vs_skill:
- skill_id_crop: (N_crop,)
- demo_ptr_crop: (D+1,)
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


# -----------------------------
# helpers
# -----------------------------
def _finite(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(x)))


def _rotvec_rel_scientific(rotvec: np.ndarray, ref_rotvec: np.ndarray) -> np.ndarray:
    """
    rotvec_rel = log( R(ref)^T * R(t) ), robust for zero rotvec using scipy.
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
    Drop demos by phase-demo index (0..D-1), rebuild concatenated arrays and ptr.
    Returns:
      Xp_new, Wp_new, ptrp_new, kept_original
    where kept_original maps new phase-demo index -> original demo index.
    """
    drop_ids = sorted(set(int(i) for i in drop_ids))
    D = int(ptr_phase.shape[0] - 1)
    keep = [i for i in range(D) if i not in drop_ids]
    kept_original = np.asarray(keep, dtype=np.int64)

    if len(keep) == D:
        return X_phase, W_phase, ptr_phase, kept_original

    Xp_new, Wp_new = [], []
    ptrp_new = [0]
    for i in keep:
        sp, ep = int(ptr_phase[i]), int(ptr_phase[i + 1])
        Xp_new.append(X_phase[sp:ep])
        Wp_new.append(W_phase[sp:ep])
        ptrp_new.append(ptrp_new[-1] + (ep - sp))

    Xp_new = np.concatenate(Xp_new, axis=0) if Xp_new else np.zeros((0, 3), dtype=np.float64)
    Wp_new = np.concatenate(Wp_new, axis=0) if Wp_new else np.zeros((0, 3), dtype=np.float64)
    return Xp_new, Wp_new, np.asarray(ptrp_new, dtype=np.int64), kept_original


def infer_demo_skill_ids_from_skill_id_crop(skill_id_crop: np.ndarray, ptr_crop: np.ndarray) -> np.ndarray:
    """
    Majority vote per demo on cropped timeline.
    Returns demo_sid: (D,) skill id per demo (original indexing).
    """
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


def _raw_to_phase_index(data: dict, demo_orig: int, raw_idx: int, T_phase: int) -> Optional[int]:
    """
    Map raw index in original timeline -> phase index [0..T_phase-1]
    via crop_s/crop_e.
    """
    if "crop_s" not in data or "crop_e" not in data:
        return None
    crop_s = int(np.asarray(data["crop_s"], dtype=np.int64).reshape(-1)[demo_orig])
    crop_e = int(np.asarray(data["crop_e"], dtype=np.int64).reshape(-1)[demo_orig])
    crop_len = int(crop_e - crop_s)
    if crop_len <= 1:
        return None

    crop_idx = int(np.clip(raw_idx - crop_s, 0, crop_len - 1))
    phase_idx = int(np.round(crop_idx / (crop_len - 1) * (T_phase - 1)))
    return int(np.clip(phase_idx, 0, T_phase - 1))


def _get_contact_start_phase_index(data: dict, demo_orig: int, T_phase: int) -> Optional[int]:
    """
    contact_start_idx (raw) -> phase index.
    """
    if "contact_start_idx" not in data:
        return None
    cs_raw = int(np.asarray(data["contact_start_idx"], dtype=np.int64).reshape(-1)[demo_orig])
    return _raw_to_phase_index(data, demo_orig, cs_raw, T_phase)


def build_feature_matrix(
    X_phase: np.ndarray,
    W_phase: np.ndarray,
    ptr_phase: np.ndarray,
    *,
    kept_original: np.ndarray,
    use_xyz: bool,
    use_rot: bool,
    window_after_contact: int,
    min_len: int,
    data_npz: dict,
    use_relative: bool,
    relative_ref: str,  # "contact_start" or "phase0"
    debug_first: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Build flattened feature matrix per demo:
      - determine contact_start_phase
      - window = [contact_start_phase, contact_start_phase + L)
      - slice xyz/rot on that window
      - (optional) convert rot to relative rotvec w.r.t ref

    Returns:
      X_flat: (N_used, D_flat)
      used_demo_indices_phase: (N_used,) indices in post-drop phase-demo indexing
      dropped_reason: {phase_demo_idx: reason}
    """
    X_phase = np.asarray(X_phase, dtype=np.float64)
    W_phase = np.asarray(W_phase, dtype=np.float64)
    ptr_phase = np.asarray(ptr_phase, dtype=np.int64)
    kept_original = np.asarray(kept_original, dtype=np.int64)

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

    if (not use_xyz) and (not use_rot):
        raise ValueError("At least one of use_xyz/use_rot must be enabled.")
    if use_relative and (not use_rot):
        raise ValueError("--use_relative requires rotation features (use_rot=True).")
    if window_after_contact is None or int(window_after_contact) <= 0:
        raise ValueError("--window_after_contact must be a positive integer.")

    L = int(window_after_contact)

    X_list: List[np.ndarray] = []
    used: List[int] = []
    dropped_reason: Dict[int, str] = {}

    for i in range(D):
        demo_orig = int(kept_original[i])  # phase-demo -> original-demo mapping

        sp, ep = int(ptr_phase[i]), int(ptr_phase[i + 1])
        xyz_full = X_phase[sp:ep]  # (T,3)
        rot_full = W_phase[sp:ep]  # (T,3)

        # contact start phase
        cs = _get_contact_start_phase_index(data_npz, demo_orig, T_phase=T)
        if cs is None:
            dropped_reason[i] = "no_contact_start_idx_or_crop_map"
            continue

        w0 = int(cs)
        w1 = min(T, w0 + L)
        if w1 <= w0:
            dropped_reason[i] = f"bad_contact_window(cs={cs},L={L})"
            continue

        # slice window FIRST (avoid UnboundLocalError)
        xyz_w = xyz_full[w0:w1]
        rot_w = rot_full[w0:w1]

        # length check
        if (w1 - w0) < min_len:
            dropped_reason[i] = f"too_short_after_contact_window(len={w1-w0})"
            continue

        # validate finite
        if use_xyz and (not _finite(xyz_w)):
            dropped_reason[i] = "xyz_nonfinite"
            continue
        if use_rot and (not _finite(rot_w)):
            dropped_reason[i] = "rot_nonfinite"
            continue

        # relative rotation (computed on windowed rot)
        if use_relative:
            if relative_ref == "contact_start":
                ref_idx = int(cs)  # phase index
                ref_rotvec = rot_full[ref_idx]  # reference from full phase
            elif relative_ref == "phase0":
                ref_rotvec = rot_full[0]
            else:
                dropped_reason[i] = f"unknown_relative_ref({relative_ref})"
                continue

            if not (_finite(ref_rotvec) and _finite(rot_w)):
                dropped_reason[i] = "nonfinite_before_relative"
                continue

            rel = np.zeros_like(rot_w, dtype=np.float64)
            ok = True
            for t in range(rot_w.shape[0]):
                rv = rot_w[t]
                if not (_finite(rv) and _finite(ref_rotvec)):
                    ok = False
                    break
                rel[t] = _rotvec_rel_scientific(rv, ref_rotvec)

            if (not ok) or (not _finite(rel)):
                dropped_reason[i] = "nonfinite_after_relative"
                continue
            rot_w = rel

        # assemble features
        parts: List[np.ndarray] = []
        if use_xyz:
            parts.append(xyz_w)
        if use_rot:
            parts.append(rot_w)

        feat = np.concatenate(parts, axis=1)  # (Tw, dims)
        if not _finite(feat):
            dropped_reason[i] = "feat_nonfinite"
            continue

        X_list.append(feat.reshape(-1))
        used.append(i)

        if debug_first and len(used) <= int(debug_first):
            print(f"[debug] phase_demo={i} orig_demo={demo_orig} T={T} "
                  f"contact_start_phase={cs} window=[{w0},{w1}) Tw={w1-w0}")

    if len(X_list) == 0:
        if dropped_reason:
            print("[drop-reasons] (first 30)")
            for k in sorted(dropped_reason.keys())[:30]:
                print(f"  demo {k}: {dropped_reason[k]}")
        raise ValueError("No demos remained after building features (all dropped).")

    X_flat = np.stack(X_list, axis=0)
    used_demo_indices_phase = np.asarray(used, dtype=np.int64)
    return X_flat, used_demo_indices_phase, dropped_reason


def _confusion_counts(labels: np.ndarray, demo_skill: np.ndarray, K: int) -> Dict[int, Dict[int, int]]:
    out: Dict[int, Dict[int, int]] = {k: {} for k in range(K)}
    for y, s in zip(labels.tolist(), demo_skill.tolist()):
        out.setdefault(int(y), {})
        out[int(y)][int(s)] = out[int(y)].get(int(s), 0) + 1
    return out


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", default="/home/sungboo/rb10_control/dataset/demo_20260122_final.npz")
    ap.add_argument("--out", required=True)

    ap.add_argument("--drop_demos", type=int, nargs="*", default=[36, 57, 98, 202])
    ap.add_argument("--min_len", type=int, default=10)

    # signals
    ap.add_argument("--use_xyz", action="store_true", help="Include xyz in feature.")
    ap.add_argument("--use_rot", action="store_true", help="Include rotvec in feature.")
    ap.add_argument("--rot_only", action="store_true", help="Shortcut: use_rot=True, use_xyz=False")

    # ONLY window option (post-contact)
    ap.add_argument("--window_after_contact", type=int, required=True,
                    help="Window length L on phase timeline starting at contact_start_phase: [cs, cs+L).")

    # relative rotation
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

    ap.add_argument("--debug_first", type=int, default=0,
                    help="Print debug for first N used demos (0 disables).")

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

    # basic arrays
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
        # default: rot-only (your main use case)
        use_xyz = False
        use_rot = True

    # Build features (post-contact window only)
    X_flat, used_demo_indices_phase, dropped_reason = build_feature_matrix(
        X_phase=Xp,
        W_phase=Wp,
        ptr_phase=ptrp,
        kept_original=kept_original,
        use_xyz=use_xyz,
        use_rot=use_rot,
        window_after_contact=int(args.window_after_contact),
        min_len=int(args.min_len),
        data_npz=data,
        use_relative=bool(args.use_relative),
        relative_ref=str(args.relative_ref),
        debug_first=int(args.debug_first),
    )

    used_demo_indices_original = kept_original[used_demo_indices_phase]

    # standardize
    scaler = None
    X_fit = X_flat
    if args.standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_fit = scaler.fit_transform(X_flat)

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
    print(f"[info] window_after_contact: {int(args.window_after_contact)}  (post-contact only)")
    print(f"[BGMM] n_components={K}, active_clusters={len(active)}")
    for k in active:
        print(f"  cluster {k:02d}: n={int(counts[k])}")

    # optional report vs skill
    report = None
    if args.report_vs_skill and ("skill_id_crop" in data) and ("demo_ptr_crop" in data):
        demo_skill = infer_demo_skill_ids_from_skill_id_crop(data["skill_id_crop"], data["demo_ptr_crop"])
        demo_skill_used = demo_skill[used_demo_indices_original]
        report = _confusion_counts(labels=labels, demo_skill=demo_skill_used, K=K)

        print("\n[report_vs_skill] cluster -> skill composition (top5 per cluster)")
        for k in active:
            comp = report.get(int(k), {})
            items = sorted(comp.items(), key=lambda x: -x[1])[:5]
            s = ", ".join([f"skill{sid}:{cnt}" for sid, cnt in items])
            print(f"  cluster {k:02d}: {s}")

    # save pkl
    payload = dict(
        source_npz=str(npz_path),
        drop_demos=drop_demos,
        kept_original=kept_original,
        used_demo_indices_phase=used_demo_indices_phase,
        used_demo_indices_original=used_demo_indices_original,

        use_xyz=use_xyz,
        use_rot=use_rot,
        window_after_contact=int(args.window_after_contact),
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