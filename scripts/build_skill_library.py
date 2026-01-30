#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build skill libraries (per skill_id) using BOTH:
  - DMP:  one primitive per demo trajectory
  - ProMP: one distribution model per skill (learned from multiple demos)

Input NPZ requires:
  - X_local_crop: (N, >=3) local position in [:,0:3]
  - demo_ptr_crop: (D+1,)
And skill labels:
  - demo_skill_id_crop: (D,)  OR
  - skill_id_crop: (N,) (per-step) -> majority vote per demo

Output:
  - pickle dict with "dmp" and "promp" libraries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import numpy as np

# movement_primitives (API may differ by version)
from movement_primitives.dmp import DMP
from movement_primitives.promp import ProMP


# -----------------------------
# skill id helpers
# -----------------------------
def infer_demo_skill_ids(npz: dict, ptr: np.ndarray) -> np.ndarray:
    D = int(ptr.shape[0] - 1)

    if "demo_skill_id_crop" in npz:
        demo_sid = np.asarray(npz["demo_skill_id_crop"]).astype(np.int64)
        if demo_sid.shape[0] != D:
            raise ValueError(f"demo_skill_id_crop length {demo_sid.shape[0]} != D {D}")
        return demo_sid

    if "skill_id_crop" in npz:
        sid = np.asarray(npz["skill_id_crop"]).astype(np.int64)
        if sid.shape[0] != int(ptr[-1]):
            raise ValueError("skill_id_crop length must equal X_local_crop length.")
        demo_sid = np.full((D,), -1, dtype=np.int64)
        for i in range(D):
            s, e = int(ptr[i]), int(ptr[i + 1])
            seg = sid[s:e]
            if seg.size:
                vals, cnt = np.unique(seg, return_counts=True)
                demo_sid[i] = int(vals[np.argmax(cnt)])
        return demo_sid

    raise KeyError(
        "Missing skill ids. Add one of:\n"
        "  - demo_skill_id_crop (D_crop,)\n"
        "  - skill_id_crop (N_crop,) aligned with X_local_crop\n"
    )


# -----------------------------
# DMP fitting
# -----------------------------
def fit_dmp(y: np.ndarray, dt: float, n_weights: int) -> dict:
    y = np.asarray(y, dtype=np.float64)
    T, n_dims = y.shape
    if T < 2:
        raise ValueError("Trajectory too short for DMP.")

    exec_time = (T - 1) * dt

    dmp = DMP(n_dims=n_dims, execution_time=exec_time, dt=dt, n_weights_per_dim=n_weights)

    # Try multiple imitate signatures across versions
    ok = False
    for call in (
        lambda: dmp.imitate(T=exec_time, y=y),
        lambda: dmp.imitate(y=y, T=exec_time),
        lambda: dmp.imitate(y),
        lambda: dmp.imitate(y=y),
    ):
        try:
            call()
            ok = True
            break
        except TypeError:
            continue
    if not ok:
        raise RuntimeError("DMP imitate() signature mismatch. Paste the TypeError traceback.")

    # open_loop reproduction
    y_hat = np.asarray(dmp.open_loop(), dtype=np.float64)
    TT = min(len(y_hat), len(y))
    rmse = float(np.sqrt(np.mean((y_hat[:TT] - y[:TT]) ** 2)))

    return {
        "dmp": dmp,
        "dt": float(dt),
        "execution_time": float(exec_time),
        "n_weights": int(n_weights),
        "T": int(T),
        "rmse": rmse,
        "y0": y[0].copy(),
        "g": y[-1].copy(),
    }


# -----------------------------
# ProMP fitting (skill-level)
# -----------------------------
def build_promp_for_skill(Y_list: list[np.ndarray], n_basis: int, dt: float) -> dict:
    """
    Fit one ProMP from multiple demos (each (T_i,3)).
    We time-normalize each demo to a common length via linear resampling.
    """
    if len(Y_list) == 0:
        raise ValueError("Empty Y_list for ProMP.")

    # choose common length (median length tends to be stable)
    lengths = np.array([y.shape[0] for y in Y_list], dtype=np.int64)
    T_common = int(np.median(lengths))
    T_common = max(T_common, 2)

    def resample(y: np.ndarray, Tnew: int) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        Told = y.shape[0]
        if Told == Tnew:
            return y
        x_old = np.linspace(0.0, 1.0, Told)
        x_new = np.linspace(0.0, 1.0, Tnew)
        out = np.zeros((Tnew, y.shape[1]), dtype=np.float64)
        for d in range(y.shape[1]):
            out[:, d] = np.interp(x_new, x_old, y[:, d])
        return out

    Y = np.stack([resample(y, T_common) for y in Y_list], axis=0)  # (N_demo, T, 3)

    # ProMP object: API may vary by version.
    # Common constructor: ProMP(n_dims, n_weights_per_dim, ...)
    promp = ProMP(n_dims=3, n_weights_per_dim=n_basis)

    # Try multiple learn/fit signatures across versions
    ok = False
    for call in (
        lambda: promp.learn(Y),           # some versions accept (N,T,D)
        lambda: promp.imitate(Y),         # others use imitate for ProMP too
        lambda: promp.fit(Y),
        lambda: promp.learn(Y, dt=dt),
        lambda: promp.fit(Y, dt=dt),
    ):
        try:
            call()
            ok = True
            break
        except (TypeError, AttributeError):
            continue
    if not ok:
        raise RuntimeError("ProMP learn/fit signature mismatch. Paste the error traceback.")

    # Quick reconstruction check if open_loop exists
    rmse = None
    try:
        y_hat = np.asarray(promp.mean_trajectory(T_common), dtype=np.float64)  # some versions
        rmse = float(np.sqrt(np.mean((y_hat - np.mean(Y, axis=0)) ** 2)))
    except Exception:
        try:
            y_hat = np.asarray(promp.open_loop(T_common), dtype=np.float64)
            rmse = float(np.sqrt(np.mean((y_hat - np.mean(Y, axis=0)) ** 2)))
        except Exception:
            rmse = None

    return {
        "promp": promp,
        "n_basis": int(n_basis),
        "T_common": int(T_common),
        "dt": float(dt),
        "n_demos": int(Y.shape[0]),
        "rmse_vs_mean_demo": rmse,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/sungboo/rb10_control/dataset/demo_20260122_final.npz")
    ap.add_argument("--out", default="/home/sungboo/rb10_control/dataset/skill_library_dmp_promp.pkl")

    # Time base: if you don't know sampling period, dt=1 is fine for shape modeling
    ap.add_argument("--dt", type=float, default=1.0)

    # DMP
    ap.add_argument("--dmp_n_weights", type=int, default=50)
    ap.add_argument("--min_len", type=int, default=10)

    # ProMP
    ap.add_argument("--promp_n_basis", type=int, default=25)
    ap.add_argument("--promp_min_demos", type=int, default=3)  # require at least this many demos per skill

    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    if "X_local_crop" not in data or "demo_ptr_crop" not in data:
        raise KeyError("NPZ must contain X_local_crop and demo_ptr_crop (crop 결과).")

    Xlc = np.asarray(data["X_local_crop"])
    ptr = np.asarray(data["demo_ptr_crop"]).astype(np.int64)
    D = int(ptr.shape[0] - 1)

    names = None
    if "demo_names_crop" in data:
        names = np.asarray(data["demo_names_crop"], dtype=object)
        if names.shape[0] != D:
            names = None

    demo_skill = infer_demo_skill_ids(data, ptr)

    # Group demos by skill
    skill_to_demo_idxs: dict[int, list[int]] = {}
    for i in range(D):
        sid = int(demo_skill[i])
        if sid < 0:
            continue
        skill_to_demo_idxs.setdefault(sid, []).append(i)

    # ---- Build DMP library (per-demo) ----
    dmp_lib: dict[int, list[dict]] = {}
    dmp_stats: dict[int, dict] = {}

    for sid, demo_idxs in skill_to_demo_idxs.items():
        for i in demo_idxs:
            s, e = int(ptr[i]), int(ptr[i + 1])
            y = Xlc[s:e, 0:3]
            if y.shape[0] < args.min_len:
                continue

            prim = fit_dmp(y=y, dt=args.dt, n_weights=args.dmp_n_weights)
            prim["skill_id"] = int(sid)
            prim["demo_index_crop"] = int(i)
            prim["demo_name"] = str(names[i]) if names is not None else f"crop_demo_{i}"
            dmp_lib.setdefault(sid, []).append(prim)

    for sid, lst in dmp_lib.items():
        rmses = [p["rmse"] for p in lst]
        dmp_stats[sid] = {
            "n": len(lst),
            "rmse_mean": float(np.mean(rmses)) if rmses else None,
            "rmse_median": float(np.median(rmses)) if rmses else None,
        }

    # ---- Build ProMP library (per-skill) ----
    promp_lib: dict[int, dict] = {}
    promp_stats: dict[int, dict] = {}

    for sid, demo_idxs in skill_to_demo_idxs.items():
        # collect trajectories
        Y_list = []
        used = []
        for i in demo_idxs:
            s, e = int(ptr[i]), int(ptr[i + 1])
            y = Xlc[s:e, 0:3]
            if y.shape[0] < args.min_len:
                continue
            Y_list.append(y)
            used.append(i)

        if len(Y_list) < args.promp_min_demos:
            # skip building ProMP for this skill
            continue

        model = build_promp_for_skill(Y_list=Y_list, n_basis=args.promp_n_basis, dt=args.dt)
        model["skill_id"] = int(sid)
        model["used_demo_indices_crop"] = np.array(used, dtype=np.int32)
        promp_lib[sid] = model

        promp_stats[sid] = {
            "n_demos": int(model["n_demos"]),
            "T_common": int(model["T_common"]),
            "rmse_vs_mean_demo": model["rmse_vs_mean_demo"],
        }

    payload = {
        "source_npz": str(npz_path),
        "dt": float(args.dt),
        "dmp": {
            "n_weights": int(args.dmp_n_weights),
            "min_len": int(args.min_len),
            "library": dmp_lib,
            "stats": dmp_stats,
        },
        "promp": {
            "n_basis": int(args.promp_n_basis),
            "min_demos": int(args.promp_min_demos),
            "library": promp_lib,
            "stats": promp_stats,
        },
        "note": (
            "DMP: demo-level primitives per skill.\n"
            "ProMP: skill-level distribution model per skill (requires enough demos; resampled to common length).\n"
        ),
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[saved] {out_path}")
    print(f"  skills (any demos): {len(skill_to_demo_idxs)}")
    print(f"  skills with DMP:    {len(dmp_lib)}")
    print(f"  skills with ProMP:  {len(promp_lib)}")

    for sid in sorted(skill_to_demo_idxs.keys()):
        dn = dmp_stats.get(sid, {}).get("n", 0)
        pn = promp_stats.get(sid, {}).get("n_demos", 0)
        print(f"  skill {sid}: DMP demos={dn}, ProMP demos={pn}")


if __name__ == "__main__":
    main()