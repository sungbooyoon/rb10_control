#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

from movement_primitives.dmp import DMP
from movement_primitives.promp import ProMP


# -----------------------------
# skill id helpers
# -----------------------------
def infer_demo_skill_ids_from_skill_id_crop(skill_id_crop: np.ndarray, ptr_crop: np.ndarray) -> np.ndarray:
    """
    skill_id_crop: (N_crop,) aligned with X_local_crop
    ptr_crop: (D+1,) demo pointers into X_local_crop (and skill_id_crop)
    returns demo_skill_id: (D,) one skill id per cropped demo (majority vote)
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


def infer_demo_skill_ids(npz: dict, ptr_crop: np.ndarray) -> np.ndarray:
    """
    Priority:
      1) demo_skill_id_crop (D,)
      2) skill_id_crop (N_crop,) aligned with X_local_crop
    """
    D = int(ptr_crop.shape[0] - 1)

    if "demo_skill_id_crop" in npz:
        demo_sid = np.asarray(npz["demo_skill_id_crop"]).astype(np.int64)
        if demo_sid.shape[0] != D:
            raise ValueError(f"demo_skill_id_crop length {demo_sid.shape[0]} != D {D}")
        return demo_sid

    if "skill_id_crop" in npz:
        return infer_demo_skill_ids_from_skill_id_crop(npz["skill_id_crop"], ptr_crop)

    raise KeyError(
        "Missing skill ids. Need one of:\n"
        "  - demo_skill_id_crop (D,)\n"
        "  - skill_id_crop (N_crop,) aligned with X_local_crop\n"
    )


# -----------------------------
# DMP
# -----------------------------
def fit_dmp(y: np.ndarray, dt: float, n_weights: int) -> dict:
    """
    Fit 3D Cartesian DMP to a single demo trajectory y (T,3).
    Returns model + open-loop reproduction + rmse.
    """
    y = np.asarray(y, dtype=np.float64)
    T, n_dims = y.shape
    if T < 2:
        raise ValueError("Trajectory too short for DMP.")

    exec_time = (T - 1) * dt
    dmp = DMP(n_dims=n_dims, execution_time=exec_time, dt=dt, n_weights_per_dim=n_weights)

    # tolerate API differences across movement_primitives versions
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
        raise RuntimeError("DMP imitate() signature mismatch. Paste the error traceback.")

    y_hat = np.asarray(dmp.open_loop(), dtype=np.float64)
    TT = min(len(y_hat), len(y))
    rmse = float(np.sqrt(np.mean((y_hat[:TT] - y[:TT]) ** 2)))

    return {"model": dmp, "y_hat": y_hat, "rmse": rmse, "T": int(T), "exec_time": float(exec_time)}


# -----------------------------
# ProMP
# -----------------------------
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


def fit_promp(Y_list: list[np.ndarray], n_basis: int, dt: float) -> dict:
    """
    Fit one ProMP per skill from multiple demos.
    We resample to a common length (median) then learn.
    """
    lengths = np.array([y.shape[0] for y in Y_list], dtype=np.int64)
    T_common = max(int(np.median(lengths)), 2)

    Y = np.stack([resample(y, T_common) for y in Y_list], axis=0)  # (N,T,3)

    promp = ProMP(n_dims=3, n_weights_per_dim=n_basis)

    ok = False
    for call in (
        lambda: promp.learn(Y),
        lambda: promp.fit(Y),
        lambda: promp.imitate(Y),
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

    # try to get mean trajectory
    y_mean = None
    for getter in (
        lambda: np.asarray(promp.mean_trajectory(T_common), dtype=np.float64),
        lambda: np.asarray(promp.open_loop(T_common), dtype=np.float64),
    ):
        try:
            y_mean = getter()
            break
        except Exception:
            continue

    return {"model": promp, "T_common": int(T_common), "y_mean": y_mean}


# -----------------------------
# Plot util
# -----------------------------
def plot_overlay_xyz(
    y: np.ndarray,
    y_hat: np.ndarray | None,
    title: str,
    out_png: Path,
    t: np.ndarray | None = None,
    vlines: dict[str, int] | None = None,
):
    """
    Plot x(t), y(t), z(t) overlay (3 subplots).
    t: (T,) optional; if None uses np.arange(T).
    vlines: dict[label] = index (on the demo timeline)
    """
    y = np.asarray(y, dtype=np.float64)
    T = y.shape[0]
    tt = np.arange(T) if t is None else np.asarray(t).reshape(-1)

    plt.figure(figsize=(12, 9))
    labels = ["x", "y", "z"]

    for k in range(3):
        ax = plt.subplot(3, 1, k + 1)
        ax.plot(tt, y[:, k], label="demo")
        if y_hat is not None:
            Th = min(len(y_hat), T)
            ax.plot(tt[:Th], y_hat[:Th, k], label="repro/mean")

        if vlines:
            for name, idx in vlines.items():
                if idx is None:
                    continue
                if 0 <= idx < T:
                    ax.axvline(tt[idx], linestyle="--")
                    ax.text(tt[idx], ax.get_ylim()[1], f" {name}", rotation=90, va="top")

        ax.set_ylabel(labels[k])
        if k == 0:
            ax.set_title(title)
        if k == 2:
            ax.set_xlabel("t")

        ax.legend(loc="upper right")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/sungboo/rb10_control/dataset/demo_20260122_final.npz")
    ap.add_argument("--out", default="/home/sungboo/rb10_control/dataset/skill_library.pkl")

    ap.add_argument("--model", choices=["dmp", "promp", "both"], default="both")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--min_len", type=int, default=10)

    # DMP params
    ap.add_argument("--dmp_n_weights", type=int, default=50)

    # ProMP params
    ap.add_argument("--promp_n_basis", type=int, default=25)
    ap.add_argument("--promp_min_demos", type=int, default=3)

    # validation / plot
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_demo", type=int, default=0, help="crop-demo index to visualize")
    ap.add_argument("--plot_dir", default="/home/sungboo/rb10_control/images/demo_20260122/skill_library")

    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_path = Path(args.out)
    plot_dir = Path(args.plot_dir)

    data = np.load(npz_path, allow_pickle=True)

    # ---- required keys (new NPZ) ----
    if "X_local_crop" not in data or "demo_ptr_crop" not in data:
        raise KeyError("NPZ must contain: X_local_crop, demo_ptr_crop")

    if ("skill_id_crop" not in data) and ("demo_skill_id_crop" not in data):
        raise KeyError("NPZ must contain: skill_id_crop (preferred) or demo_skill_id_crop")

    X = np.asarray(data["X_local_crop"])          # (N_crop, 7)
    ptr = np.asarray(data["demo_ptr_crop"]).astype(np.int64)
    D = int(ptr.shape[0] - 1)

    # optional time vector (crop timeline)
    t_crop = np.asarray(data["t_crop"]) if "t_crop" in data else None
    if t_crop is not None and t_crop.shape[0] != X.shape[0]:
        raise ValueError(f"t_crop length {t_crop.shape[0]} != X_local_crop length {X.shape[0]}")

    # per-demo skill id
    demo_skill = infer_demo_skill_ids(data, ptr)

    # optional meta for plotting (crop timeline indices)
    has_contact_meta = ("contact_start_idx" in data) and ("contact_end_idx" in data)
    has_chosen_crop = "chosen_index_crop" in data

    # group by skill
    skill_to_demo_idxs: dict[int, list[int]] = {}
    for i in range(D):
        sid = int(demo_skill[i])
        if sid >= 0:
            skill_to_demo_idxs.setdefault(sid, []).append(i)

    library = {"dmp": {}, "promp": {}}
    stats = {"dmp": {}, "promp": {}}

    # --------------------
    # DMP build + stats (per-demo)
    # --------------------
    if args.model in ("dmp", "both"):
        for sid, demo_idxs in skill_to_demo_idxs.items():
            lst = []
            rmses = []
            for i in demo_idxs:
                s, e = int(ptr[i]), int(ptr[i + 1])
                y = X[s:e, 0:3]
                if y.shape[0] < args.min_len:
                    continue

                fit = fit_dmp(y=y, dt=args.dt, n_weights=args.dmp_n_weights)

                lst.append({
                    "skill_id": sid,
                    "demo_index_crop": i,
                    "dmp": fit["model"],
                    "rmse": fit["rmse"],
                    "T": fit["T"],
                })
                rmses.append(fit["rmse"])

            if lst:
                library["dmp"][sid] = lst
                stats["dmp"][sid] = {
                    "n": len(lst),
                    "rmse_mean": float(np.mean(rmses)),
                    "rmse_median": float(np.median(rmses)),
                }

    # --------------------
    # ProMP build + stats (per-skill)
    # --------------------
    if args.model in ("promp", "both"):
        for sid, demo_idxs in skill_to_demo_idxs.items():
            Y_list = []
            used = []
            for i in demo_idxs:
                s, e = int(ptr[i]), int(ptr[i + 1])
                y = X[s:e, 0:3]
                if y.shape[0] < args.min_len:
                    continue
                Y_list.append(y)
                used.append(i)

            if len(Y_list) < args.promp_min_demos:
                continue
            
            fit = fit_promp(Y_list=Y_list, n_basis=args.promp_n_basis, dt=args.dt)

            # validation: mean trajectory vs each demo (resampled) RMSE
            rmses = []
            if fit["y_mean"] is not None:
                y_mean = fit["y_mean"]
                Tc = y_mean.shape[0]
                for y in Y_list:
                    yr = resample(y, Tc)
                    rmses.append(float(np.sqrt(np.mean((yr - y_mean) ** 2))))

            library["promp"][sid] = {
                "skill_id": sid,
                "promp": fit["model"],
                "T_common": fit["T_common"],
                "y_mean": fit["y_mean"],
                "used_demo_indices_crop": np.array(used, dtype=np.int32),
            }
            stats["promp"][sid] = {
                "n_demos": len(used),
                "T_common": fit["T_common"],
                "rmse_mean_vs_meantraj": float(np.mean(rmses)) if rmses else None,
                "rmse_median_vs_meantraj": float(np.median(rmses)) if rmses else None,
            }

    payload = {
        "source_npz": str(npz_path),
        "dt": float(args.dt),
        "min_len": int(args.min_len),
        "model_choice": args.model,
        "library": library,
        "stats": stats,
        "skill_ids_present": sorted(list(skill_to_demo_idxs.keys())),
        "n_crop_demos": int(D),
        "n_crop_steps": int(X.shape[0]),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[saved] {out_path}")
    print(f"  skills_total_in_data: {len(skill_to_demo_idxs)}")
    if args.model in ("dmp", "both"):
        print(f"  DMP skills built: {len(library['dmp'])}")
    if args.model in ("promp", "both"):
        print(f"  ProMP skills built: {len(library['promp'])}")

    # --------------------
    # Plot validation (one demo)
    # --------------------
    if args.plot:
        i = int(args.plot_demo)
        if not (0 <= i < D):
            raise ValueError(f"--plot_demo must be within [0, {D-1}]")

        s, e = int(ptr[i]), int(ptr[i + 1])
        y = X[s:e, 0:3]
        sid = int(demo_skill[i])

        # choose x-axis
        t_demo = None
        if t_crop is not None:
            t_demo = t_crop[s:e]

        vlines = {}
        if has_contact_meta:
            cs = int(np.asarray(data["contact_start_idx"])[i])
            ce = int(np.asarray(data["contact_end_idx"])[i])
            vlines["contact_start"] = cs
            vlines["contact_end"] = ce
        if has_chosen_crop:
            ci = int(np.asarray(data["chosen_index_crop"])[i])
            vlines["chosen"] = ci

        # overlay selection
        y_hat = None
        title = f"demo={i} skill={sid} | {args.model}"

        if args.model in ("dmp", "both") and sid in library["dmp"]:
            # find this demo's DMP
            hit = None
            for p in library["dmp"][sid]:
                if int(p["demo_index_crop"]) == i:
                    hit = p
                    break
            if hit is not None:
                dmp = hit["dmp"]
                y_hat = np.asarray(dmp.open_loop(), dtype=np.float64)
                title += f" | DMP rmse={hit['rmse']:.6f}"

        if y_hat is None and args.model in ("promp", "both") and sid in library["promp"]:
            y_mean = library["promp"][sid]["y_mean"]
            if y_mean is not None:
                y_hat = y_mean
                title += " | ProMP mean"

        out_png = plot_dir / f"verify_demo_{i:03d}_skill_{sid}_{args.model}.png"
        plot_overlay_xyz(y=y, y_hat=y_hat, title=title, out_png=out_png, t=t_demo, vlines=vlines)
        print(f"[plot] {out_png}")


if __name__ == "__main__":
    main()
