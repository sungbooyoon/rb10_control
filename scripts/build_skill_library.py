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
    skill_id_crop: (N_crop,) aligned with X_*_crop + demo_ptr_crop
    ptr_crop: (D+1,) demo pointers into *_crop arrays (and skill_id_crop)
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
      2) skill_id_crop (N_crop,) aligned with demo_ptr_crop
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
        "  - skill_id_crop (N_crop,) aligned with demo_ptr_crop\n"
    )


# -----------------------------
# DMP
# -----------------------------
def fit_dmp(y: np.ndarray, dt: float, n_weights: int) -> dict:
    """
    Fit DMP to a single demo trajectory y (T,n_dims).
    Returns model + open-loop reproduction + rmse.
    """
    y = np.asarray(y, dtype=np.float64)
    T, n_dims = y.shape
    if T < 2:
        raise ValueError("Trajectory too short for DMP.")

    exec_time = (T - 1) * dt
    dmp = DMP(n_dims=n_dims, execution_time=exec_time, dt=dt, n_weights_per_dim=n_weights)

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


def fit_promp(Y_list: list[np.ndarray], n_basis: int) -> dict:
    """
    movement_primitives.promp.ProMP (네가 붙여준 소스) 기준:
      imitate(Ts, Ys)
        Ts: (N, T)
        Ys: (N, T, D)
      mean_trajectory(T): T는 1D time vector (shape (T,))
    """
    if len(Y_list) == 0:
        raise ValueError("Empty Y_list.")

    lengths_raw = np.array([y.shape[0] for y in Y_list], dtype=np.int64)

    T_common = max(int(np.median(lengths_raw)), 2)
    D = int(Y_list[0].shape[1])

    # (N,T,D)
    Y_rs = [resample(np.asarray(y, dtype=np.float64), T_common) for y in Y_list]
    Ys = np.stack(Y_rs, axis=0)
    N = int(Ys.shape[0])

    # (T,) and (N,T)
    t = np.linspace(0.0, 1.0, T_common, dtype=np.float64)
    Ts = np.tile(t[None, :], (N, 1))

    promp = ProMP(n_dims=D, n_weights_per_dim=n_basis)
    try:
        promp.imitate(Ts, Ys)
    except Exception as e:
        raise RuntimeError(
            "ProMP imitate() failed.\n"
            f"Ts shape={Ts.shape} (need (N,T)), Ys shape={Ys.shape} (need (N,T,D))\n"
            f"lengths_raw(first20)={lengths_raw[:20]}\n"
            f"Error: {repr(e)}"
        )

    # mean only (요구사항: mean만 있으면 됨)
    y_mean = None
    try:
        y_mean = np.asarray(promp.mean_trajectory(t), dtype=np.float64)  # (T,D)
    except Exception:
        y_mean = None

    return {"model": promp, "T_common": int(T_common), "t": t, "y_mean": y_mean}


# -----------------------------
# Metrics (no endpoint)
# -----------------------------
def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def summarize_errors(errs: list[float]) -> dict:
    x = np.asarray(errs, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
        "per_demo": x,
    }


def eval_promp_rmse_vs_mean(Y_list: list[np.ndarray], y_mean: np.ndarray) -> dict:
    """
    Returns RMSE summaries vs ProMP mean:
      - rmse_all: 6D
      - rmse_pos: xyz
      - rmse_rot: wx wy wz
    """
    y_mean = np.asarray(y_mean, dtype=np.float64)
    Tm, Dm = y_mean.shape
    if Dm < 6:
        raise ValueError(f"Expected y_mean dim>=6, got {y_mean.shape}")

    all_err, pos_err, rot_err = [], [], []
    for y in Y_list:
        y = np.asarray(y, dtype=np.float64)
        if y.shape[0] != Tm:
            y = resample(y, Tm)
        if y.shape[1] < 6:
            raise ValueError(f"Expected demo dim>=6, got {y.shape}")

        all_err.append(_rmse(y[:, :6], y_mean[:, :6]))
        pos_err.append(_rmse(y[:, 0:3], y_mean[:, 0:3]))
        rot_err.append(_rmse(y[:, 3:6], y_mean[:, 3:6]))

    return {
        "n": int(len(Y_list)),
        "rmse_all": summarize_errors(all_err),
        "rmse_pos": summarize_errors(pos_err),
        "rmse_rot": summarize_errors(rot_err),
    }


# -----------------------------
# Plot util
# -----------------------------
def plot_overlay_6d(
    y: np.ndarray,
    y_hat: np.ndarray | None,
    title: str,
    out_png: Path,
    t: np.ndarray | None = None,
    vlines: dict[str, int] | None = None,
):
    """
    Plot pos(3) + rotvec(3) overlay (6 subplots).
    y: (T,6) [x,y,z, wx,wy,wz]
    """
    y = np.asarray(y, dtype=np.float64)
    T = y.shape[0]
    tt = np.arange(T) if t is None else np.asarray(t).reshape(-1)

    labels = ["x", "y", "z", "wx", "wy", "wz"]
    plt.figure(figsize=(12, 14))

    for k in range(6):
        ax = plt.subplot(6, 1, k + 1)
        ax.plot(tt, y[:, k], label="demo")
        if y_hat is not None:
            Th = min(len(y_hat), T)
            ax.plot(tt[:Th], y_hat[:Th, k], label="mean")

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
        if k == 5:
            ax.set_xlabel("phase_idx" if t is None else "t")

        ax.legend(loc="upper right")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

def plot_outlier_overlays_for_skill(
    sid: int,
    demo_indices_phase: list[int],
    ptrp: np.ndarray,
    Y_all: np.ndarray,
    phase_grid: np.ndarray,
    y_mean: np.ndarray,
    rmse_pos_per_demo: np.ndarray,
    out_dir: Path,
    topk: int = 3,
    maxplot: int = 10,
):
    """
    - pick topk demos by rmse_pos_per_demo (descending)
    - save:
        * npy: outlier demo indices + their rmse
        * png: overlay plot (all selected outlier demos + ProMP mean) for 6 channels
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_indices_phase = list(map(int, demo_indices_phase))
    rmse_pos_per_demo = np.asarray(rmse_pos_per_demo, dtype=np.float64).reshape(-1)
    if rmse_pos_per_demo.shape[0] != len(demo_indices_phase):
        raise ValueError("rmse_pos_per_demo length must match demo_indices_phase length")

    K = int(min(max(1, topk), len(demo_indices_phase)))
    order = np.argsort(-rmse_pos_per_demo)  # descending
    pick = order[:K]

    out_demo_idxs = [demo_indices_phase[i] for i in pick.tolist()]
    out_rmses = rmse_pos_per_demo[pick].copy()

    # save indices + rmses
    np.savez_compressed(
        out_dir / f"skill_{sid:02d}_outliers_top{K}_rmsepos.npz",
        skill_id=np.array([sid], dtype=np.int32),
        demo_indices_phase=np.array(out_demo_idxs, dtype=np.int32),
        rmse_pos=np.array(out_rmses, dtype=np.float64),
    )

    # overlay plot: outlier demos + mean
    labels = ["x", "y", "z", "wx", "wy", "wz"]
    t = phase_grid.reshape(-1)

    plt.figure(figsize=(12, 14))
    for k in range(6):
        ax = plt.subplot(6, 1, k + 1)

        # plot mean
        ax.plot(t, y_mean[:, k], label="ProMP mean")

        # plot each outlier demo
        for j, di in enumerate(out_demo_idxs[:maxplot]):
            s, e = int(ptrp[di]), int(ptrp[di + 1])
            y = np.asarray(Y_all[s:e], dtype=np.float64)
            if y.shape[0] != y_mean.shape[0]:
                y = resample(y, y_mean.shape[0])
            ax.plot(t, y[:, k], label=f"demo {di} (rmse_pos={out_rmses[j]:.4f})")

        ax.set_ylabel(labels[k])
        if k == 0:
            ax.set_title(f"skill={sid} | top-{K} outliers by RMSE_pos (overlay)")

        if k == 5:
            ax.set_xlabel("t (0..1)")

        # legend only on first subplot (덜 지저분하게)
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / f"skill_{sid:02d}_outliers_top{K}_overlay_6d.png")
    plt.close()


def plot_skill_means_1to8(
    skill_means: dict[int, np.ndarray],
    t: np.ndarray,
    out_png: Path,
    title: str = "ProMP means for skills 1~8 (xyz + wx wy wz)",
):
    """
    One figure:
      6 subplots: x y z wx wy wz
      overlay ProMP mean curves for skill 1..8 (if present)
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    labels = ["x", "y", "z", "wx", "wy", "wz"]

    plt.figure(figsize=(14, 10))
    for k in range(6):
        ax = plt.subplot(6, 1, k + 1)
        for sid in range(1, 9):
            if sid not in skill_means:
                continue
            ym = np.asarray(skill_means[sid], dtype=np.float64)
            if ym.shape[0] != t.shape[0]:
                ym = resample(ym, t.shape[0])
            if ym.shape[1] < 6:
                continue
            ax.plot(t, ym[:, k], label=f"skill {sid}")
        ax.set_ylabel(labels[k])
        if k == 0:
            ax.set_title(title)
            ax.legend(loc="upper right", ncol=4, fontsize=9)
        if k == 5:
            ax.set_xlabel("t (0..1)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

def compute_y_length(y: np.ndarray) -> float:
    """
    y: (T,) y-coordinate trajectory
    returns total progress length (signed, assuming monotonic)
    """
    return float(y[-1] - y[0])



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/sungboo/rb10_control/dataset/demo_20260122_final_phase_logmap.npz")
    ap.add_argument("--out", default="/home/sungboo/rb10_control/dataset/skill_library_phase.pkl")

    ap.add_argument("--model", choices=["dmp", "promp", "both"], default="both")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--min_len", type=int, default=10)

    # DMP params
    ap.add_argument("--dmp_n_weights", type=int, default=50)

    # ProMP params
    ap.add_argument("--promp_n_basis", type=int, default=25)
    ap.add_argument("--promp_min_demos", type=int, default=3)

    # plot
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_demo", type=int, default=0, help="phase-demo index to visualize")
    ap.add_argument("--plot_dir", default="/home/sungboo/rb10_control/images/demo_20260122/skill_library_phase")

    ap.add_argument("--outlier_topk", type=int, default=3, help="per-skill top-k outliers by RMSE_pos vs ProMP mean")
    ap.add_argument("--outlier_maxplot", type=int, default=10, help="cap number of demos to overlay per outlier plot (safety)")

    args = ap.parse_args()

    npz_path = Path(args.npz)
    out_path = Path(args.out)
    plot_dir = Path(args.plot_dir)

    data = np.load(npz_path, allow_pickle=True)

    # --------------------
    # REQUIRED (phase-aligned) keys
    # --------------------
    required = ["X_phase_crop", "W_phase_crop", "demo_ptr_phase", "demo_ptr_crop"]
    for k in required:
        if k not in data:
            raise KeyError(f"NPZ must contain key: {k}")

    if ("skill_id_crop" not in data) and ("demo_skill_id_crop" not in data):
        raise KeyError("NPZ must contain: skill_id_crop (preferred) or demo_skill_id_crop")

    Xp = np.asarray(data["X_phase_crop"], dtype=np.float64)  # (N_phase,3)
    Wp = np.asarray(data["W_phase_crop"], dtype=np.float64)  # (N_phase,3)
    ptrp = np.asarray(data["demo_ptr_phase"], dtype=np.int64)  # (D_phase+1,)
    Dc_phase = int(ptrp.shape[0] - 1)

    # For skill ids, we infer from crop timeline (demo_ptr_crop), then map to phase demos 1:1
    ptrc = np.asarray(data["demo_ptr_crop"], dtype=np.int64)
    Dc_crop = int(ptrc.shape[0] - 1)
    if Dc_phase != Dc_crop:
        raise ValueError(f"Mismatch demos: D_phase={Dc_phase} != D_crop={Dc_crop}. Phase build assumed 1:1 demo.")

    # Phase grid (optional)
    phase_grid = np.asarray(data["phase_grid"], dtype=np.float64) if "phase_grid" in data else None
    if phase_grid is None:
        # infer constant phase length from first demo
        if Dc_phase <= 0:
            raise ValueError("No phase demos found.")
        phase_len = int(ptrp[1] - ptrp[0])
        phase_grid = np.linspace(0.0, 1.0, phase_len, dtype=np.float64)
    else:
        phase_len = int(phase_grid.shape[0])

    # Build concatenated 6D trajectories for learning (확인: 6D 맞음)
    # Y_all shape: (N_phase, 6)
    Y_all = np.concatenate([Xp, Wp], axis=1)
    if Y_all.shape[1] != 6:
        raise ValueError(f"Expected 6D (xyz+rotvec), but got Y_all.shape={Y_all.shape}")

    # skill ids per demo
    demo_skill = infer_demo_skill_ids(data, ptrc)  # (D,)

    # group by skill
    skill_to_demo_idxs: dict[int, list[int]] = {}
    for i in range(Dc_phase):
        sid = int(demo_skill[i])
        if sid >= 0:
            skill_to_demo_idxs.setdefault(sid, []).append(i)

    library = {"dmp": {}, "promp": {}}
    stats = {"dmp": {}, "promp": {}}

    # --------------------
    # DMP per-demo (skill groups)
    # --------------------
    if args.model in ("dmp", "both"):
        for sid, demo_idxs in skill_to_demo_idxs.items():
            lst = []
            rmses = []
            for i in demo_idxs:
                s, e = int(ptrp[i]), int(ptrp[i + 1])
                y = Y_all[s:e]  # (T,6)
                if y.shape[0] < args.min_len:
                    continue

                fit = fit_dmp(y=y, dt=args.dt, n_weights=args.dmp_n_weights)

                lst.append({
                    "skill_id": int(sid),
                    "demo_index_phase": int(i),
                    "dmp": fit["model"],
                    "rmse": float(fit["rmse"]),
                    "T": int(fit["T"]),
                })
                rmses.append(float(fit["rmse"]))

            if lst:
                library["dmp"][int(sid)] = lst
                stats["dmp"][int(sid)] = {
                    "n": int(len(lst)),
                    "rmse_mean": float(np.mean(rmses)),
                    "rmse_median": float(np.median(rmses)),
                    "rmse_max": float(np.max(rmses)),
                }

    # --------------------
    # ProMP per-skill (multi-demo)
    # --------------------
    if args.model in ("promp", "both"):
        for sid, demo_idxs in skill_to_demo_idxs.items():
            Y_list = []
            used = []
            for i in demo_idxs:
                s, e = int(ptrp[i]), int(ptrp[i + 1])
                y = Y_all[s:e]
                if y.shape[0] < args.min_len:
                    continue
                Y_list.append(y)
                used.append(i)

            if len(Y_list) < args.promp_min_demos:
                continue

            fit = fit_promp(Y_list=Y_list, n_basis=args.promp_n_basis)

            # validation metrics vs mean (endpoint 제외)
            metrics = None
            if fit["y_mean"] is not None:
                metrics = eval_promp_rmse_vs_mean(Y_list=Y_list, y_mean=fit["y_mean"])

            library["promp"][int(sid)] = {
                "skill_id": int(sid),
                "promp": fit["model"],
                "T_common": int(fit["T_common"]),
                "t": fit["t"],                 # (T_common,)
                "y_mean": fit["y_mean"],       # (T_common,6)
                "used_demo_indices_phase": np.array(used, dtype=np.int32),
                "metrics": metrics,            # NEW
            }

            # store summary for quick reading
            if metrics is not None:
                stats["promp"][int(sid)] = {
                    "n_demos": int(metrics["n"]),
                    "T_common": int(fit["T_common"]),
                    "rmse_pos_mean": float(metrics["rmse_pos"]["mean"]),
                    "rmse_pos_median": float(metrics["rmse_pos"]["median"]),
                    "rmse_pos_max": float(metrics["rmse_pos"]["max"]),
                    "rmse_rot_mean": float(metrics["rmse_rot"]["mean"]),
                    "rmse_rot_median": float(metrics["rmse_rot"]["median"]),
                    "rmse_rot_max": float(metrics["rmse_rot"]["max"]),
                    "rmse_all_mean": float(metrics["rmse_all"]["mean"]),
                    "rmse_all_median": float(metrics["rmse_all"]["median"]),
                    "rmse_all_max": float(metrics["rmse_all"]["max"]),
                }
            else:
                stats["promp"][int(sid)] = {
                    "n_demos": int(len(used)),
                    "T_common": int(fit["T_common"]),
                    "rmse_pos_mean": None,
                    "rmse_rot_mean": None,
                    "rmse_all_mean": None,
                }

            # --- NEW: outlier 저장 + plot (rmse_pos 기준) ---
            if (metrics is not None) and (fit["y_mean"] is not None):
                # metrics["rmse_pos"]["per_demo"]는 Y_list 순서(used 순서)와 1:1
                rmse_pos_per_demo = np.asarray(metrics["rmse_pos"]["per_demo"], dtype=np.float64)

                outlier_dir = plot_dir / "outliers"
                plot_outlier_overlays_for_skill(
                    sid=int(sid),
                    demo_indices_phase=used,         # phase demo indices (library에서 쓰는 인덱스)
                    ptrp=ptrp,
                    Y_all=Y_all,
                    phase_grid=phase_grid,
                    y_mean=fit["y_mean"],
                    rmse_pos_per_demo=rmse_pos_per_demo,
                    out_dir=outlier_dir,
                    topk=int(args.outlier_topk),
                    maxplot=int(args.outlier_maxplot),
                )


    payload = {
        "source_npz": str(npz_path),
        "dt": float(args.dt),
        "min_len": int(args.min_len),
        "model_choice": args.model,
        "phase_len": int(phase_len),
        "X_is_6d": True,  # 확인 결과
        "library": library,
        "stats": stats,
        "skill_ids_present": sorted(list(skill_to_demo_idxs.keys())),
        "n_phase_demos": int(Dc_phase),
        "n_phase_steps": int(Y_all.shape[0]),
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

    # show ProMP validation summary (no endpoint)
    if args.model in ("promp", "both") and len(library["promp"]) > 0:
        print("\n[ProMP validation (RMSE vs mean) summary]")
        for sid in sorted(library["promp"].keys()):
            m = library["promp"][sid].get("metrics", None)
            if m is None:
                print(f"  skill {sid:>2d}: metrics=None")
                continue
            print(
                f"  skill {sid:>2d} | n={m['n']:>3d} | "
                f"RMSE_pos mean/med/max = {m['rmse_pos']['mean']:.6g} / {m['rmse_pos']['median']:.6g} / {m['rmse_pos']['max']:.6g} | "
                f"RMSE_rot mean/med/max = {m['rmse_rot']['mean']:.6g} / {m['rmse_rot']['median']:.6g} / {m['rmse_rot']['max']:.6g} | "
                f"RMSE_all mean/med/max = {m['rmse_all']['mean']:.6g} / {m['rmse_all']['median']:.6g} / {m['rmse_all']['max']:.6g}"
            )

    # --------------------
    # Plot validation (one phase demo)
    # --------------------
    if args.plot:
        i = int(args.plot_demo)
        if not (0 <= i < Dc_phase):
            raise ValueError(f"--plot_demo must be within [0, {Dc_phase-1}]")

        s, e = int(ptrp[i]), int(ptrp[i + 1])
        y = Y_all[s:e]  # (phase_len,6)
        sid = int(demo_skill[i])

        t_demo = phase_grid
        vlines = {"phase0": 0, "phase1": (y.shape[0] - 1)}

        y_hat = None
        title = f"phase-demo={i} skill={sid} | {args.model}"

        if args.model in ("dmp", "both") and sid in library["dmp"]:
            hit = None
            for p in library["dmp"][sid]:
                if int(p["demo_index_phase"]) == i:
                    hit = p
                    break
            if hit is not None:
                dmp = hit["dmp"]
                y_hat = np.asarray(dmp.open_loop(), dtype=np.float64)
                title += f" | DMP rmse={hit['rmse']:.6f}"

        # 요구사항: ProMP mean만 있으면 됨 (검증 plot에서는 mean overlay)
        if y_hat is None and sid in library["promp"]:
            y_mean = library["promp"][sid].get("y_mean", None)
            if y_mean is not None:
                y_hat = y_mean
                title += " | ProMP mean"

        out_png = plot_dir / f"verify_phase_demo_{i:03d}_skill_{sid}_{args.model}_6d.png"
        plot_overlay_6d(y=y, y_hat=y_hat, title=title, out_png=out_png, t=t_demo, vlines=vlines)
        print(f"[plot] {out_png}")

        # --------------------
        # Plot: skills 1~8 mean curves in one figure (xyz + wx wy wz)
        # --------------------
        skill_means = {}
        for sid2 in range(1, 9):
            if sid2 in library["promp"]:
                ym = library["promp"][sid2].get("y_mean", None)
                if ym is not None:
                    skill_means[int(sid2)] = ym

        if len(skill_means) > 0:
            # use common t: prefer phase_grid if lengths match; else use each skill's t length (fallback to resample inside)
            out_png2 = plot_dir / "skills_01_to_08_promp_mean_xyz_w.png"
            plot_skill_means_1to8(skill_means=skill_means, t=phase_grid, out_png=out_png2)
            print(f"[plot] {out_png2}")


if __name__ == "__main__":
    main()
