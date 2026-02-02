#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_spromp.py

Evaluate + plot style-conditioned ProMP library (NO DMP/NO cProMP).

Key requirements enforced:
1) Evaluate per-style (cluster)  ✅
2) Evaluate demos that belong to that style (and optionally within a target primitive skill) ✅
3) Plot per-style ✅

Inputs:
- --pkl: PKL produced by train_spromp.py (style-conditioned ProMPs)
- --style_pkl: (optional) BGMM clustering PKL with labels per demo (post-drop indexing)
  If train_spromp.py already embedded style labels/mapping, you can omit --style_pkl.
- --npz: optional override (default: payload["source_npz"])
- --skill_id: optional filter to restrict evaluation to a primitive skill id

NPZ expected:
- X_phase_crop (N,3)
- W_phase_crop (N,3)
- demo_ptr_phase (D+1,)
- demo_ptr_crop  (D+1,)
- skill_id_crop (N_crop,) OR demo_skill_id_crop (D,)

Style mapping expected (either in train payload or style_pkl):
- labels per demo in "phase demo index" space after drop_demos
  e.g. style_payload["labels"] aligned with style_payload["used_demo_indices_phase"]
  OR directly "demo_style" length D_after_drop

Outputs:
- prints RMSE summary per style
- optional plots saved under --plot_dir (default: ./spromp_eval_plots)

Example:
  python3 eval_spromp.py \
    --pkl /home/sungboo/rb10_control/dataset/spromp.pkl \
    --style_pkl /home/sungboo/rb10_control/dataset/test_bgmm.pkl \
    --plot --plot_dir /home/sungboo/rb10_control/images/spromp_eval \
    --skill_id 3
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# skill id helpers (same as your pipeline)
# -----------------------------
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


def infer_demo_skill_ids(npz: dict, ptr_crop: np.ndarray) -> np.ndarray:
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
# resample / rmse
# -----------------------------
def resample(y: np.ndarray, Tnew: int) -> np.ndarray:
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


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rmse_pos_rot_all(y: np.ndarray, yref: np.ndarray) -> Dict[str, float]:
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
        return {"mean": np.nan, "median": np.nan, "max": np.nan}
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "max": float(np.max(x))}


# -----------------------------
# drop demos (must match training)
# -----------------------------
def filter_demos_by_index(
    X_phase: np.ndarray,
    W_phase: np.ndarray,
    ptr_phase: np.ndarray,
    ptr_crop: np.ndarray,
    demo_skill: np.ndarray,
    drop_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drop_ids = sorted(set(int(i) for i in drop_ids))
    D = int(ptr_phase.shape[0] - 1)
    keep = [i for i in range(D) if i not in drop_ids]
    if len(keep) == D:
        return X_phase, W_phase, ptr_phase, ptr_crop, demo_skill

    Xp_new, Wp_new = [], []
    ptrp_new = [0]
    ptrc_new = [0]
    demo_skill_new = []

    for i in keep:
        sp, ep = int(ptr_phase[i]), int(ptr_phase[i + 1])
        Xp_new.append(X_phase[sp:ep])
        Wp_new.append(W_phase[sp:ep])
        ptrp_new.append(ptrp_new[-1] + (ep - sp))

        sc, ec = int(ptr_crop[i]), int(ptr_crop[i + 1])
        ptrc_new.append(ptrc_new[-1] + (ec - sc))

        demo_skill_new.append(int(demo_skill[i]))

    Xp_new = np.concatenate(Xp_new, axis=0)
    Wp_new = np.concatenate(Wp_new, axis=0)
    return (
        Xp_new,
        Wp_new,
        np.asarray(ptrp_new, dtype=np.int64),
        np.asarray(ptrc_new, dtype=np.int64),
        np.asarray(demo_skill_new, dtype=np.int64),
    )


# -----------------------------
# style mapping loader / alignment
# -----------------------------
def build_demo_style_map(
    *,
    D_after_drop: int,
    train_payload: dict,
    style_payload: Optional[dict],
) -> np.ndarray:
    """
    Returns demo_style: (D_after_drop,) where demo_style[i] is style label for phase-demo i.
    Priority:
      1) train_payload['demo_style'] if exists (already length D)
      2) train_payload['style']['demo_style'] if exists
      3) style_payload mapping from (used_demo_indices_phase, labels)
      4) style_payload['demo_style'] if exists
    """
    # (1)
    if "demo_style" in train_payload:
        ds = np.asarray(train_payload["demo_style"], dtype=np.int64).reshape(-1)
        if ds.shape[0] == D_after_drop:
            return ds
        raise ValueError(f"train_payload['demo_style'] length {ds.shape[0]} != D_after_drop {D_after_drop}")

    # (2)
    if isinstance(train_payload.get("style", None), dict) and ("demo_style" in train_payload["style"]):
        ds = np.asarray(train_payload["style"]["demo_style"], dtype=np.int64).reshape(-1)
        if ds.shape[0] == D_after_drop:
            return ds
        raise ValueError(f"train_payload['style']['demo_style'] length {ds.shape[0]} != D_after_drop {D_after_drop}")

    if style_payload is None:
        raise ValueError(
            "No demo_style mapping found in train PKL, and --style_pkl was not provided.\n"
            "Fix: either embed style labels into train_spromp.py payload, or pass --style_pkl."
        )

    # (4) direct
    if "demo_style" in style_payload:
        ds = np.asarray(style_payload["demo_style"], dtype=np.int64).reshape(-1)
        if ds.shape[0] == D_after_drop:
            return ds
        # else fall through: maybe it's in different indexing.

    # (3) mapping via (used_demo_indices_phase, labels)
    if ("used_demo_indices_phase" in style_payload) and ("labels" in style_payload):
        used = np.asarray(style_payload["used_demo_indices_phase"], dtype=np.int64).reshape(-1)
        labels = np.asarray(style_payload["labels"], dtype=np.int64).reshape(-1)
        if used.shape[0] != labels.shape[0]:
            raise ValueError(f"style used_demo_indices_phase len {used.shape[0]} != labels len {labels.shape[0]}")
        demo_style = np.full((D_after_drop,), -1, dtype=np.int64)
        for di, lb in zip(used.tolist(), labels.tolist()):
            if 0 <= int(di) < D_after_drop:
                demo_style[int(di)] = int(lb)
        # sanity: require at least some assigned
        if np.sum(demo_style >= 0) < max(3, int(0.2 * D_after_drop)):
            # not fatal but highly suspicious
            print(f"[warn] demo_style assigned only {int(np.sum(demo_style>=0))}/{D_after_drop}. "
                  "Check indexing alignment between style_pkl and drop_demos.")
        return demo_style

    raise ValueError(
        "style_pkl does not contain a usable style mapping.\n"
        "Need one of:\n"
        "  - style_payload['demo_style'] length D_after_drop\n"
        "  - style_payload['used_demo_indices_phase'] + style_payload['labels']\n"
    )


# -----------------------------
# confidence helpers
# -----------------------------
def _as_TD(arr: Optional[np.ndarray], T: int, D: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 1:
        if x.size == D:
            x = np.tile(x.reshape(1, D), (T, 1))
        elif x.size == T:
            x = np.tile(x.reshape(T, 1), (1, D))
        else:
            return None
    if x.ndim != 2:
        return None
    if x.shape[0] != T:
        if x.shape[1] == D:
            x = resample(x, T)
        else:
            return None
    if x.shape[1] != D:
        return None
    return x


def extract_mean_conf(entry: dict, T: int, D: int, z: float = 1.96) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    mu = _as_TD(entry.get("y_mean", None), T, D)
    var = _as_TD(entry.get("y_var", None), T, D)
    if mu is None:
        return None, None
    if var is None:
        return mu, None
    conf = z * np.sqrt(np.maximum(var, 0.0))
    return mu, conf


# -----------------------------
# plot utils (style-level)
# -----------------------------
def plot_overlay_6d(
    y: np.ndarray,
    overlays: List[Tuple[str, np.ndarray]],
    title: str,
    out_png: Path,
    t: Optional[np.ndarray] = None,
):
    y = np.asarray(y, dtype=np.float64)
    Tn = y.shape[0]
    tt = np.arange(Tn) if t is None else np.asarray(t).reshape(-1)

    labels = ["x", "y", "z", "wx", "wy", "wz"]
    plt.figure(figsize=(12, 14))
    for k in range(6):
        ax = plt.subplot(6, 1, k + 1)
        ax.plot(tt, y[:, k], label="demo")

        for name, yhat in overlays:
            yhat = np.asarray(yhat, dtype=np.float64)
            if yhat.shape[0] != Tn:
                yhat = resample(yhat, Tn)
            ax.plot(tt, yhat[:, k], label=name)

        ax.set_ylabel(labels[k])
        if k == 0:
            ax.set_title(title)
        if k == 5:
            ax.set_xlabel("t (0..1)" if t is not None else "phase_idx")
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_mean_std_across_demos(
    demos: List[np.ndarray],   # list of (T,6)
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


def plot_mean_conf(
    t: np.ndarray,
    y_mean: np.ndarray,
    y_conf: np.ndarray,
    out_png: Path,
    title: str,
):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y_mean = np.asarray(y_mean, dtype=np.float64)
    y_conf = np.asarray(y_conf, dtype=np.float64)
    if y_mean.shape[0] != t.shape[0]:
        y_mean = resample(y_mean, t.shape[0])
    if y_conf.shape[0] != t.shape[0]:
        y_conf = resample(y_conf, t.shape[0])

    labels = ["x", "y", "z", "wx", "wy", "wz"]
    plt.figure(figsize=(12, 14))
    for k in range(6):
        ax = plt.subplot(6, 1, k + 1)
        ax.plot(t, y_mean[:, k], label="mean")
        ax.fill_between(
            t,
            (y_mean[:, k] - y_conf[:, k]).ravel(),
            (y_mean[:, k] + y_conf[:, k]).ravel(),
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
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="style-conditioned ProMP PKL from train_spromp.py")
    ap.add_argument("--style_pkl", default=None, help="BGMM style clustering PKL (optional if embedded in train pkl)")
    ap.add_argument("--npz", default=None, help="Override NPZ path (default: payload['source_npz'])")
    ap.add_argument("--skill_id", type=int, default=None, help="Restrict evaluation to this primitive skill id (recommended).")

    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_dir", default=None)
    ap.add_argument("--plot_demo", type=int, default=None,
                    help="Pick a specific phase-demo index (after drop) to overlay plot. "
                         "If not set, one demo per style is auto-picked.")

    ap.add_argument("--min_len", type=int, default=None, help="Override min_len (default: payload['min_len'] or 10)")
    ap.add_argument("--drop_demos", type=int, nargs="*", default=None, help="Override drop_demos (default: payload['drop_demos'])")

    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    lib = payload.get("library", {})
    promp_lib = lib.get("promp", {})
    if not isinstance(promp_lib, dict) or len(promp_lib) == 0:
        raise ValueError("PKL has no library['promp'] entries (style-conditioned ProMP expected).")

    # optional style_pkl
    style_payload = None
    if args.style_pkl is not None:
        with open(Path(args.style_pkl), "rb") as f:
            style_payload = pickle.load(f)

    # npz path
    source_npz = Path(args.npz) if args.npz is not None else Path(payload["source_npz"])

    # params
    min_len = int(args.min_len) if args.min_len is not None else int(payload.get("min_len", 10))
    drop_demos = args.drop_demos if args.drop_demos is not None else list(map(int, payload.get("drop_demos", [])))

    # plot dir
    plot_dir = Path(args.plot_dir) if args.plot_dir is not None else Path(payload.get("plot_dir", "./spromp_eval_plots"))
    if args.plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # ---- load npz ----
    npz = np.load(source_npz, allow_pickle=True)
    for k in ("X_phase_crop", "W_phase_crop", "demo_ptr_phase", "demo_ptr_crop"):
        if k not in npz:
            raise KeyError(f"NPZ must contain key: {k}")

    Xp = np.asarray(npz["X_phase_crop"], dtype=np.float64)
    Wp = np.asarray(npz["W_phase_crop"], dtype=np.float64)
    ptrp = np.asarray(npz["demo_ptr_phase"], dtype=np.int64)
    ptrc = np.asarray(npz["demo_ptr_crop"], dtype=np.int64)
    D_phase = int(ptrp.shape[0] - 1)

    # phase grid/len
    if "phase_grid" in npz:
        phase_grid = np.asarray(npz["phase_grid"], dtype=np.float64).reshape(-1)
        phase_len = int(phase_grid.shape[0])
    else:
        phase_len = int(ptrp[1] - ptrp[0]) if D_phase > 0 else int(payload.get("phase_len", 0))
        if phase_len <= 0:
            raise ValueError("Cannot infer phase_len.")
        phase_grid = np.linspace(0.0, 1.0, phase_len, dtype=np.float64)

    demo_skill = infer_demo_skill_ids(npz, ptrc)

    # ---- apply drop_demos (must match training to align indices) ----
    if drop_demos:
        bad = [i for i in drop_demos if i < 0 or i >= D_phase]
        if bad:
            raise ValueError(f"drop_demos out-of-range: {bad} (valid 0..{D_phase-1})")

        Xp, Wp, ptrp, ptrc, demo_skill = filter_demos_by_index(
            X_phase=Xp, W_phase=Wp, ptr_phase=ptrp, ptr_crop=ptrc, demo_skill=demo_skill, drop_ids=drop_demos
        )
        D_phase = int(ptrp.shape[0] - 1)

    # concat y6
    Y_all = np.concatenate([Xp, Wp], axis=1)
    if Y_all.shape[1] != 6:
        raise ValueError(f"Expected 6D (xyz+rotvec), got {Y_all.shape}")

    # ---- build demo_style map in the SAME (post-drop) indexing ----
    demo_style = build_demo_style_map(
        D_after_drop=D_phase,
        train_payload=payload,
        style_payload=style_payload,
    )

    # ---- styles present in trained promp_lib ----
    # Here, keys are assumed to be style ids (cluster ids).
    # If your train_spromp.py uses nested dict (skill->style), adapt this part accordingly.
    style_ids_trained = sorted([int(k) for k in promp_lib.keys()])

    # ---- report setup ----
    print(f"[info] pkl: {pkl_path}")
    print(f"[info] npz: {source_npz}")
    print(f"[info] demos after drop: D={D_phase}, drop_demos={drop_demos}")
    if args.skill_id is not None:
        print(f"[info] filtering to primitive skill_id={int(args.skill_id)}")
    print(f"[info] trained styles in PKL: {style_ids_trained}")

    # ---- evaluate per style ----
    for st in style_ids_trained:
        entry = promp_lib.get(int(st), None)
        if entry is None:
            continue

        y_mean = entry.get("y_mean", None)
        if y_mean is None:
            print(f"\n=== style {st} ===")
            print("[skip] y_mean missing in PKL entry")
            continue
        y_mean = np.asarray(y_mean, dtype=np.float64)

        # demos in this style (cluster)
        idx_style = np.where(demo_style == int(st))[0].astype(np.int64)

        # optionally restrict to a primitive skill
        if args.skill_id is not None:
            idx_style = idx_style[np.where(demo_skill[idx_style] == int(args.skill_id))[0]]

        if idx_style.size == 0:
            print(f"\n=== style {st} ===")
            print("[skip] no demos after filtering (style and/or skill)")
            continue

        rmse_pos_all, rmse_rot_all, rmse_all_all = [], [], []
        demos_this_style: List[np.ndarray] = []

        for di in idx_style.tolist():
            sp, ep = int(ptrp[di]), int(ptrp[di + 1])
            y_demo = np.asarray(Y_all[sp:ep], dtype=np.float64)
            if y_demo.shape[0] < min_len:
                continue
            demos_this_style.append(y_demo)
            m = rmse_pos_rot_all(y_demo, y_mean)
            rmse_pos_all.append(m["rmse_pos"])
            rmse_rot_all.append(m["rmse_rot"])
            rmse_all_all.append(m["rmse_all"])

        s_pos = summarize(rmse_pos_all)
        s_rot = summarize(rmse_rot_all)
        s_all = summarize(rmse_all_all)

        print(f"\n=== style {st} ===")
        print(f"[demos] n={len(rmse_all_all)} (after min_len={min_len})")
        print("[ProMP(style) RMSE]")
        print(f"  pos mean/med/max = {s_pos['mean']:.6g}/{s_pos['median']:.6g}/{s_pos['max']:.6g}")
        print(f"  rot mean/med/max = {s_rot['mean']:.6g}/{s_rot['median']:.6g}/{s_rot['max']:.6g}")
        print(f"  all mean/med/max = {s_all['mean']:.6g}/{s_all['median']:.6g}/{s_all['max']:.6g}")

        # ---- plotting per style ----
        if args.plot:
            style_dir = plot_dir / f"style_{st:02d}"
            style_dir.mkdir(parents=True, exist_ok=True)

            # (A) demo mean ± std inside this style
            if len(demos_this_style) > 0:
                out_png = style_dir / f"style_{st:02d}_demo_mean_std.png"
                plot_mean_std_across_demos(
                    demos=demos_this_style,
                    t=phase_grid,
                    out_png=out_png,
                    title=f"Style {st}: demo mean ± std"
                )
                print(f"[plot] {out_png}")

            # (B) style promp mean ± conf (if var exists)
            mu, conf = extract_mean_conf(entry=entry, T=phase_len, D=6, z=1.96)
            if (mu is not None) and (conf is not None):
                out_png2 = style_dir / f"style_{st:02d}_promp_mean_conf.png"
                plot_mean_conf(
                    t=phase_grid,
                    y_mean=mu,
                    y_conf=conf,
                    out_png=out_png2,
                    title=f"Style {st}: ProMP mean ± 95% conf"
                )
                print(f"[plot] {out_png2}")

            # (C) overlay: pick one demo in this style (user-specified takes priority)
            pick_di = None
            if args.plot_demo is not None:
                if 0 <= int(args.plot_demo) < D_phase:
                    if demo_style[int(args.plot_demo)] == int(st):
                        if (args.skill_id is None) or (demo_skill[int(args.plot_demo)] == int(args.skill_id)):
                            pick_di = int(args.plot_demo)
            if pick_di is None:
                pick_di = int(idx_style[0])

            sp, ep = int(ptrp[pick_di]), int(ptrp[pick_di + 1])
            y_pick = np.asarray(Y_all[sp:ep], dtype=np.float64)

            overlays = [("style ProMP mean", y_mean)]
            title = f"style={st} demo={pick_di} (skill={int(demo_skill[pick_di])})"
            out_png3 = style_dir / f"style_{st:02d}_overlay_demo_{pick_di:03d}.png"
            plot_overlay_6d(
                y=y_pick,
                overlays=overlays,
                title=title,
                out_png=out_png3,
                t=phase_grid
            )
            print(f"[plot] {out_png3}")


if __name__ == "__main__":
    main()