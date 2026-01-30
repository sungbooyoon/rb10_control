#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def _set_equal_aspect_3d(ax, X, Y, Z):
    """3D에서 x=y=z 등축으로 보이게 조정."""
    x_range = np.max(X) - np.min(X)
    y_range = np.max(Y) - np.min(Y)
    z_range = np.max(Z) - np.min(Z)
    max_range = max(x_range, y_range, z_range)
    if max_range == 0:
        max_range = 1.0
    x_mid = 0.5 * (np.max(X) + np.min(X))
    y_mid = 0.5 * (np.max(Y) + np.min(Y))
    z_mid = 0.5 * (np.max(Z) + np.min(Z))
    r = 0.5 * max_range
    ax.set_xlim(x_mid - r, x_mid + r)
    ax.set_ylim(y_mid - r, y_mid + r)
    ax.set_zlim(z_mid - r, z_mid + r)

def plot_ee_pose_3d(name: str, ee_pose: np.ndarray, t: np.ndarray):
    """
    ee_pose: (N, >=3). 앞의 3개는 x,y,z로 가정.
    t: (N,) time array (색상에 사용할 수 있음; 여기선 기본 라인으로 표시)
    """
    if ee_pose.ndim != 2 or ee_pose.shape[1] < 3:
        print(f"[skip] {name}: shape={ee_pose.shape} -> 최소 (N,3) 필요")
        return

    X = ee_pose[:, 0]
    Y = ee_pose[:, 1]
    Z = ee_pose[:, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X, Y, Z)
    ax.scatter([X[0]], [Y[0]], [Z[0]], marker="o", s=40, label="start")
    ax.scatter([X[-1]], [Y[-1]], [Z[-1]], marker="^", s=40, label="end")
    ax.set_title(f"{name} 3D trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    _set_equal_aspect_3d(ax, X, Y, Z)
    fig.tight_layout()
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"{name}_3d.png")
        fig.savefig(fname, dpi=200)
        print(f"[saved] {fname}")


def is_image_array(arr) -> bool:
    """Return True if array/dataset looks like an image stack (N,H,W,3) or (H,W,3)."""
    shape = getattr(arr, "shape", None)
    if shape is None:
        return False
    if len(shape) >= 3 and int(shape[-1]) == 3:
        return True
    return False

def first_value_str(arr) -> str:
    """Pretty-print the first element of an HDF5 dataset / numpy array."""
    shape = getattr(arr, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[0]) == 0:
        return "<empty>"
    try:
        v = arr[0]
    except Exception:
        try:
            v = np.asarray(arr)
        except Exception:
            return "<unreadable>"

    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return str(v.item())
        if v.ndim == 1 and v.size <= 16:
            return np.array2string(v, precision=4, floatmode="fixed")
        if v.ndim == 1 and v.size > 16:
            head = np.array2string(v[:8], precision=4, floatmode="fixed")
            tail = np.array2string(v[-8:], precision=4, floatmode="fixed")
            return f"{head} ... {tail}"
        return f"<ndarray shape={v.shape} dtype={v.dtype}>"

    # numpy scalar
    try:
        if np.isscalar(v):
            return str(v)
    except Exception:
        pass
    return str(v)

def plot_multichannel(name: str, arr: np.ndarray, t: np.ndarray, max_dims: int = 32):
    """
    arr: shape (N,) or (N,D). 이미지(HWC)는 제외하고 전달.
    여러 채널이면 D개의 서브플롯을 만듭니다.
    """
    if arr.ndim == 1:
        fig = plt.figure(figsize=(10, 2.8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, arr)
        ax.set_title(f"{name}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            fname = os.path.join(out_dir, f"{name}.png")
            fig.savefig(fname, dpi=200)
            print(f"[saved] {fname}")
        return

    if arr.ndim >= 2:
        N, D = arr.shape[0], int(np.prod(arr.shape[1:]))
        # (N,D)로 평탄화
        flat = arr.reshape(N, D)
        D_plot = min(D, max_dims)
        rows = D_plot
        fig = plt.figure(figsize=(10, 2.2 * rows))
        for i in range(D_plot):
            ax = fig.add_subplot(rows, 1, i + 1)
            ax.plot(t, flat[:, i])
            ax.set_ylabel(f"ch{i}")
            if i == 0:
                ax.set_title(f"{name} (showing {D_plot}/{D} ch)")
            if i == rows - 1:
                ax.set_xlabel("time [s]")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            fname = os.path.join(out_dir, f"{name}.png")
            fig.savefig(fname, dpi=200)
            print(f"[saved] {fname}")

        if D > D_plot:
            print(f"[note] {name}: {D}차원 중 {D_plot}개만 표시했습니다 (max_dims={max_dims}).")

def main():
    ap = argparse.ArgumentParser(description="robomimic HDF5 개요 + demo_0 첫 값 + 멀티플롯")
    ap.add_argument("--hdf5", required=True, help="HDF5 경로")
    ap.add_argument("--max-dims", type=int, default=30, help="플롯에 표시할 최대 채널 수(너무 많을 때 제한)")
    ap.add_argument("--skill-id", type=str, default=None, help="Plot only demos in /data/mask/skill_<id> (e.g., 1 -> skill_1)")
    ap.add_argument("--out-dir", type=str, default="/home/sungboo/rb10_control/images/", help="If set, save all figures into this directory.")
    args = ap.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        # ----- 1. 전체 개요 -----
        print("=== HDF5 Summary ===")
        if "data" not in f:
            raise KeyError("HDF5에 'data' 그룹이 없습니다.")
        data = f["data"]
        demo_keys = sorted([k for k in data.keys() if k.startswith("demo_")],
                           key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else x)
        print("Trajectories:", demo_keys)
        print("Total samples:", data.attrs.get("total", "N/A"))
        raw_env = data.attrs.get("env_args", "{}")
        try:
            env_args = json.loads(raw_env)
            print("Env Args:", env_args)
        except Exception:
            print("Env Args (raw):", raw_env)

        # ---- skill mask 리스트업 ----
        skill_ids = []
        skill_keys = []
        if "mask" not in f:
            print("  <no /data/mask group>")
        else:
            mask = f["mask"]

            skill_keys = sorted(
                [k for k in mask.keys() if k.startswith("skill_")],
                key=lambda s: int(s.split("_", 1)[1]) if s.split("_", 1)[1].isdigit() else 10**9
            )

            if not skill_keys:
                print("  <no skill_* masks>")
            else:
                for sk in skill_keys:
                    try:
                        raw = mask[sk][...]
                        demo_list = []
                        for s in raw:
                            if isinstance(s, (bytes, bytearray)):
                                demo_list.append(s.decode("utf-8"))
                            else:
                                demo_list.append(str(s))

                        print(f"  /{sk} -> {demo_list}")
                    except Exception as e:
                        print(f"  /{sk} -> <unreadable> ({e})")

        # (옵션) 각 skill에 포함된 demo 개수도 같이 보고 싶으면:
        if skill_keys:
            for sk in skill_keys:
                try:
                    raw = data["mask"][sk][...]
                    demos_in_skill = []
                    for s in raw:
                        if isinstance(s, (bytes, bytearray)):
                            demos_in_skill.append(s.decode("utf-8"))
                        else:
                            demos_in_skill.append(str(s))
                    demos_in_skill = [d for d in demos_in_skill if d in data]
                    print(f"  - {sk}: {len(demos_in_skill)} demos")
                except Exception as e:
                    print(f"  - {sk}: <unreadable> ({e})")

        # ----- 2. demo_0의 obs와 action -----
        if "demo_0" not in data:
            raise KeyError("demo_0 그룹이 없습니다.")
        demo = data["demo_0"]
        N = int(demo.attrs.get("num_samples", 0))
        print("\n=== demo_0 summary ===")
        print("num_samples:", N)
        print("datasets:", list(demo.keys()))

        # time axis: prefer demo['meta/ref_ts'] if present; else fallback to demo.attrs['meta']; else sample index
        t = np.arange(N, dtype=np.float64)
        ref_ts = None
        if "meta" in demo and "ref_ts" in demo["meta"]:
            try:
                ref_ts = np.asarray(demo["meta"]["ref_ts"][...], dtype=np.float64)
            except Exception:
                ref_ts = None
        if ref_ts is None and "meta" in demo.attrs:
            try:
                meta = json.loads(demo.attrs["meta"])
                ref_ts = meta.get("ref_ts", None)
                if isinstance(ref_ts, list):
                    ref_ts = np.asarray(ref_ts, dtype=np.float64)
            except Exception:
                ref_ts = None
        if isinstance(ref_ts, np.ndarray) and ref_ts.shape[0] == N:
            t = ref_ts - float(ref_ts[0])

        # actions 첫 값
        if "actions" in demo:
            print("\n[action]:", first_value_str(demo["actions"]))
        else:
            print("\n[action]: <missing>")

        # obs 첫 값들
        print("\n[obs]")
        if "obs" not in demo:
            raise KeyError("demo_0에 'obs' 그룹이 없습니다.")
        obs = demo["obs"]
        for k in sorted(list(obs.keys())):
            arr = obs[k]
            if is_image_array(arr):
                print(f"  {k}: image shape={arr.shape}")
            else:
                print(f"  {k}: {first_value_str(arr)}")
                try:
                    arr_np = np.asarray(arr)
                    plot_multichannel(f"demo_0_obs_{k}", arr_np, t,
                                    max_dims=args.max_dims,
                                    out_dir=args.out_dir)
                except Exception as e:
                    print(f"[skip plot] {k}: {e}")

        # ----- 3. 모든 demo의 ee_pos 궤적을 한 그림에 -----
        fig3d = plt.figure(figsize=(6, 6))
        ax3d = fig3d.add_subplot(111, projection="3d")

        # Apply skill mask if requested
        selected_demo_keys = demo_keys
        if args.skill_id is not None:
            mask_key = f"skill_{str(args.skill_id)}"
            if "mask" in data and mask_key in data["mask"]:
                raw = data["mask"][mask_key][...]
                selected_demo_keys = []
                for s in raw:
                    if isinstance(s, (bytes, bytearray)):
                        selected_demo_keys.append(s.decode("utf-8"))
                    else:
                        # numpy.str_ / python str / others
                        selected_demo_keys.append(str(s))
                selected_demo_keys = [k for k in selected_demo_keys if k in data]
                ax3d.set_title(f"Skill {args.skill_id}: ee_pos trajectories ({len(selected_demo_keys)} demos)")
                print(f"Selected demos: {selected_demo_keys}")
            else:
                print(f"[WARN] mask not found: /data/mask/{mask_key}. Falling back to all demos.")
                selected_demo_keys = demo_keys
                ax3d.set_title("All demos: ee_pos trajectories")
        else:
            ax3d.set_title("All demos: ee_pos trajectories")

        ax3d.set_xlabel("x [m]")
        ax3d.set_ylabel("y [m]")
        ax3d.set_zlabel("z [m]")
        ax3d.grid(True, alpha=0.3)

        plotted_any = False
        all_X, all_Y, all_Z = [], [], []

        for demo_key in selected_demo_keys:
            demo_i = data[demo_key]
            if "obs" not in demo_i:
                continue
            obs_i = demo_i["obs"]
            if "ee_pos" not in obs_i:
                print(f"[skip] {demo_key}: obs/ee_pos 없음")
                continue

            ee_pos = np.asarray(obs_i["ee_pos"])
            if ee_pos.ndim != 2 or ee_pos.shape[1] < 3:
                print(f"[skip] {demo_key}: obs/ee_pos shape={ee_pos.shape} (N,3) 필요")
                continue

            X, Y, Z = ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2]
            ax3d.plot(X, Y, Z, label=demo_key)
            ax3d.scatter([X[0]], [Y[0]], [Z[0]], s=20, marker="o")  # start
            ax3d.scatter([X[-1]], [Y[-1]], [Z[-1]], s=30, marker="^")  # end

            all_X.append(X)
            all_Y.append(Y)
            all_Z.append(Z)
            plotted_any = True

        if plotted_any:
            # 등축 비율 맞추기
            X = np.concatenate(all_X)
            Y = np.concatenate(all_Y)
            Z = np.concatenate(all_Z)
            xmid, ymid, zmid = np.mean(X), np.mean(Y), np.mean(Z)
            max_range = max(X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()) / 2.0
            ax3d.set_xlim(xmid - max_range, xmid + max_range)
            ax3d.set_ylim(ymid - max_range, ymid + max_range)
            ax3d.set_zlim(zmid - max_range, zmid + max_range)
            if len(selected_demo_keys) <= 12:
                ax3d.legend(fontsize=8, loc="best", ncol=2)
            else:
                print(f"[note] {len(selected_demo_keys)} demos selected; legend omitted (too crowded).")
            if args.out_dir is not None:
                os.makedirs(args.out_dir, exist_ok=True)
                fname = os.path.join(args.out_dir, "ee_pos_trajectories.png")
                fig3d.savefig(fname, dpi=200)
                print(f"[saved] {fname}")
        else:
            print("[note] ee_pos 데이터를 가진 demo가 없습니다.")

        plt.show()


if __name__ == "__main__":
    main()
