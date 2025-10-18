#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

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


def is_image_array(arr: np.ndarray) -> bool:
    return arr.ndim >= 3 and arr.shape[-1] == 3

def first_value_str(arr: np.ndarray) -> str:
    v = arr[0]
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
        if D > D_plot:
            print(f"[note] {name}: {D}차원 중 {D_plot}개만 표시했습니다 (max_dims={max_dims}).")

def main():
    ap = argparse.ArgumentParser(description="robomimic HDF5 개요 + demo_0 첫 값 + 멀티플롯")
    ap.add_argument("--hdf5", required=True, help="HDF5 경로")
    ap.add_argument("--max-dims", type=int, default=32, help="플롯에 표시할 최대 채널 수(너무 많을 때 제한)")
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

        # ----- 2. demo_0의 obs와 action -----
        if "demo_0" not in data:
            raise KeyError("demo_0 그룹이 없습니다.")
        demo = data["demo_0"]
        N = int(demo.attrs.get("num_samples", 0))
        print("\n=== demo_0 summary ===")
        print("num_samples:", N)
        print("datasets:", list(demo.keys()))

        # meta에서 hz 가져오기 (없으면 1.0Hz)
        hz = 1.0
        if "meta" in demo.attrs:
            try:
                meta = json.loads(demo.attrs["meta"])
                hz = float(meta.get("timeline_hz", hz))
            except Exception:
                pass
        t = np.arange(N, dtype=np.float64) / max(hz, 1e-9)

        # actions 첫 값
        if "actions" in demo:
            a0 = demo["actions"][0]
            print("\n[action]:", first_value_str(demo["actions"]))
        else:
            print("\n[action]: <missing>")

        # obs 첫 값들
        print("\n[obs]")
        obs = demo["obs"]
        for k in obs.keys():
            arr = obs[k]
            if is_image_array(arr):
                print(f"  {k}: image shape={arr.shape}")
            else:
                print(f"  {k}: {first_value_str(arr)}")

        """
        # ----- 3. 플롯: actions + 각 obs(이미지 제외) -----
        # actions
        if "actions" in demo:
            actions = np.asarray(demo["actions"])
            plot_multichannel("actions", actions, t, max_dims=args.max_dims)

        # obs/*
        for k in obs.keys():
            arr = np.asarray(obs[k])
            if is_image_array(arr):
                continue  # 이미지 제외
            name = f"obs/{k}"
            # (N,) 또는 (N,D)만 처리
            if arr.ndim == 1 or arr.ndim == 2:
                plot_multichannel(name, arr, t, max_dims=args.max_dims)
            else:
                # 예: (N,7) 같은 건 위에서 잡힘. 더 높은 차원(HWC 등)은 스킵.
                try:
                    flat = arr.reshape(arr.shape[0], -1)
                    plot_multichannel(name, flat, t, max_dims=args.max_dims)
                except Exception:
                    print(f"[skip] {name}: shape={arr.shape}는 선 그래프에 부적합하여 스킵합니다.")

        # ---- ee_pose 3D 시각화 (자동 탐지) ----
        ee_arr = np.asarray(obs["ee_pos"])
        # (N, >=3) 보장되면 그대로
        if ee_arr.ndim == 1:
            ee_arr = ee_arr.reshape(-1, 1)
        if ee_arr.shape[1] >= 3:
            plot_ee_pose_3d(f"obs/ee_pos", ee_arr, t)
        else:
            print(f"[skip] obs/ee_pos: shape={ee_arr.shape} -> 최소 (N,3) 필요")
        
        plt.show()
        """

        # ----- 4. 모든 demo의 ee_pos 궤적을 한 그림에 -----
        fig3d = plt.figure(figsize=(6, 6))
        ax3d = fig3d.add_subplot(111, projection="3d")
        ax3d.set_title("All demos: ee_pos trajectories")
        ax3d.set_xlabel("x [m]")
        ax3d.set_ylabel("y [m]")
        ax3d.set_zlabel("z [m]")
        ax3d.grid(True, alpha=0.3)

        plotted_any = False
        all_X, all_Y, all_Z = [], [], []

        for demo_key in demo_keys:
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
            ax3d.legend(fontsize=8, loc="best", ncol=2)
        else:
            print("[note] ee_pos 데이터를 가진 demo가 없습니다.")

        plt.show()


if __name__ == "__main__":
    main()
