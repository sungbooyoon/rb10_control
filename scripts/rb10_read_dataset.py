#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt

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

        plt.show()

if __name__ == "__main__":
    main()
