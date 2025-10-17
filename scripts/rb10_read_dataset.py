import h5py
import numpy as np
import json

h5_path = "/home/sungboo/ros2_ws/src/rb10_control/dataset/251017_kin_3.hdf5"   # 수정: 네 HDF5 경로

with h5py.File(h5_path, "r") as f:
    # ----- 1. 전체 개요 -----
    print("=== HDF5 Summary ===")
    if "data" not in f:
        raise KeyError("HDF5에 'data' 그룹이 없습니다.")
    data = f["data"]
    print("Trajectories:", [k for k in data.keys() if k.startswith("demo_")])
    print("Total samples:", data.attrs.get("total", "N/A"))
    env_args = data.attrs.get("env_args", "{}")
    try:
        print("Env Args:", json.loads(env_args))
    except Exception:
        print("Env Args (raw):", env_args)

    # ----- 2. demo_0의 obs와 action -----
    demo = data["demo_0"]
    print("\n=== demo_0 summary ===")
    print("num_samples:", demo.attrs["num_samples"])
    print("datasets:", list(demo.keys()))

    # action 첫 번째 값
    if "actions" in demo:
        a0 = demo["actions"][0]
        print("\n[action_0]:", a0)

    # obs 첫 번째 값들
    print("\n[obs_0]")
    obs = demo["obs"]
    for k in obs.keys():
        arr = obs[k]
        val = arr[0]
        # 이미지면 모양만 표시
        if arr.ndim >= 3 and arr.shape[-1] == 3:
            print(f"  {k}: image shape={arr.shape}")
        else:
            print(f"  {k}: {val}")

    # next_obs도 보고 싶으면 아래 주석 해제
    # print("\n[next_obs_0]")
    # for k in demo["next_obs"].keys():
    #     arr = demo["next_obs"][k]
    #     val = arr[0]
    #     if arr.ndim >= 3 and arr.shape[-1] == 3:
    #         print(f"  {k}: image shape={arr.shape}")
    #     else:
    #         print(f"  {k}: {val}")
