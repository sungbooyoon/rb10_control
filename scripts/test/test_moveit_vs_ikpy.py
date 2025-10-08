#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import numpy as np
import rclpy
from geometry_msgs.msg import Pose

from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder

from ikpy.chain import Chain

# ========= 사용자 환경 =========
GROUP        = "manipulator"     # SRDF 그룹명
TIP_LINK     = "tcp"             # EE 링크명
TIMEOUT      = 0.02              # MoveItPy IK timeout (s)
N_ITERS      = 300               # 반복 횟수
SEED_NOISE   = 0.05              # 시드 섭동 (rad)
JOINT_LIMIT  = 1.2               # 랜덤 q 범위 ±limit (rad)

BASE_LINK = "link0"
EE_LINK   = "tcp"
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
URDF_PATH  = "/home/sungboo/ros2_ws/src/rbpodo_ros2/rbpodo_description/robots/rb10_1300e_u.urdf"
ACTIVE_LINKS_MASK = [False, True, True, True, True, True, True, False]
BASE_ELEMENTS     = [BASE_LINK]
# ==============================

def quat_from_rotm(R):
    """3x3 -> (x,y,z,w)"""
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2,1]-R[1,2]) / s
        y = (R[0,2]-R[2,0]) / s
        z = (R[1,0]-R[0,1]) / s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = math.sqrt(1.0 + R[0,0]-R[1,1]-R[2,2]) * 2.0
            w = (R[2,1]-R[1,2]) / s
            x = 0.25*s
            y = (R[0,1]+R[1,0]) / s
            z = (R[0,2]+R[2,0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1,1]-R[0,0]-R[2,2]) * 2.0
            w = (R[0,2]-R[2,0]) / s
            x = (R[0,1]+R[1,0]) / s
            y = 0.25*s
            z = (R[1,2]+R[2,1]) / s
        else:
            s = math.sqrt(1.0 + R[2,2]-R[0,0]-R[1,1]) * 2.0
            w = (R[1,0]-R[0,1]) / s
            x = (R[0,2]+R[2,0]) / s
            y = (R[1,2]+R[2,1]) / s
            z = 0.25*s
    return np.array([x,y,z,w], dtype=float)

def pose_from_T(T):
    p = Pose()
    p.position.x, p.position.y, p.position.z = T[0,3], T[1,3], T[2,3]
    q = quat_from_rotm(T[:3,:3])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
    return p

def summarize(x):
    a = np.array(x, dtype=float)
    return dict(
        mean=float(np.mean(a)),
        std=float(np.std(a)),
        p50=float(np.percentile(a, 50)),
        p90=float(np.percentile(a, 90)),
        p99=float(np.percentile(a, 99)),
        min=float(np.min(a)),
        max=float(np.max(a)),
    )

def main():
    # ---------- MoveItPy 준비 ----------
    rclpy.init()
    mc = (
        MoveItConfigsBuilder("rbpodo", package_name="rbpodo_moveit_config")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .moveit_cpp(file_path="/home/sungboo/ros2_ws/src/rb10_control/config/moveit_py_config.yaml")
        .to_moveit_configs()
    )
    robot = MoveItPy(node_name="moveit_py", config_dict=mc.to_dict())
    model = robot.get_robot_model()
    jmg   = model.get_joint_model_group(GROUP)
    state = RobotState(model)
    state.set_to_default_values()

    # ---------- ikpy 준비 ----------
    chain_full = Chain.from_urdf_file(URDF_PATH, base_elements=BASE_ELEMENTS)
    chain = Chain(name="rb10_reduced",
                  links=[l for (l,active) in zip(chain_full.links, ACTIVE_LINKS_MASK) if active])
    dof = len(JOINT_NAMES)  # 6
    # ikpy seed 길이는 chain의 joint 개수에 맞춰야 함
    ikpy_nb_joints = 6

    # ---------- 워밍업 ----------
    for _ in range(3):
        q_rand = np.random.uniform(-JOINT_LIMIT, JOINT_LIMIT, dof)
        T = chain.forward_kinematics(q_rand)
        target_pose = pose_from_T(T)
        q_seed = q_rand + np.random.uniform(-SEED_NOISE, SEED_NOISE, dof)

        # (선택) MoveIt 시드 동일하게 주입 — snake_case 바인딩 가정
        names = jmg.active_joint_model_names
        state.joint_positions = {n: float(v) for n, v in zip(names, q_seed)}
        state.set_from_ik(GROUP, target_pose, TIP_LINK, TIMEOUT)

        # ikpy IK: 3D 위치 + 3×3 방향행렬을 별도 인자로
        pos = T[:3, 3]; R = T[:3, :3]
        chain.inverse_kinematics(
            target_position=pos,
            target_orientation=R,
            initial_position=np.resize(q_seed, ikpy_nb_joints),
        )

    # ---------- 본 벤치 ----------
    times_moveit = []; times_ikpy = []
    succ_moveit = 0;   succ_ikpy = 0

    for i in range(N_ITERS):
        q_true = np.random.uniform(-JOINT_LIMIT, JOINT_LIMIT, dof)
        T = chain.forward_kinematics(q_true)
        target_pose = pose_from_T(T)
        q_seed = q_true + np.random.uniform(-SEED_NOISE, SEED_NOISE, dof)

        # MoveIt: 공정 비교 위해 시드 주입
        names = jmg.active_joint_model_names
        state.joint_positions = {n: float(v) for n, v in zip(names, q_seed)}

        t0 = time.perf_counter()
        ok_m = state.set_from_ik(GROUP, target_pose, TIP_LINK, TIMEOUT)
        dt_m = (time.perf_counter() - t0) * 1000.0
        times_moveit.append(dt_m); succ_moveit += int(ok_m)

        # ikpy
        pos = T[:3, 3]; R = T[:3, :3]
        t0 = time.perf_counter()
        try:
            chain.inverse_kinematics(
                target_position=pos,
                target_orientation=R,
                initial_position=np.resize(q_seed, ikpy_nb_joints),
            )
            ok_i = True
        except Exception:
            ok_i = False
        dt_i = (time.perf_counter() - t0) * 1000.0
        times_ikpy.append(dt_i); succ_ikpy += int(ok_i)

        if (i+1) % max(1, N_ITERS//10) == 0:
            print(f"[{i+1}/{N_ITERS}] MoveIt {dt_m:.3f} ms | ikpy {dt_i:.3f} ms")


    # ---------- 결과 ----------
    s_m = summarize(times_moveit)
    s_i = summarize(times_ikpy)

    print("\n===== IK Benchmark (MoveItPy vs ikpy) =====")
    print(f"iters: {N_ITERS}, seed_noise: ±{SEED_NOISE} rad, joint_limit: ±{JOINT_LIMIT} rad\n")

    print("[MoveItPy set_from_ik]")
    print(f"  success {succ_moveit}/{N_ITERS}")
    for k in ["mean","std","p50","p90","p99","min","max"]:
        print(f"  {k:>4}: {s_m[k]:7.3f} ms")

    print("\n[ikpy inverse_kinematics]")
    print(f"  success {succ_ikpy}/{N_ITERS}")
    for k in ["mean","std","p50","p90","p99","min","max"]:
        print(f"  {k:>4}: {s_i[k]:7.3f} ms")

    if s_m["mean"] < s_i["mean"]:
        print(f"\n➡ MoveItPy IK is faster on average by {s_i['mean']/s_m['mean']:.2f}×")
    else:
        print(f"\n➡ ikpy IK is faster on average by {s_m['mean']/s_i['mean']:.2f}×")

    rclpy.shutdown()

if __name__ == "__main__":
    main()

