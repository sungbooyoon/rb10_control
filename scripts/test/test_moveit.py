#!/usr/bin/env python3
import time
import rclpy
from rclpy.logging import get_logger
from geometry_msgs.msg import Pose
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from moveit_configs_utils import MoveItConfigsBuilder

GROUP   = "manipulator"   # SRDF 그룹명
TIP     = "tcp"           # EE 링크명 (환경에 맞게 수정)
TIMEOUT = 0.02            # 20 ms
N       = 200             # 벤치 반복 수

def main():
    rclpy.init()
    log = get_logger("ik_inproc_moveitpy")

    # MoveIt 설정은 이미 패키지에 있다고 가정 (URDF/SRDF/kinematics/OMPL은 패키지 쪽)
    mc = (
        MoveItConfigsBuilder("rbpodo", package_name="rbpodo_moveit_config")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .moveit_cpp(file_path="/home/sungboo/ros2_ws/src/rb10_control/config/moveit_py_config.yaml")
        .to_moveit_configs()
    )

    robot = MoveItPy(node_name="moveit_py", config_dict=mc.to_dict())
    model = robot.get_robot_model()

    jmg   = model.get_joint_model_group(GROUP)  # 정보 조회용
    state = RobotState(model)
    state.set_to_default_values()               # seed 초기화

    # 목표 포즈 (예시)
    target = Pose()
    target.position.x, target.position.y, target.position.z = 0.5, 0.0, 0.4
    target.orientation.w = 1.0

    # IK 호출 (문자열 그룹명, Pose, TIP 링크명, timeout)
    ok = state.set_from_ik(GROUP, target, TIP, TIMEOUT)
    if not ok:
        log.warning("IK failed")
        rclpy.shutdown()
        return

    # ---- 결과 joint vector 추출 (snake_case 전제) ----
    joint_names = jmg.active_joint_model_names           # list
    jp_dict     = state.joint_positions                  # dict: name -> float | [float]

    def pos_of(name: str) -> float:
        v = jp_dict[name]
        return float(v[0] if isinstance(v, (list, tuple)) else v)

    q = [round(pos_of(n), 6) for n in joint_names]
    print("[IK] q:", q)

    # ---- 간단 벤치마크 ----
    t0 = time.perf_counter()
    succ = 0
    for _ in range(N):
        if state.set_from_ik(GROUP, target, TIP, TIMEOUT):
            succ += 1
    avg_ms = (time.perf_counter() - t0) * 1000.0 / N
    print(f"[Bench] IK avg: {avg_ms:.3f} ms over {N} iters (success {succ}/{N})")

    rclpy.shutdown()

if __name__ == "__main__":
    main()
