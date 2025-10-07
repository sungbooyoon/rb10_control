import time
import numpy as np

from ikpy.chain import Chain
import roboticstoolbox as rtb
from spatialmath import SE3

# =======================
# 설정
# =======================
URDF_PATH  = "/home/sungboo/ros2_ws/src/rbpodo_ros2/rbpodo_description/robots/rb10_1300e_u.urdf"
N_ITER = 1000
# [x, y, z, roll, pitch, yaw]  (라디안)
TARGET_POSE = [0.5, 0.0, 0.4, 0.0, 0.0, 0.0]
SEED = None  # 초기값. None이면 각 라이브러리 기본값 사용

# =======================
# 1) ikpy 초기화
# =======================
print("=== ikpy 초기화 ===")
chain = Chain.from_urdf_file(URDF_PATH)
base_link_name = chain.links[0].name
ee_link_name   = chain.links[-1].name
n_dof = chain.get_dof()

# =======================
# 2) RTB-P 초기화 (URDF → ERobot)
# =======================
print("=== RTB-P 초기화 (URDF 로드 시도) ===")
robot = None
try:
    # RTB-P 1.x: ERobot.URDF(...) 로드 (버전에 따라 이름공간이 다를 수 있어 try/except)
    robot = rtb.ERobot.URDF(URDF_PATH)
    print(f"URDF 로드 성공: dof={robot.n}")
except Exception as e:
    print(f"[경고] URDF 로드 실패: {e}")
    print("→ 내장 모델(Puma560)로 대체해서 벤치마크만 수행합니다.")
    robot = rtb.models.DH.Puma560()

# =======================
# ikpy 벤치마크
# =======================
print("=== ikpy 속도 테스트 시작 ===")
start = time.perf_counter()
for _ in range(N_ITER):
    # ikpy는 4x4 동차 변환행렬을 목표로 받음
    T = np.eye(4)
    T[:3, 3] = TARGET_POSE[:3]
    # 필요하면 RPY→R 변환해서 T[:3,:3] 채움 (여기선 0,0,0)
    chain.inverse_kinematics(
        T,
        initial_position=(SEED if SEED is not None else [0.0]*n_dof)
    )
end = time.perf_counter()
ikpy_avg_ms = (end - start) * 1000.0 / N_ITER
print(f"[ikpy] 평균 수행 시간: {ikpy_avg_ms:.3f} ms")

# =======================
# RTB-P 벤치마크 (ikine_LM)
# =======================
print("=== RTB-P(ikine_LM) 속도 테스트 시작 ===")
# 목표 자세
x, y, z, r, p, yw = TARGET_POSE
Tep = SE3.Trans(x, y, z) * SE3.RPY([r, p, yw], order="xyz")

# 초기값
q0 = None
if hasattr(robot, "qz"):
    q0 = robot.qz
elif hasattr(robot, "q"):
    q0 = robot.q
else:
    q0 = np.zeros(robot.n)

if SEED is not None:
    q0 = np.array(SEED[:robot.n], dtype=float)

start = time.perf_counter()
for _ in range(N_ITER):
    sol = robot.ikine_LM(Tep, q0=q0)  # LM: ms급, 수렴성/안정성 좋음
    # 원하면 다음 반복의 초기값으로 갱신 (수렴성 개선)
    # q0 = sol.q
end = time.perf_counter()
rtb_avg_ms = (end - start) * 1000.0 / N_ITER
print(f"[RTB-P:ikine_LM] 평균 수행 시간: {rtb_avg_ms:.3f} ms")

# =======================
# 결과 요약
# =======================
print("\n===== 결과 요약 =====")
print(f"ikpy       : {ikpy_avg_ms:.3f} ms")
print(f"RTB ikine  : {rtb_avg_ms:.3f} ms")
if rtb_avg_ms < ikpy_avg_ms:
    print(f"RTB가 ikpy보다 {(ikpy_avg_ms / rtb_avg_ms):.2f}배 빠름")
else:
    print("ikpy가 RTB보다 빠름(모델/초기값/상용량에 따라 달라질 수 있음)")
