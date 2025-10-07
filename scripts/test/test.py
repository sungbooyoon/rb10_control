import time
import numpy as np

from ikpy.chain import Chain
from tracikpy import TracIKSolver

# ✅ 전역변수: 로봇 URDF 파일 경로 (사용자가 직접 지정)
URDF_PATH  = "/home/sungboo/ros2_ws/src/rbpodo_ros2/rbpodo_description/robots/rb10_1300e_u.urdf"

# ---------------------------
# 테스트 설정값
# ---------------------------
N_ITER = 1000  # 반복 횟수
TARGET_POSE = [0.5, 0.0, 0.4, 0.0, 0.0, 0.0]  # [x, y, z, roll, pitch, yaw]
SEED = None    # 초기 joint guess (None이면 라이브러리 기본 사용)

# ---------------------------
# 1) ikpy 세팅
# ---------------------------
print("=== ikpy 초기화 중 ===")
chain = Chain.from_urdf_file(URDF_PATH)
# end-effector 링크 이름이 URDF마다 다를 수 있음 -> 필요시 수정

# ---------------------------
# 2) tracikpy 세팅
# ---------------------------
print("=== tracikpy 초기화 중 ===")
base_link = chain.links[0].name
ee_link = chain.links[-1].name
solver = TracIKSolver(base_link, ee_link, URDF_PATH)

# ---------------------------
# ikpy 벤치마크
# ---------------------------
print("=== ikpy 속도 테스트 시작 ===")
start = time.perf_counter()
for _ in range(N_ITER):
    # ikpy는 target frame을 4x4 homogeneous matrix로 줘야 함
    target_matrix = chain.forward_kinematics([0]*len(chain.links))  # identity 기반
    target_matrix[:3, 3] = TARGET_POSE[:3]  # 위치 적용 (회전은 생략 or 필요시 변환)
    chain.inverse_kinematics(target_matrix, initial_position=SEED)
end = time.perf_counter()
ikpy_avg_ms = (end - start) * 1000 / N_ITER
print(f"[ikpy] 평균 수행 시간: {ikpy_avg_ms:.3f} ms")

# ---------------------------
# tracikpy 벤치마크
# ---------------------------
print("=== tracikpy 속도 테스트 시작 ===")
start = time.perf_counter()
for _ in range(N_ITER):
    solver.solve(TARGET_POSE, SEED)
end = time.perf_counter()
tracik_avg_ms = (end - start) * 1000 / N_ITER
print(f"[tracikpy] 평균 수행 시간: {tracik_avg_ms:.3f} ms")

# ---------------------------
# 결과 요약
# ---------------------------
print("\n===== 결과 요약 =====")
print(f"ikpy     : {ikpy_avg_ms:.3f} ms")
print(f"tracikpy : {tracik_avg_ms:.3f} ms")
print(f"tracikpy가 ikpy보다 {(ikpy_avg_ms / tracik_avg_ms):.2f}배 빠름" if tracik_avg_ms < ikpy_avg_ms else "ikpy가 tracikpy보다 빠름")
