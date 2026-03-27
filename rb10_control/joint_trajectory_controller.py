#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RB10 joint_trajectory_controller helpers.

This module is intended to be imported from other local projects as:

    from rb10_control import RB10Controller

while still supporting the legacy script entrypoint in
``scripts/rb10_controller.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
try:
    from trac_ik import TracIK
except ImportError:
    TracIK = None  # type: ignore[assignment]
from rclpy.duration import Duration
from rclpy.node import Node
from rbpodo_msgs.srv import TaskStop
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_from_matrix, quaternion_matrix
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ================= Config =================
JTC_TOPIC = "/joint_trajectory_controller/joint_trajectory"
JOINT_STATES_TOPIC = "/joint_states"
TASK_STOP_SERVICE = "/rbpodo_hardware/task_stop"

BASE_LINK = "link0"
EE_LINK = "tcp"

# ROS 퍼블리시/구독 기준 6개 조인트 이름 (순서 중요)
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

URDF_PATH = "/home/sungboo/ros2_ws/src/rbpodo_ros2/rbpodo_description/robots/rb10_1300e_u.urdf"

# 안전가드
MAX_STEP_PER_JOINT_RAD = 0.27
MAX_STEP_L2_RAD = 0.45

# TRAC-IK 잔차 허용치
IK_ACCEPT_POS_ERR_M = 0.015
IK_ACCEPT_ANG_ERR_DEG = 7.5
IK_ALLOW_APPROXIMATE = True
IK_SCORE_ANG_WEIGHT_M = 0.03
IK_BASE_SEED_OFFSETS_RAD = (0.0, math.pi, -math.pi)

# 디버그 토글
DEBUG = False


@dataclass
class IKCandidate:
    seed6: np.ndarray
    q6: np.ndarray
    pos_err_m: float
    ang_err_rad: float
    score: float
    guard_max_abs_rad: float
    guard_l2_rad: float
    guard_ok: bool
    has_fk: bool


def _fk_pose_to_matrix(pos_fk: Sequence[float], rot_fk: Sequence[Sequence[float]]) -> np.ndarray:
    pos = np.asarray(pos_fk, dtype=float).reshape(3)
    rot = np.asarray(rot_fk, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rot)):
        raise ValueError("FK returned NaN/Inf")
    T = np.eye(4, dtype=float)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def _rot_angle_rad(r_target: np.ndarray, r_actual: np.ndarray) -> float:
    r_delta = np.asarray(r_target, dtype=float).T @ np.asarray(r_actual, dtype=float)
    trace = float(np.trace(r_delta))
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return float(math.acos(cos_theta))


def _quat_normalize(quat_xyzw: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Invalid quaternion: norm is zero or NaN")
    return q / norm


def _quat_slerp_xyzw(q0: Sequence[float], q1: Sequence[float], alpha: float) -> np.ndarray:
    qa = _quat_normalize(q0)
    qb = _quat_normalize(q1)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot

    if dot > 0.9995:
        out = qa + float(alpha) * (qb - qa)
        return _quat_normalize(out)

    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    if abs(sin_theta_0) < 1e-12:
        return qa.copy()

    theta = theta_0 * float(alpha)
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _quat_normalize((s0 * qa) + (s1 * qb))


class RB10Controller(Node):
    """BASE(frame) 기준 pos(3) + quat(xyzw) -> IK -> JointTrajectory publish."""

    def __init__(
        self,
        base_link: str = BASE_LINK,
        ee_link: str = EE_LINK,
        urdf_path: str = URDF_PATH,
        joint_states_topic: str = JOINT_STATES_TOPIC,
        jtc_topic: str = JTC_TOPIC,
        task_stop_service: str = TASK_STOP_SERVICE,
        wait_for_joint_state_sec: float = 5.0,
        accept_pos_err_m: float = IK_ACCEPT_POS_ERR_M,
        accept_ang_err_deg: float = IK_ACCEPT_ANG_ERR_DEG,
        allow_approximate: bool = IK_ALLOW_APPROXIMATE,
    ):
        super().__init__("rb10_controller")

        self.base_link = str(base_link)
        self.ee_link = str(ee_link)
        self.urdf_path = str(urdf_path)
        self.joint_states_topic = str(joint_states_topic)
        self.jtc_topic = str(jtc_topic)
        self.task_stop_service = str(task_stop_service)
        self.accept_pos_err_m = float(accept_pos_err_m)
        self.accept_ang_err_deg = float(accept_ang_err_deg)
        self.allow_approximate = bool(allow_approximate)

        # 외부에서 확인 가능한 최근 IK 실패 사유
        self.last_ik_fail: Optional[str] = None

        # pubs/subs
        self.joint_sub = self.create_subscription(JointState, self.joint_states_topic, self._joint_cb, 10)
        self.traj_pub = self.create_publisher(JointTrajectory, self.jtc_topic, 10)
        self.task_stop_client = self.create_client(TaskStop, self.task_stop_service)

        # TF
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)

        if TracIK is None:
            raise RuntimeError(
                "TRAC-IK backend requires the Python package 'trac_ik'. "
                "Install it in the sourced ROS environment first."
            )

        try:
            self._tracik_solver = TracIK(
                base_link_name=self.base_link,
                tip_link_name=self.ee_link,
                urdf_path=self.urdf_path,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize TRAC-IK: {exc}") from exc

        # 최신 JointState(6개, JOINT_NAMES 순)
        self._latest_positions: Optional[np.ndarray] = None
        self._joint_index_map = None

        # 첫 JointState 대기
        self.get_logger().info("Waiting for first JointState...")
        end = self.get_clock().now().nanoseconds + int(max(0.0, float(wait_for_joint_state_sec)) * 1e9)
        while rclpy.ok() and (self.get_clock().now().nanoseconds < end) and (self._latest_positions is None):
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._latest_positions is None:
            self.get_logger().warn("No joint_states received. joint_states를 확인하세요.")
        else:
            self.get_logger().info("Got initial joint_states.")

    # ---------- 공용 진단 헬퍼 ----------
    def _ik_fail(self, reason: str, extra: Optional[str] = None) -> None:
        self.last_ik_fail = reason if extra is None else f"{reason} | {extra}"
        self.get_logger().warn(self.last_ik_fail)

    # ---------- JointState 콜백 ----------
    def _joint_cb(self, msg: JointState):
        if self._joint_index_map is None:
            self._joint_index_map = {name: i for i, name in enumerate(msg.name)}
            missing = [n for n in JOINT_NAMES if n not in self._joint_index_map]
            if missing:
                self.get_logger().warn(f"JointState에 없는 조인트: {missing}")

        q = np.zeros(len(JOINT_NAMES), dtype=float)
        for k, name in enumerate(JOINT_NAMES):
            idx = self._joint_index_map.get(name)
            if idx is not None and idx < len(msg.position):
                q[k] = msg.position[idx]
        self._latest_positions = q

    def _coerce_q6(self, q6: Sequence[float], name: str = "q6") -> np.ndarray:
        q = np.asarray(q6, dtype=float).reshape(-1)
        if q.shape[0] != len(JOINT_NAMES):
            raise ValueError(f"{name} must have {len(JOINT_NAMES)} elements, got shape {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError(f"{name} contains NaN/Inf")
        return q.copy()

    def _resolve_seed_q6(self, seed_q: Optional[Sequence[float]]) -> Optional[np.ndarray]:
        if seed_q is not None:
            try:
                return self._coerce_q6(seed_q, name="seed_q")
            except Exception as e:
                self._ik_fail("Invalid seed_q", str(e))
                return None
        if self._latest_positions is None:
            self._ik_fail("No joint_states — IK seed unavailable")
            return None
        return np.asarray(self._latest_positions, dtype=float).copy()

    # ---------- FK ----------
    def _fk_current_T(self) -> Optional[np.ndarray]:
        if self._latest_positions is None:
            return None
        try:
            pos_fk, rot_fk = self._tracik_solver.fk(np.asarray(self._latest_positions, dtype=float))
            return _fk_pose_to_matrix(pos_fk, rot_fk)
        except Exception as e:
            if DEBUG:
                self.get_logger().warn(f"FK 실패: {e}")
            return None

    def _fk_current_T_of(self, q6: np.ndarray) -> Optional[np.ndarray]:
        try:
            q = self._coerce_q6(q6, name="q6")
            pos_fk, rot_fk = self._tracik_solver.fk(q)
            return _fk_pose_to_matrix(pos_fk, rot_fk)
        except Exception:
            return None

    # ---------- 안전가드 ----------
    @staticmethod
    def _wrap_pi(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _format_array(values: Sequence[float], precision: int = 4) -> str:
        return np.array2string(np.asarray(values, dtype=float), precision=precision, suppress_small=False)

    def _guard_metrics(self, q6: np.ndarray, seed6: np.ndarray) -> tuple[float, float]:
        diffs = self._wrap_pi(np.asarray(q6, dtype=float) - np.asarray(seed6, dtype=float))
        return float(np.max(np.abs(diffs))), float(np.linalg.norm(diffs))

    def _guard_ok(self, q6: np.ndarray, seed6: np.ndarray) -> bool:
        max_abs, l2 = self._guard_metrics(q6, seed6)
        if (max_abs > MAX_STEP_PER_JOINT_RAD) or (l2 > MAX_STEP_L2_RAD):
            self._ik_fail(f"Guard reject: Δq too large (max={max_abs:.3f} rad, L2={l2:.3f} rad)")
            return False
        return True

    @staticmethod
    def _wrap_pi_scalar(value: float) -> float:
        return float((value + math.pi) % (2.0 * math.pi) - math.pi)

    def _make_seed_candidates(self, seed6: np.ndarray, target_pos: np.ndarray) -> List[np.ndarray]:
        heading = math.atan2(float(target_pos[1]), float(target_pos[0]))
        candidates: List[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()

        def add_candidate(q6: Sequence[float]) -> None:
            q = self._coerce_q6(q6, name="seed_candidate")
            q = self._wrap_pi(q)
            key = tuple(np.round(q, decimals=6).tolist())
            if key in seen:
                return
            seen.add(key)
            candidates.append(q)

        add_candidate(seed6)

        for offset in IK_BASE_SEED_OFFSETS_RAD:
            q = np.asarray(seed6, dtype=float).copy()
            q[0] = self._wrap_pi_scalar(heading + float(offset))
            add_candidate(q)

        zero_seed = np.zeros_like(seed6)
        add_candidate(zero_seed)
        zero_heading = zero_seed.copy()
        zero_heading[0] = self._wrap_pi_scalar(heading)
        add_candidate(zero_heading)

        return candidates

    def _solve_candidate(
        self,
        candidate_seed6: np.ndarray,
        actual_seed6: np.ndarray,
        target_pos: np.ndarray,
        target_rotmat: np.ndarray,
    ) -> tuple[Optional[IKCandidate], Optional[str]]:
        try:
            result = self._tracik_solver.ik(
                target_pos,
                target_rotmat,
                seed_jnt_values=np.asarray(candidate_seed6, dtype=float),
            )
        except Exception as exc:
            return None, f"TRAC-IK exception: {exc}"

        if result is None:
            return None, None

        try:
            q6 = self._coerce_q6(result, name="q6")
        except Exception as exc:
            return None, f"TRAC-IK returned invalid joint solution: {exc}"

        guard_max_abs_rad, guard_l2_rad = self._guard_metrics(q6, actual_seed6)
        guard_ok = (guard_max_abs_rad <= MAX_STEP_PER_JOINT_RAD) and (guard_l2_rad <= MAX_STEP_L2_RAD)

        T = self._fk_current_T_of(q6)
        has_fk = T is not None
        pos_err_m = 0.0
        ang_err_rad = 0.0
        if has_fk:
            fk_pos = np.asarray(T[:3, 3], dtype=float)
            fk_rot = np.asarray(T[:3, :3], dtype=float)
            pos_err_m = float(np.linalg.norm(fk_pos - target_pos))
            ang_err_rad = _rot_angle_rad(target_rotmat, fk_rot)

        score = float(
            pos_err_m
            + (IK_SCORE_ANG_WEIGHT_M * ang_err_rad)
            + (1e-4 * guard_l2_rad)
        )
        return (
            IKCandidate(
                seed6=np.asarray(candidate_seed6, dtype=float).copy(),
                q6=np.asarray(q6, dtype=float).copy(),
                pos_err_m=pos_err_m,
                ang_err_rad=ang_err_rad,
                score=score,
                guard_max_abs_rad=guard_max_abs_rad,
                guard_l2_rad=guard_l2_rad,
                guard_ok=guard_ok,
                has_fk=has_fk,
            ),
            None,
        )

    def _candidate_summary(self, candidate: IKCandidate) -> str:
        summary = (
            f"seed={self._format_array(candidate.seed6)} | "
            f"q={self._format_array(candidate.q6)} | "
            f"guard_max={candidate.guard_max_abs_rad:.3f} rad | "
            f"guard_l2={candidate.guard_l2_rad:.3f} rad"
        )
        if candidate.has_fk:
            summary += (
                f" | pos_err={candidate.pos_err_m:.4f} m"
                f" | ang_err={math.degrees(candidate.ang_err_rad):.2f} deg"
            )
        return summary

    def _interpolate_pose(
        self,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        end_pos: np.ndarray,
        end_quat: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos = ((1.0 - float(alpha)) * np.asarray(start_pos, dtype=float)) + (float(alpha) * np.asarray(end_pos, dtype=float))
        quat = _quat_slerp_xyzw(start_quat, end_quat, alpha)
        return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)

    def _refine_guard_failing_segment(
        self,
        start_ee_pos: np.ndarray,
        start_ee_rot: np.ndarray,
        target_ee_pos: np.ndarray,
        target_ee_rot: np.ndarray,
        duration: float,
        seed_q: np.ndarray,
        enforce_guard: bool,
        remaining_depth: int,
    ) -> Optional[tuple[List[np.ndarray], List[float]]]:
        q_goal = self.compute_target_qpos_from_pose(
            target_ee_pos,
            target_ee_rot,
            enforce_guard=enforce_guard,
            seed_q=seed_q,
        )
        if q_goal is not None:
            return [np.asarray(q_goal, dtype=float)], [float(duration)]

        prev_reason = self.last_ik_fail or "unknown IK failure"
        if remaining_depth <= 0 or "guard reject" not in prev_reason.lower():
            return None

        mid_pos, mid_rot = self._interpolate_pose(start_ee_pos, start_ee_rot, target_ee_pos, target_ee_rot, 0.5)
        first_half = self._refine_guard_failing_segment(
            start_ee_pos=start_ee_pos,
            start_ee_rot=start_ee_rot,
            target_ee_pos=mid_pos,
            target_ee_rot=mid_rot,
            duration=float(duration) * 0.5,
            seed_q=np.asarray(seed_q, dtype=float),
            enforce_guard=enforce_guard,
            remaining_depth=remaining_depth - 1,
        )
        if first_half is None:
            return None

        first_q_path, first_durations = first_half
        second_half = self._refine_guard_failing_segment(
            start_ee_pos=mid_pos,
            start_ee_rot=mid_rot,
            target_ee_pos=target_ee_pos,
            target_ee_rot=target_ee_rot,
            duration=float(duration) * 0.5,
            seed_q=np.asarray(first_q_path[-1], dtype=float),
            enforce_guard=enforce_guard,
            remaining_depth=remaining_depth - 1,
        )
        if second_half is None:
            return None

        second_q_path, second_durations = second_half
        return first_q_path + second_q_path, first_durations + second_durations

    # ---------- IK ----------
    def compute_target_qpos_from_pose(
        self,
        target_ee_pos: np.ndarray,
        target_ee_rot: np.ndarray,
        enforce_guard: bool = True,
        seed_q: Optional[Sequence[float]] = None,
    ) -> Optional[np.ndarray]:
        """Compute joint angles for a BASE-frame TCP pose target."""
        seed6 = self._resolve_seed_q6(seed_q)
        if seed6 is None:
            return None

        p = np.asarray(target_ee_pos, dtype=float).reshape(3,)
        q = np.asarray(target_ee_rot, dtype=float).reshape(4,)
        n = float(np.linalg.norm(q))
        if not np.isfinite(n) or n <= 0:
            self._ik_fail("Invalid target quaternion (norm <= 0 or NaN)")
            return None
        q /= n
        r_tgt = quaternion_matrix(q)[:3, :3]

        evaluated: List[IKCandidate] = []
        last_error: Optional[str] = None
        for candidate_seed6 in self._make_seed_candidates(seed6, p):
            candidate, error = self._solve_candidate(candidate_seed6, seed6, p, r_tgt)
            if error is not None:
                last_error = error
            if candidate is not None:
                evaluated.append(candidate)

        if not evaluated:
            extra = f"target_pos={self._format_array(p)}"
            if last_error is not None:
                extra = f"{extra} | {last_error}"
            self._ik_fail("TRAC-IK returned no solution", extra)
            return None

        if enforce_guard:
            guarded = [candidate for candidate in evaluated if candidate.guard_ok]
            if not guarded:
                best_any = min(evaluated, key=lambda candidate: candidate.score)
                self._ik_fail("TRAC-IK guard reject after multi-seed", self._candidate_summary(best_any))
                return None
            evaluated = guarded

        best = min(evaluated, key=lambda candidate: candidate.score)
        if best.has_fk:
            ang_err_deg = math.degrees(best.ang_err_rad)
            summary = self._candidate_summary(best)
            if (best.pos_err_m > self.accept_pos_err_m) or (ang_err_deg > self.accept_ang_err_deg):
                if not self.allow_approximate:
                    self._ik_fail("TRAC-IK rejected approximate solution", summary)
                    return None
                self.get_logger().warn(f"TRAC-IK accepting approximate solution | {summary}")

        self.last_ik_fail = None
        return np.asarray(best.q6, dtype=float).copy()

    def compute_joint_path_from_pose_sequence(
        self,
        target_ee_positions: np.ndarray,
        target_ee_rots: np.ndarray,
        enforce_guard: bool = True,
        seed_q: Optional[Sequence[float]] = None,
    ) -> Optional[List[np.ndarray]]:
        pos = np.asarray(target_ee_positions, dtype=float)
        rot = np.asarray(target_ee_rots, dtype=float)

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"target_ee_positions must be (N,3), got {pos.shape}")
        if rot.ndim != 2 or rot.shape[1] != 4:
            raise ValueError(f"target_ee_rots must be (N,4) [xyzw], got {rot.shape}")
        if pos.shape[0] != rot.shape[0]:
            raise ValueError("target_ee_positions and target_ee_rots length mismatch")

        q_seed = self._resolve_seed_q6(seed_q)
        if q_seed is None:
            return None

        q_path: List[np.ndarray] = []
        for i in range(pos.shape[0]):
            q_goal = self.compute_target_qpos_from_pose(
                pos[i],
                rot[i],
                enforce_guard=enforce_guard,
                seed_q=q_seed,
            )
            if q_goal is None:
                prev_reason = self.last_ik_fail or "unknown IK failure"
                self._ik_fail(f"Sequence IK failed at index {i}", prev_reason)
                return None
            q_goal = np.asarray(q_goal, dtype=float)
            q_path.append(q_goal)
            q_seed = q_goal

        return q_path

    # ---------- Trajectory publish ----------
    def publish_qpos(
        self,
        q_goal: List[float],
        duration: float = 0.3,
        min_point_duration: float = 0.20,
    ) -> bool:
        # Backward-compatible convenience wrapper for single-point commands.
        return self.publish_joint_trajectory([q_goal], [duration], min_point_duration=min_point_duration)

    def publish_joint_trajectory(
        self,
        q_goals: Sequence[Sequence[float]],
        durations: Sequence[float],
        min_point_duration: float = 0.20,
    ) -> bool:
        if len(q_goals) == 0:
            raise ValueError("q_goals must not be empty")
        if len(q_goals) != len(durations):
            raise ValueError("q_goals and durations length mismatch")

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        traj.points = []

        elapsed = 0.0
        min_dt = max(0.0, float(min_point_duration))
        for i, (q_goal, duration) in enumerate(zip(q_goals, durations)):
            q = self._coerce_q6(q_goal, name=f"q_goal[{i}]")
            dt = float(duration)
            if not np.isfinite(dt):
                raise ValueError(f"duration[{i}] is NaN/Inf")
            elapsed += max(min_dt, dt)

            p = JointTrajectoryPoint()
            p.positions = q.tolist()
            p.time_from_start = Duration(seconds=elapsed).to_msg()
            traj.points.append(p)

        self.traj_pub.publish(traj)
        if DEBUG:
            self.get_logger().info(
                f"Trajectory published (points={len(traj.points)}, total_duration={elapsed:.2f}s)"
            )
        return True

    def execute_pose_sequence(
        self,
        target_ee_positions: np.ndarray,
        target_ee_rots: np.ndarray,
        point_durations: Sequence[float],
        enforce_guard: bool = True,
        seed_q: Optional[Sequence[float]] = None,
        min_point_duration: float = 0.20,
        start_ee_position: Optional[np.ndarray] = None,
        start_ee_rot: Optional[np.ndarray] = None,
        max_guard_refine_depth: int = 0,
    ) -> Optional[List[np.ndarray]]:
        if len(point_durations) != int(np.asarray(target_ee_positions).shape[0]):
            raise ValueError("point_durations length must match the number of target poses")
        pos = np.asarray(target_ee_positions, dtype=float)
        rot = np.asarray(target_ee_rots, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"target_ee_positions must be (N,3), got {pos.shape}")
        if rot.ndim != 2 or rot.shape[1] != 4:
            raise ValueError(f"target_ee_rots must be (N,4) [xyzw], got {rot.shape}")
        if pos.shape[0] != rot.shape[0]:
            raise ValueError("target_ee_positions and target_ee_rots length mismatch")

        q_seed = self._resolve_seed_q6(seed_q)
        if q_seed is None:
            return None

        start_pos = None
        start_quat = None
        if start_ee_position is not None and start_ee_rot is not None:
            start_pos = np.asarray(start_ee_position, dtype=float).reshape(3,)
            start_quat = _quat_normalize(start_ee_rot)
        else:
            current_pose = self.get_current_ee_pose()
            if current_pose is not None:
                start_pos = np.asarray(current_pose[0], dtype=float).reshape(3,)
                start_quat = _quat_normalize(current_pose[1])

        q_path: List[np.ndarray] = []
        publish_durations: List[float] = []
        for i in range(pos.shape[0]):
            q_goal = self.compute_target_qpos_from_pose(
                pos[i],
                rot[i],
                enforce_guard=enforce_guard,
                seed_q=q_seed,
            )
            if q_goal is None:
                segment_start_pos = start_pos if i == 0 else pos[i - 1]
                segment_start_rot = start_quat if i == 0 else rot[i - 1]
                refined = None
                if (
                    int(max_guard_refine_depth) > 0
                    and segment_start_pos is not None
                    and segment_start_rot is not None
                ):
                    refined = self._refine_guard_failing_segment(
                        start_ee_pos=np.asarray(segment_start_pos, dtype=float),
                        start_ee_rot=np.asarray(segment_start_rot, dtype=float),
                        target_ee_pos=pos[i],
                        target_ee_rot=rot[i],
                        duration=float(point_durations[i]),
                        seed_q=np.asarray(q_seed, dtype=float),
                        enforce_guard=enforce_guard,
                        remaining_depth=int(max_guard_refine_depth),
                    )
                if refined is None:
                    prev_reason = self.last_ik_fail or "unknown IK failure"
                    self._ik_fail(f"Sequence IK failed at index {i}", prev_reason)
                    return None

                refined_q_path, refined_durations = refined
                q_path.extend(np.asarray(q_refined, dtype=float) for q_refined in refined_q_path)
                publish_durations.extend(float(duration) for duration in refined_durations)
                q_seed = np.asarray(refined_q_path[-1], dtype=float)
                continue

            q_goal = np.asarray(q_goal, dtype=float)
            q_path.append(q_goal)
            publish_durations.append(float(point_durations[i]))
            q_seed = q_goal

        self.publish_joint_trajectory(q_path, publish_durations, min_point_duration=min_point_duration)
        return q_path

    def emergency_stop(self, timeout: float = 1.0, wait_for_service_sec: float = 0.5) -> bool:
        req = TaskStop.Request()
        req.timeout = float(timeout)

        if not self.task_stop_client.wait_for_service(timeout_sec=max(0.0, float(wait_for_service_sec))):
            self.get_logger().warn(f"TaskStop service unavailable: {self.task_stop_service}")
            return False

        future = self.task_stop_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=max(0.0, float(timeout)))

        if not future.done():
            self.get_logger().warn("TaskStop service call timed out")
            return False

        exc = future.exception()
        if exc is not None:
            self.get_logger().warn(f"TaskStop service call failed: {exc}")
            return False

        result = future.result()
        if result is None:
            self.get_logger().warn("TaskStop service returned no response")
            return False

        ok = bool(result.success)
        if ok:
            self.get_logger().warn("Emergency stop requested via TaskStop")
        else:
            self.get_logger().warn("TaskStop service responded with success=False")
        return ok

    # ---------- 상태 조회 ----------
    def get_chain_tip_link_name(self) -> str:
        return str(self.ee_link)

    def get_current_joint_positions(self) -> Optional[np.ndarray]:
        if self._latest_positions is None:
            return None
        return np.asarray(self._latest_positions, dtype=float).copy()

    def get_current_ee_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(pos[3], quat[4]) — FK 우선, 실패 시 TF 조회."""
        t = self._fk_current_T()
        if t is not None:
            return t[:3, 3].astype(float), np.asarray(quaternion_from_matrix(t), dtype=float)
        try:
            tf = self.tf_buf.lookup_transform(self.base_link, self.ee_link, rclpy.time.Time())
            pos = np.array(
                [
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                ],
                dtype=float,
            )
            quat = np.array(
                [
                    tf.transform.rotation.x,
                    tf.transform.rotation.y,
                    tf.transform.rotation.z,
                    tf.transform.rotation.w,
                ],
                dtype=float,
            )
            return pos, quat
        except Exception:
            return None


def main() -> None:
    rclpy.init()
    node = RB10Controller()
    try:
        if DEBUG:
            node.get_logger().info(f"Current EE pose: {node.get_current_ee_pose()}")

        target_ee_pos = [0.7, 0.1, 0.3]
        target_ee_rot_xyzw = [0, -0.7071068, -0.7071068, 0]

        q = node.compute_target_qpos_from_pose(target_ee_pos, target_ee_rot_xyzw, enforce_guard=False)
        if q is not None:
            node.publish_joint_trajectory([q.tolist()], [10.0])
            if DEBUG:
                node.get_logger().info("Moved to target pose.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


__all__ = [
    "BASE_LINK",
    "DEBUG",
    "EE_LINK",
    "JOINT_NAMES",
    "JOINT_STATES_TOPIC",
    "JTC_TOPIC",
    "MAX_STEP_L2_RAD",
    "MAX_STEP_PER_JOINT_RAD",
    "RB10Controller",
    "TASK_STOP_SERVICE",
    "URDF_PATH",
    "main",
]
