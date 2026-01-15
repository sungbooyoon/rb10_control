#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rosbag2 -> /joint_states 추출 -> 요약 출력 -> 사용자 승인 시 joint_trajectory_controller로 재생

- /joint_states 토픽은 고정(JOINT_STATES_TOPIC)
- joint_trajectory_controller target topic은 RB10Controller.publish_qpos 설정을 따름
- RB10Controller(퍼블리셔 + joint cache + publish_qpos)를 최대한 재사용
- warmup은 duration만 사용(첫 bag pose로 한 번 이동)
- /rb/* 상태 기록 토픽은 별도 RbBridge 노드에서 publish (이 스크립트는 재생만 담당)

Usage:
  python3 rosbag_replay.py --bag /path/to/bag_folder
  python3 rosbag_replay.py --bag /path/to/bag_folder --rate 0.5
  python3 rosbag_replay.py --bag /path/to/bag_folder --warmup-duration 3.0
"""

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import rclpy

from rb10_controller import (
    RB10Controller,
    JOINT_NAMES,
)


@dataclass
class Sample:
    t_ns: int
    q: List[float]


@dataclass
class BagSummary:
    bag_path: str
    msg_count: int
    start_t_ns: int
    end_t_ns: int
    duration_sec: float
    approx_hz: float
    dt_min_ms: Optional[float]
    dt_med_ms: Optional[float]
    dt_mean_ms: Optional[float]
    dt_max_ms: Optional[float]
    joint_names_expected: List[str]
    joint_names_in_bag: List[str]
    missing_joints: List[str]


JOINT_STATES_TOPIC = "/joint_states"


def _median(xs: List[float]) -> float:
    xs2 = sorted(xs)
    n = len(xs2)
    mid = n // 2
    if n % 2 == 1:
        return xs2[mid]
    return 0.5 * (xs2[mid - 1] + xs2[mid])


def read_joint_states_from_bag(
    bag_path: str,
    joint_names: List[str],
    storage_id: str = "mcap",
) -> Tuple[List[Sample], BagSummary]:
    """
    Read rosbag2 and extract /joint_states into time-sorted list of Sample(t_ns, q6).
    Also returns a summary for user approval.
    """
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    topic_type = {t.name: t.type for t in topics}

    if JOINT_STATES_TOPIC not in topic_type:
        available = ", ".join(sorted(topic_type.keys()))
        raise RuntimeError(f"Bag에 '{JOINT_STATES_TOPIC}' 토픽이 없습니다. Available: {available}")

    msg_type = get_message(topic_type[JOINT_STATES_TOPIC])

    samples: List[Sample] = []
    name_to_idx: Optional[Dict[str, int]] = None
    joint_names_in_bag: List[str] = []
    missing_joints: List[str] = []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic != JOINT_STATES_TOPIC:
            continue

        msg = deserialize_message(data, msg_type)  # sensor_msgs/JointState

        if name_to_idx is None:
            joint_names_in_bag = list(msg.name)
            name_to_idx = {n: i for i, n in enumerate(msg.name)}
            missing_joints = [n for n in joint_names if n not in name_to_idx]

        if name_to_idx is None:
            continue

        if missing_joints:
            raise RuntimeError(
                f"/joint_states에 없는 조인트가 있어요: {missing_joints}\n"
                f"bag msg.name = {joint_names_in_bag}\n"
                f"JOINT_NAMES = {joint_names}"
            )

        q: List[float] = []
        for jn in joint_names:
            idx = name_to_idx[jn]
            if idx >= len(msg.position):
                raise RuntimeError(
                    f"JointState.position 길이가 부족합니다. idx={idx}, len={len(msg.position)}"
                )
            q.append(float(msg.position[idx]))

        samples.append(Sample(t_ns=int(t_ns), q=q))

    samples.sort(key=lambda s: s.t_ns)

    if not samples:
        raise RuntimeError(f"'{JOINT_STATES_TOPIC}'에서 읽힌 메시지가 0개입니다.")

    start_t = samples[0].t_ns
    end_t = samples[-1].t_ns
    duration_sec = max(0.0, (end_t - start_t) / 1e9)

    dts_ms: List[float] = []
    for i in range(1, len(samples)):
        dt_ns = samples[i].t_ns - samples[i - 1].t_ns
        if dt_ns > 0:
            dts_ms.append(dt_ns / 1e6)

    if dts_ms:
        dt_min = min(dts_ms)
        dt_max = max(dts_ms)
        dt_mean = sum(dts_ms) / len(dts_ms)
        dt_med = _median(dts_ms)
        approx_hz = (len(samples) - 1) / duration_sec if duration_sec > 0 else float("inf")
    else:
        dt_min = dt_max = dt_mean = dt_med = None
        approx_hz = float("inf")

    summary = BagSummary(
        bag_path=bag_path,
        msg_count=len(samples),
        start_t_ns=start_t,
        end_t_ns=end_t,
        duration_sec=duration_sec,
        approx_hz=approx_hz if math.isfinite(approx_hz) else 0.0,
        dt_min_ms=dt_min,
        dt_med_ms=dt_med,
        dt_mean_ms=dt_mean,
        dt_max_ms=dt_max,
        joint_names_expected=list(joint_names),
        joint_names_in_bag=joint_names_in_bag,
        missing_joints=missing_joints,
    )

    return samples, summary


def print_summary(s: BagSummary) -> None:
    def fmt(x: Optional[float]) -> str:
        return "N/A" if x is None else f"{x:.3f}"

    print("\n==================== ROSBAG SUMMARY ====================")
    print(f"Bag path          : {s.bag_path}")
    print(f"Message count     : {s.msg_count}")
    print(f"Duration          : {s.duration_sec:.3f} s")
    print(f"Approx frequency  : {s.approx_hz:.2f} Hz (rough)")
    print(f"dt (ms) min/med/mean/max : {fmt(s.dt_min_ms)} / {fmt(s.dt_med_ms)} / {fmt(s.dt_mean_ms)} / {fmt(s.dt_max_ms)}")
    print("\nJoint name check")
    print(f"Expected JOINT_NAMES: {s.joint_names_expected}")
    if s.joint_names_in_bag:
        print(f"Bag joint_states.name: {s.joint_names_in_bag}")
    if s.missing_joints:
        print(f"!! Missing joints : {s.missing_joints}")
    else:
        print("OK: all expected joints exist in bag /joint_states")
    print("========================================================\n")


class BagJointTrajectoryPlayer(RB10Controller):
    def __init__(
        self,
        samples: List[Sample],
        rate: float = 1.0,
        point_duration: float = 0.10,
        warmup_duration: float = 3.0,
    ):
        # NOTE: RB10Controller가 node_name/wait_joint_state_sec를 받지 않으면 여기서 제거하세요.
        super().__init__()

        self.samples = samples
        self.rate = float(rate)
        self.point_duration = float(point_duration)
        self.warmup_duration = float(warmup_duration)

        if self.rate <= 0.0:
            raise ValueError(f"rate must be > 0, got {self.rate}")
        if self.point_duration <= 0.0:
            raise ValueError(f"point_duration must be > 0, got {self.point_duration}")
        if self.warmup_duration < 0.0:
            raise ValueError(f"warmup_duration must be >= 0, got {self.warmup_duration}")

        # internal state flags
        self._playback_approved = False

        self.i = 0
        self.t0_bag = self.samples[0].t_ns
        self.t0_wall = self.get_clock().now().nanoseconds

        # warmup first (best-effort)
        did_warmup = self._warmup_to_first_sample()
        if did_warmup:
            self._warmup_end_ns = self.get_clock().now().nanoseconds + int(1e9 * max(0.0, self.warmup_duration))
        else:
            self._warmup_end_ns = self.get_clock().now().nanoseconds

        # Initialize timing for playback but do not start until armed
        self._armed = False

        # tick
        self.timer = self.create_timer(1.0 / 200.0, self._tick)
        self.max_points_per_tick = 5  # safety cap to avoid flooding the controller

        self.get_logger().info(
            f"Loaded samples: {len(self.samples)} | rate={self.rate} | point_duration={self.point_duration}s | warmup_duration={self.warmup_duration}s"
        )

    def arm_playback(self) -> None:
        # start playback from current wall time after warmup end
        now = self.get_clock().now().nanoseconds
        start_wall = max(now, getattr(self, "_warmup_end_ns", now))
        self.t0_wall = start_wall
        self._armed = True
        self.get_logger().info("Playback armed.")

    def _warmup_to_first_sample(self) -> bool:
        if self.warmup_duration <= 0.0:
            return False

        # RB10Controller가 joint_states를 구독하고 _latest_positions를 채운다는 전제
        if getattr(self, "_latest_positions", None) is None:
            self.get_logger().warn("Warmup skipped: no live /joint_states received yet.")
            return False

        q1 = self.samples[self.i].q if self.i < len(self.samples) else self.samples[0].q
        self.get_logger().warn(
            f"Warmup: moving to first bag q over {self.warmup_duration:.2f}s."
        )
        self.publish_qpos(q1, duration=self.warmup_duration)

        return True


    def _tick(self):
        now_wall = self.get_clock().now().nanoseconds
        if now_wall < getattr(self, "_warmup_end_ns", now_wall):
            return  # still warming up
        if not getattr(self, "_armed", False):
            return  # waiting for second approval

        if self.i >= len(self.samples):
            self.get_logger().info("Done.")
            self.timer.cancel()
            return

        elapsed_wall = now_wall - self.t0_wall
        target_elapsed_bag = int(elapsed_wall * self.rate)
        target_bag_t = self.t0_bag + target_elapsed_bag

        sent = 0
        while (
            self.i < len(self.samples)
            and self.samples[self.i].t_ns <= target_bag_t
            and sent < self.max_points_per_tick
        ):
            q_cmd = self.samples[self.i].q
            self.publish_qpos(q_cmd, duration=self.point_duration)

            self.i += 1
            sent += 1

        # If we are far behind, we intentionally drip-feed points to avoid flooding the controller.
        if self.i < len(self.samples) and self.samples[self.i].t_ns <= target_bag_t:
            self.get_logger().warn(
                f"Behind playback: capped at {sent} points/tick. Consider reducing --rate or increasing controller capacity."
            )


def ask_approval() -> bool:
    ans = input("Replay this bag to joint_trajectory_controller? [y/N]: ").strip().lower()
    return ans in ("y", "yes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="rosbag2 folder path (mcap storage)")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--point-duration", type=float, default=0.08)
    parser.add_argument("--warmup-duration", type=float, default=10.0)
    args = parser.parse_args()

    # 1) bag 읽기 + 요약
    samples, summary = read_joint_states_from_bag(
        bag_path=args.bag,
        joint_names=JOINT_NAMES,
    )
    print_summary(summary)

    # 2) 사용자 승인
    if not ask_approval():
        print("Aborted by user.")
        return

    # 3) 승인되면 ROS node로 재생
    rclpy.init()
    node = BagJointTrajectoryPlayer(
        samples=samples,
        rate=args.rate,
        point_duration=args.point_duration,
        warmup_duration=args.warmup_duration,
    )

    try:
        # Spin until warmup is expected to finish
        warmup_end_ns = getattr(node, "_warmup_end_ns", node.get_clock().now().nanoseconds)
        while rclpy.ok() and node.get_clock().now().nanoseconds < warmup_end_ns:
            rclpy.spin_once(node, timeout_sec=0.1)

        # Second approval after warmup move
        ans2 = input("Warmup done. Start playback now? [y/N]: ").strip().lower()
        if ans2 not in ("y", "yes"):
            print("Playback aborted by user after warmup.")
            return

        node.arm_playback()
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()