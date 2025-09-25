#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

# === 사용 환경에 맞게 필요하면 수정 ===
CONTROLLER_CMD_TOPIC = "/position_controllers/commands"   # JointGroupPositionController 입력
JOINT_STATES_TOPIC   = "/rbpodo/joint_states"              # launch에서 remap한 joint_states
PUBLISH_RATE_HZ      = 50
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

class PositionControllerTester(Node):
    def __init__(self):
        super().__init__("position_controller_tester")

        self.cmd_pub = self.create_publisher(Float64MultiArray, CONTROLLER_CMD_TOPIC, 10)
        self.joint_sub = self.create_subscription(JointState, JOINT_STATES_TOPIC, self._joint_cb, 10)

        self._latest_positions = None  # type: List[float]
        self._joint_index_map = None   # name -> idx (from first received JointState)

        self.get_logger().info("Waiting for first JointState...")
        # 잠깐 기다리며 첫 joint_states 수신
        end = time.time() + 5.0
        while rclpy.ok() and time.time() < end and self._latest_positions is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._latest_positions is None:
            self.get_logger().warn("No joint_states received. "
                                   "실로봇/시뮬레이터가 joint_states를 내보내고 있는지 확인하세요.")
        else:
            self.get_logger().info(f"Got initial joint_states: {self._latest_positions}")

    def _joint_cb(self, msg: JointState):
        # 첫 수신에서 name->index 맵핑 생성
        if self._joint_index_map is None:
            self._joint_index_map = {name: i for i, name in enumerate(msg.name)}
            missing = [n for n in JOINT_NAMES if n not in self._joint_index_map]
            if missing:
                self.get_logger().warn(f"JointState에 없는 조인트가 있습니다: {missing}")

        # 컨트롤러 순서(JOINT_NAMES)에 맞춰 포지션 배열 재정렬
        positions = [0.0] * len(JOINT_NAMES)
        for i, name in enumerate(JOINT_NAMES):
            idx = self._joint_index_map.get(name, None)
            if idx is None or idx >= len(msg.position):
                # 데이터가 없으면 이전값 유지 or 0.0
                positions[i] = positions[i] if self._latest_positions else 0.0
            else:
                positions[i] = msg.position[idx]
        self._latest_positions = positions

    def send_positions(self, positions: List[float]):
        msg = Float64MultiArray()
        msg.data = positions
        self.cmd_pub.publish(msg)

    def nudge_small_motion(self, delta=0.05):
        """현재 자세에서 base, shoulder에 아주 작은 각도(라디안)로 살짝 이동"""
        if self._latest_positions is None:
            self.get_logger().warn("현재 자세를 몰라서 nudge 생략")
            return
        target = self._latest_positions.copy()
        if len(target) >= 2:
            target[0] += delta  # base
            target[1] += delta  # shoulder
        self.get_logger().info(f"Small nudge to: {['%.3f' % a for a in target]}")
        # 한 번만 보내도 되지만, 컨트롤이 안정적으로 잡히게 짧게 몇 번 보냄
        for _ in range(10):
            self.send_positions(target)
            rclpy.spin_once(self, timeout_sec=0.02)

    def move_step(self, joint_index: int = 0, delta: float = 0.05, hold_sec: float = 1.0):
        """지정한 조인트를 delta 라디안만큼 한 번 '툭' 움직였다가 원위치로 복귀합니다."""
        if self._latest_positions is None:
            self.get_logger().warn("현재 자세를 몰라서 move_step 생략")
            return

        if joint_index < 0 or joint_index >= len(JOINT_NAMES):
            self.get_logger().warn(f"joint_index {joint_index} 가 유효 범위를 벗어났습니다.")
            return

        start_pos = self._latest_positions.copy()
        target = start_pos.copy()
        target[joint_index] += delta

        rate = self.create_rate(PUBLISH_RATE_HZ)
        self.get_logger().info(f"Step joint[{joint_index}] by {delta:.3f} rad")

        # 목표 자세로 잠시 유지
        t_end = time.time() + hold_sec
        while rclpy.ok() and time.time() < t_end:
            self.send_positions(target)
            rclpy.spin_once(self, timeout_sec=0.0)
            rate.sleep()

        # 시작 자세로 복귀
        t_end = time.time() + hold_sec
        while rclpy.ok() and time.time() < t_end:
            self.send_positions(start_pos)
            rclpy.spin_once(self, timeout_sec=0.0)
            rate.sleep()

def main():
    rclpy.init()
    node = PositionControllerTester()

    try:
        # 간단한 단계 동작: base(0번 조인트) → shoulder(1번 조인트)
        node.move_step(joint_index=0, delta=0.05, hold_sec=1.0)
        node.move_step(joint_index=1, delta=0.05, hold_sec=1.0)

        node.get_logger().info("Done. You should have seen simple step motions via position controller.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()