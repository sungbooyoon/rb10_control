#!/usr/bin/env python3
"""
sudo systemctl start spacenavd     # 이미 enable했다면 생략 가능
ros2 run spacenav spacenav_node

ros2 launch rbpodo_moveit_config rb10_servo.launch.py

SpaceNavigator -> MoveIt 2 Servo teleop (ROS 2 Humble)
- ros2 run pymoveit2 ex_servo_spacenav.py --ros-args -r /delta_twist_cmds:=/servo_node/delta_twist_cmds
"""
from typing import Tuple
from math import copysign

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from sensor_msgs.msg import Joy
from pymoveit2 import MoveIt2Servo
from pymoveit2.robots import rb10 as robot


def deadband(v: float, db: float) -> float:
    return 0.0 if abs(v) < db else v

class SpacenavServo(Node):
    def __init__(self):
        super().__init__("ex_servo_spacenav")
        self.cb = ReentrantCallbackGroup()
        self.servo = MoveIt2Servo(
            node=self,
            frame_id=robot.base_link_name(), 
            callback_group=self.cb,
        )

        # 매핑/스케일 파라미터 (원하는 감도에 맞게 수정)
        self.db_lin = 0.05    # translation deadband
        self.db_ang = 0.05    # rotation deadband
        self.scale_lin = 0.15 # m/s 정도 느낌
        self.scale_ang = 0.8  # rad/s 정도 느낌

        # spacenav_node는 Joy.axes에 [tx, ty, tz, rx, ry, rz]가 들어오는 게 일반적
        # (장치/드라이버에 따라 부호가 다를 수 있으니 아래 sign만 바꿔주면 됨)
        self.axes_idx = {
            "tx": 0, "ty": 1, "tz": 2,
            "rx": 3, "ry": 4, "rz": 5,
        }
        self.sign = {  # 축 방향 뒤집고 싶으면 -1로 바꾸기
            "tx": +1, "ty": +1, "tz": +1,
            "rx": +1, "ry": +1, "rz": +1,
        }

        self.create_subscription(Joy, "/spacenav/joy", self.joy_cb, 10, callback_group=self.cb)

    def joy_cb(self, msg: Joy):
        ax = msg.axes
        try:
            # Joy -> cartesian twist
            vx = self.scale_lin * deadband(self.sign["tx"] * ax[self.axes_idx["tx"]], self.db_lin)
            vy = self.scale_lin * deadband(self.sign["ty"] * ax[self.axes_idx["ty"]], self.db_lin)
            vz = self.scale_lin * deadband(self.sign["tz"] * ax[self.axes_idx["tz"]], self.db_lin)
            wx = self.scale_ang * deadband(self.sign["rx"] * ax[self.axes_idx["rx"]], self.db_ang)
            wy = self.scale_ang * deadband(self.sign["ry"] * ax[self.axes_idx["ry"]], self.db_ang)
            wz = self.scale_ang * deadband(self.sign["rz"] * ax[self.axes_idx["rz"]], self.db_ang)

            # MoveIt Servo 호출 (geometry_msgs/TwistStamped 내부로 전달됨)
            self.servo(linear=(vx, vy, vz), angular=(wx, wy, wz))

        except (IndexError, KeyError):
            # 축 인덱스가 다르면 여기서 조정
            self.get_logger().warn("Spacenav axis mapping mismatch. Adjust axes_idx/sign.")

def main():
    rclpy.init()
    node = SpacenavServo()
    try:
        rclpy.spin(node)
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
