#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rbpodo -> ROS2 bridge (for rosbag)
- Reads rb.CobotData(IP)
- Publishes:
  /rb/joint_states      sensor_msgs/JointState        (rad)
  /rb/tcp_pose          geometry_msgs/PoseStamped     (m + quaternion xyzw)
  /rb/freedrive         std_msgs/Bool
  /rb/ee_wrench         geometry_msgs/WrenchStamped   (if available)
"""

import time
import sys
import math
import argparse

import numpy as np
import rbpodo as rb

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Header, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped

from tf_transformations import quaternion_from_euler

# ===== QoS (bagging/로컬 기록 용) =====
DEFAULT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

def deg2rad_list(a, n):
    return np.deg2rad([float(a[i]) for i in range(n)], dtype=np.float64).tolist()

def tcp_to_pose_msg(node: Node, tcp_pos_deg: list, base_frame: str, ee_frame: str) -> PoseStamped:
    """
    tcp_pos_deg: [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] (ZYX 회전 정의)
    PoseStamped: position[m], orientation(quat xyzw)
    """
    x_m = float(tcp_pos_deg[0]) / 1000.0
    y_m = float(tcp_pos_deg[1]) / 1000.0
    z_m = float(tcp_pos_deg[2]) / 1000.0
    rx = math.radians(float(tcp_pos_deg[3]))
    ry = math.radians(float(tcp_pos_deg[4]))
    rz = math.radians(float(tcp_pos_deg[5]))
    # R = Rz(rz) * Ry(ry) * Rx(rx)  (ZYX)
    qx, qy, qz, qw = quaternion_from_euler(rz, ry, rx, axes='rzyx')

    msg = PoseStamped()
    msg.header = Header()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = base_frame
    msg.pose.position.x = x_m
    msg.pose.position.y = y_m
    msg.pose.position.z = z_m
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg

class RbBridge(Node):
    def __init__(self, ip: str, hz: float, base_frame: str, ee_frame: str, joint_names):
        super().__init__("rbpodo_bridge")
        self.ip = ip
        self.hz = float(hz)
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.joint_names = joint_names

        # rb channel
        self.ch = rb.CobotData(self.ip)
        self.first_timeout = 1.0
        self.loop_timeout = max(0.001, 1.0 / self.hz * 0.5)

        # pubs
        self.pub_js = self.create_publisher(JointState, "/rb/joint_states", DEFAULT_QOS)
        self.pub_pose = self.create_publisher(PoseStamped, "/rb/tcp_pose", DEFAULT_QOS)
        self.pub_freedrive = self.create_publisher(Bool, "/rb/freedrive", DEFAULT_QOS)
        self.pub_wrench = self.create_publisher(WrenchStamped, "/rb/ee_wrench", DEFAULT_QOS)

        # timer
        self.timer = self.create_timer(1.0 / self.hz, self._on_timer)

        self.get_logger().info(f"Connecting to rbpodo at {self.ip} ...")
        # Warm-up: try one packet
        deadline = time.time() + 5.0
        data = None
        while time.time() < deadline and rclpy.ok():
            try:
                data = self.ch.request_data(self.first_timeout)
            except Exception as e:
                self.get_logger().warn(f"request_data error: {e}")
                data = None
            if data is not None and getattr(data, "sdata", None) is not None:
                break
            time.sleep(0.05)
        if data is None or getattr(data, "sdata", None) is None:
            self.get_logger().error("No first packet. Will continue trying in timer.")
        else:
            self.get_logger().info("First packet received.")

    def _on_timer(self):
        # pull one packet
        try:
            data = self.ch.request_data(self.loop_timeout)
        except Exception as e:
            self.get_logger().warn(f"request_data error: {e}")
            return

        if data is None or getattr(data, "sdata", None) is None:
            return
        s = data.sdata

        now = self.get_clock().now().to_msg()

        # --- JointState ---
        try:
            js = JointState()
            js.header = Header(stamp=now, frame_id=self.base_frame)
            js.name = list(self.joint_names)
            # rbpodo는 deg 기준 → rad로 변환
            q_rad = deg2rad_list(s.jnt_ang, 6)
            js.position = [float(x) for x in q_rad]
            # velocity/effort는 미기록
            self.pub_js.publish(js)
        except Exception as e:
            self.get_logger().warn(f"JointState publish failed: {e}")

        # --- TCP Pose ---
        try:
            pose_msg = tcp_to_pose_msg(self, [s.tcp_pos[i] for i in range(6)], self.base_frame, self.ee_frame)
            self.pub_pose.publish(pose_msg)
        except Exception as e:
            self.get_logger().warn(f"TCP pose publish failed: {e}")

        # --- Freedrive ---
        try:
            freedrive = int(getattr(s, "is_freedrive_mode", 0))
            self.pub_freedrive.publish(Bool(data=bool(freedrive)))
        except Exception as e:
            self.get_logger().warn(f"Freedrive publish failed: {e}")

        # --- Wrench (있을 때만) ---
        try:
            # 속성이 없으면 None
            fx = getattr(s, "eft_fx", None)
            fy = getattr(s, "eft_fy", None)
            fz = getattr(s, "eft_fz", None)
            mx = getattr(s, "eft_mx", None)
            my = getattr(s, "eft_my", None)
            mz = getattr(s, "eft_mz", None)
            have_eft = all(v is not None for v in (fx, fy, fz, mx, my, mz))
            if have_eft:
                w = WrenchStamped()
                w.header = Header(stamp=now, frame_id=self.ee_frame)
                w.wrench.force.x = float(fx)
                w.wrench.force.y = float(fy)
                w.wrench.force.z = float(fz)
                w.wrench.torque.x = float(mx)
                w.wrench.torque.y = float(my)
                w.wrench.torque.z = float(mz)
                self.pub_wrench.publish(w)
        except Exception as e:
            self.get_logger().warn(f"Wrench publish failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="rbpodo -> ROS2 publishers (for rosbag)")
    parser.add_argument("--ip", type=str, default="10.0.2.7")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--base-frame", type=str, default="link0")
    parser.add_argument("--ee-frame", type=str, default="tcp")
    parser.add_argument("--joint-names", type=str,
                        default="base,shoulder,elbow,wrist1,wrist2,wrist3",
                        help="comma-separated 6 names")
    args = parser.parse_args()

    rclpy.init(args=None)
    node = RbBridge(
        ip=args.ip,
        hz=args.hz,
        base_frame=args.base_frame,
        ee_frame=args.ee_frame,
        joint_names=[n.strip() for n in args.joint_names.split(",")]
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
