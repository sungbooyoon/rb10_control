#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time, os, sys, select, tty, termios
from datetime import datetime
import numpy as np
import cv2

import rbpodo as rb

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ===== User params =====
IP = "10.0.2.7"
HZ = 30.0
FIRST_TIMEOUT = 1.0
LOOP_TIMEOUT = 0.05

# ===== QoS: RealSense 호환 (BestEffort / volatile / depth 5) =====
IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)

class WallRate:
    def __init__(self, hz: float):
        self._period = 1.0 / float(hz)
        self._next = time.perf_counter()
    def sleep(self):
        self._next += self._period
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            # 심하게 지연됐으면 드리프트 보정
            self._next = time.perf_counter()

def safe_rclpy_init():
    try:
        rclpy.init(args=None)
    except RuntimeError:
        # 이미 초기화된 경우
        pass

def read_once(ch, timeout):
    try:
        return ch.request_data(timeout)
    except Exception as e:
        print(f"[WARN] request_data error: {e}")
        return None

def as_list(a, n): return [float(a[i]) for i in range(n)]
def diff_arr(prev, curr, dt):
    if prev is None or curr is None or dt is None or dt <= 0: return None
    return [(curr[i] - prev[i]) / dt for i in range(len(curr))]

class KeyListener:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self
    def __exit__(self, type, value, traceback):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
    def get_key(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

class MultiCameraRecorder(Node):
    def __init__(self, topics):
        super().__init__('multi_camera_logger')
        self.bridge = CvBridge()
        self.latest_frames = {}
        for t in topics:
            self.latest_frames[t] = None
            # RealSense 기본 QoS와 호환되도록 BestEffort
            self.create_subscription(
                Image, t,
                lambda msg, t=t: self.image_callback(msg, t),
                IMAGE_QOS
            )

    def image_callback(self, msg, topic):
        try:
            # OpenCV는 BGR 기대 → 바로 bgr8로 변환
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_frames[topic] = frame
        except Exception as e:
            self.get_logger().warn(f"[{topic}] Image conversion failed: {e}")

def main():
    # ---------- CLI args ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-camera", action="store_true", help="Enable camera logging")
    parser.add_argument("--camera-topic", action="append", default=[],
                        help="Image topic to subscribe (repeatable). e.g., --camera-topic /camera/camera/color/image_raw")
    parser.add_argument("--save-mp4", action="store_true", help="Save MP4 files for each topic")
    args = parser.parse_args()

    ch = rb.CobotData(IP)

    # 첫 패킷 대기
    deadline = time.time() + 5.0
    data = None
    while time.time() < deadline:
        data = read_once(ch, FIRST_TIMEOUT)
        if data is not None and getattr(data, "sdata", None) is not None:
            break
        print("[INFO] waiting first packet...")
    if data is None or getattr(data, "sdata", None) is None:
        print("[ERROR] no data received (sdata is None).")
        return
    print("[OK] first packet received.")

    # 카메라 노드 (옵션)
    cam_node = None
    if args.enable_camera and len(args.camera_topic) > 0:
        safe_rclpy_init()
        cam_node = MultiCameraRecorder(args.camera_topic)
        print(f"[INFO] Camera topics: {args.camera_topic}")
    else:
        print("[INFO] Camera disabled or no topics provided. Skipping camera recording.")

    rate = WallRate(HZ)

    print("Press ENTER to start recording; press ESC to stop and save (recording ALL packets).")
    with KeyListener() as kl:
        while True:
            key = kl.get_key()
            if key == '\r' or key == '\n':
                break
            time.sleep(0.01)

        print("[INFO] Recording started. Logging ALL packets (with freedrive flag).")

        stamps, frees, jnt_angs, jnt_vels, tcp_poss, tcp_vels = [], [], [], [], [], []
        efts = []
        frames_dict = {t: [] for t in (cam_node.latest_frames.keys() if cam_node else [])}

        prev_pc = time.perf_counter()
        prev_jnt_ang = as_list(data.sdata.jnt_ang, 6)
        prev_tcp_pos = as_list(data.sdata.tcp_pos, 6)

        while True:
            key = kl.get_key()
            if key == '\x1b':  # ESC
                print("[INFO] ESC pressed. Stopping recording.")
                break

            data = read_once(ch, LOOP_TIMEOUT)

            if cam_node:
                rclpy.spin_once(cam_node, timeout_sec=0.0)

            if data is None or getattr(data, "sdata", None) is None:
                # print("[WARN] no packet this cycle.")
                pass
            else:
                s = data.sdata

                # 공통 시간축: 벽시계(시스템 시간). RealSense도 global_time_enabled로 맞추면 거의 동일축.
                stamp = time.time()
                now_pc = time.perf_counter()
                dt = now_pc - prev_pc if prev_pc is not None else None
                
                jnt_ang = as_list(s.jnt_ang, 6)
                tcp_pos = as_list(s.tcp_pos, 6)
                jnt_vel = diff_arr(prev_jnt_ang, jnt_ang, dt)
                tcp_vel = diff_arr(prev_tcp_pos, tcp_pos, dt)
                freedrive = int(getattr(s, "is_freedrive_mode", 0))

                # 단위 변환
                jnt_ang_rad = np.deg2rad(jnt_ang).astype(np.float64)
                jnt_vel_rad = (np.deg2rad(jnt_vel).astype(np.float64)
                               if jnt_vel is not None else np.full(6, np.float64(np.nan)))
                tcp_pos_conv = np.empty(6, dtype=np.float64)
                tcp_pos_conv[:3] = np.array(tcp_pos[:3], dtype=np.float64) / 1000.0
                tcp_pos_conv[3:] = np.deg2rad(tcp_pos[3:6]).astype(np.float64)
                tcp_vel_conv = np.full(6, np.float64(np.nan))
                if tcp_vel is not None:
                    tcp_vel_conv[:3] = np.array(tcp_vel[:3], dtype=np.float64) / 1000.0
                    tcp_vel_conv[3:] = np.deg2rad(tcp_vel[3:6]).astype(np.float64)

                stamps.append(stamp)
                frees.append(freedrive)
                jnt_angs.append(jnt_ang_rad)
                jnt_vels.append(jnt_vel_rad)
                tcp_poss.append(tcp_pos_conv)
                tcp_vels.append(tcp_vel_conv)

                # 외력/토크 (없으면 NaN)
                eft_fx = getattr(s, "eft_fx", None)
                eft_fy = getattr(s, "eft_fy", None)
                eft_fz = getattr(s, "eft_fz", None)
                eft_mx = getattr(s, "eft_mx", None)
                eft_my = getattr(s, "eft_my", None)
                eft_mz = getattr(s, "eft_mz", None)
                efts.append(np.array([
                    np.nan if eft_fx is None else float(eft_fx),
                    np.nan if eft_fy is None else float(eft_fy),
                    np.nan if eft_fz is None else float(eft_fz),
                    np.nan if eft_mx is None else float(eft_mx),
                    np.nan if eft_my is None else float(eft_my),
                    np.nan if eft_mz is None else float(eft_mz),
                ], dtype=np.float64))

                if cam_node:
                    for topic, frame in cam_node.latest_frames.items():
                        if frame is not None:
                            frames_dict[topic].append(frame.copy())

                prev_pc = now_pc
                prev_jnt_ang = jnt_ang
                prev_tcp_pos = tcp_pos

            rate.sleep()

    # 저장
    if len(stamps) == 0:
        print("[WARN] No samples captured; nothing to save.")
    else:
        os.makedirs("/home/sungboo/ros2_ws/src/rb10_control/dataset", exist_ok=True)
        filename = f"/home/sungboo/ros2_ws/src/rb10_control/dataset/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(
            filename,
            stamp=np.asarray(stamps, dtype=np.float64),
            freedrive=np.asarray(frees, dtype=np.uint8),
            jnt_ang=np.vstack(jnt_angs),
            jnt_vel=np.vstack(jnt_vels),
            tcp_pos=np.vstack(tcp_poss),
            tcp_vel=np.vstack(tcp_vels),
            eft=np.vstack(efts),
        )
        print(f"[OK] Saved {len(stamps)} samples to {filename}")

        if cam_node:
            for topic, frames in frames_dict.items():
                if len(frames) == 0:
                    continue
                safe_topic = topic.replace('/', '_').strip('_')
                vid_filename = filename.replace('.npz', f'_{safe_topic}.npz')
                np.savez_compressed(vid_filename, frames=np.stack(frames))
                print(f"[OK] Saved {len(frames)} frames from {topic} to {vid_filename}")
            if args.save_mp4:
                for topic, frames in frames_dict.items():
                    if len(frames) == 0:
                        continue
                    safe_topic = topic.replace('/', '_').strip('_')
                    vid_filename = filename.replace('.npz', f'_{safe_topic}.mp4')
                    np.savez_compressed(vid_filename, frames=np.stack(frames))
                    print(f"[OK] Saved {len(frames)} frames from {topic} to {vid_filename}")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    h, w, _ = frames[0].shape
                    out = cv2.VideoWriter(vid_filename, fourcc, HZ, (w, h))
                    for f in frames:
                        out.write(f)  # 이미 BGR 프레임
                    out.release()
                    print(f"[OK] Saved {len(frames)} frames from {topic} to {vid_filename}")

    # 종료 정리
    if cam_node:
        cam_node.destroy_node()
        try:
            rclpy.shutdown()
        except RuntimeError:
            pass

if __name__ == "__main__":
    main()
