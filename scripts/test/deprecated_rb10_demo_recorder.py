#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time, os, sys, select, tty, termios, threading
from datetime import datetime
import numpy as np
import cv2
import threading

import rbpodo as rb

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import SingleThreadedExecutor
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
            self._next = time.perf_counter()  # 드리프트 보정

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
        # latest_frames: topic -> {"frame": np.ndarray, "stamp": float}
        self.latest_frames = {t: None for t in topics}
        # mp4 writers (topic -> cv2.VideoWriter)
        self.writers = {}
        self.writer_fps = {}
        self.last_written_stamp = {t: None for t in topics}
        for t in topics:
            self.create_subscription(
                Image, t,
                lambda msg, t=t: self.image_callback(msg, t),
                IMAGE_QOS
            )

    def image_callback(self, msg, topic):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.latest_frames[topic] = {"frame": frame, "stamp": stamp}
        except Exception as e:
            self.get_logger().warn(f"[{topic}] Image conversion failed: {e}")

    def get_or_create_writer(self, topic, path_mp4, fps):
        """Lazy-create writer when first frame arrives (we need width/height)."""
        if topic in self.writers:
            return self.writers[topic]
        item = self.latest_frames.get(topic)
        if not item:
            return None
        frame = item["frame"]
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path_mp4, fourcc, float(fps), (w, h))
        if not writer.isOpened():
            self.get_logger().error(f"[{topic}] Failed to open VideoWriter: {path_mp4}")
            return None
        self.writers[topic] = writer
        self.writer_fps[topic] = float(fps)
        self.get_logger().info(f"[{topic}] VideoWriter opened {w}x{h}@{fps}: {path_mp4}")
        return writer

    def write_frame_if_new(self, topic, path_mp4, fps):
        """Write to mp4 only if a new frame (by ROS stamp) arrived."""
        item = self.latest_frames.get(topic)
        if not item:
            return
        stamp = item["stamp"]
        if self.last_written_stamp[topic] is not None and stamp <= self.last_written_stamp[topic]:
            return  # same or older frame; skip
        writer = self.get_or_create_writer(topic, path_mp4, fps)
        if writer is None:
            return
        writer.write(item["frame"])
        self.last_written_stamp[topic] = stamp

    def close_writers(self):
        for t, w in self.writers.items():
            try:
                w.release()
            except Exception:
                pass
        self.writers.clear()
        self.writer_fps.clear()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-camera", action="store_true", help="Enable camera logging")
    parser.add_argument("--camera-topic", action="append", default=[],
                        help="Image topic to subscribe (repeatable). e.g., --camera-topic /camera/color/image_raw")
    parser.add_argument("--save-mp4", action="store_true", help="Stream-save MP4 files for each topic")
    parser.add_argument("--npz-compress", action="store_true", help="Compress main NPZ (slower, smaller)")
    parser.add_argument("--save-frames-npz", action="store_true", help="Also save frames as NPZ (memory heavy)")
    parser.add_argument("--video-fps", type=float, default=30.0, help="MP4 FPS (default 30.0)")
    args = parser.parse_args()

    # ---- camera auto-enable if topics are given ----
    if len(args.camera_topic) > 0:
        args.enable_camera = True

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

    # 카메라 노드 & executor 스레드
    cam_node = None
    executor = None
    spin_thread = None
    stop_event = threading.Event()

    if args.enable_camera and len(args.camera_topic) > 0:
        rclpy.init(args=None)
        cam_node = MultiCameraRecorder(args.camera_topic)
        from rclpy.executors import SingleThreadedExecutor
        executor = SingleThreadedExecutor()
        executor.add_node(cam_node)

        def spin_loop():
            while not stop_event.is_set():
                try:
                    executor.spin_once(timeout_sec=0.05)
                except Exception:
                    break

        spin_thread = threading.Thread(target=spin_loop, daemon=True)
        spin_thread.start()
        print(f"[INFO] Camera topics: {args.camera_topic}")
    else:
        print("[INFO] Camera disabled or no topics provided. Skipping camera recording.")

    # 저장 준비
    out_dir = "/home/sungboo/ros2_ws/src/rb10_control/dataset"
    os.makedirs(out_dir, exist_ok=True)
    base_name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_npz = os.path.join(out_dir, f"{base_name}.npz")

    # mp4 경로 사전
    topic2mp4 = {}
    if cam_node and args.save_mp4:
        for t in args.camera_topic:
            safe_topic = t.replace('/', '_').strip('_')
            topic2mp4[t] = os.path.join(out_dir, f"{base_name}_{safe_topic}.mp4")
            # writer는 첫 프레임 들어올 때 생성 (lazy)

    # 프레임 누적(옵션)
    frames_dict = {t: [] for t in (args.camera_topic if (cam_node and args.save_frames_npz) else [])}

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

        prev_pc = time.perf_counter()
        prev_jnt_ang = as_list(data.sdata.jnt_ang, 6)
        prev_tcp_pos = as_list(data.sdata.tcp_pos, 6)

        while True:
            key = kl.get_key()
            if key == '\x1b':  # ESC
                print("[INFO] ESC pressed. Stopping recording.")
                break

            data = read_once(ch, LOOP_TIMEOUT)

            # cam_node는 executor 스레드가 계속 돌려줌 (spin_once 불필요)

            if data is None or getattr(data, "sdata", None) is None:
                pass
            else:
                s = data.sdata

                stamp = time.time()
                now_pc = time.perf_counter()
                dt = now_pc - prev_pc if prev_pc is not None else None

                jnt_ang = as_list(s.jnt_ang, 6)
                tcp_pos = as_list(s.tcp_pos, 6)
                jnt_vel = diff_arr(prev_jnt_ang, jnt_ang, dt)
                tcp_vel = diff_arr(prev_tcp_pos, tcp_pos, dt)
                freedrive = int(getattr(s, "is_freedrive_mode", 0))

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

                eft = np.array([
                    float(getattr(s, "eft_fx", np.nan)),
                    float(getattr(s, "eft_fy", np.nan)),
                    float(getattr(s, "eft_fz", np.nan)),
                    float(getattr(s, "eft_mx", np.nan)),
                    float(getattr(s, "eft_my", np.nan)),
                    float(getattr(s, "eft_mz", np.nan)),
                ], dtype=np.float64)
                efts.append(eft)

                # 카메라 처리
                if cam_node:
                    for topic in args.camera_topic:
                        if args.save_mp4:
                            # 최초 1프레임은 stamp 비교 없이도 write 보장
                            item = cam_node.latest_frames.get(topic)
                            if item:
                                if cam_node.last_written_stamp[topic] is None:
                                    w = cam_node.get_or_create_writer(topic, topic2mp4[topic], fps=args.video_fps)
                                    if w is not None:
                                        w.write(item["frame"])
                                        cam_node.last_written_stamp[topic] = item["stamp"]
                                else:
                                    cam_node.write_frame_if_new(topic, topic2mp4[topic], fps=args.video_fps)

                        if args.save_frames_npz:
                            item = cam_node.latest_frames.get(topic)
                            if item and (cam_node.last_written_stamp[topic] is None or item["stamp"] > cam_node.last_written_stamp[topic]):
                                frames_dict[topic].append(item["frame"].copy())


                prev_pc = now_pc
                prev_jnt_ang = jnt_ang
                prev_tcp_pos = tcp_pos

            rate.sleep()

    # 저장
    if len(stamps) == 0:
        print("[WARN] No samples captured; nothing to save.")
    else:
        saver = (np.savez_compressed if args.npz_compress else np.savez)
        saver(
            base_npz,
            stamp=np.asarray(stamps, dtype=np.float64),
            freedrive=np.asarray(frees, dtype=np.uint8),
            jnt_ang=np.vstack(jnt_angs),
            jnt_vel=np.vstack(jnt_vels),
            tcp_pos=np.vstack(tcp_poss),
            tcp_vel=np.vstack(tcp_vels),
            eft=np.vstack(efts),
        )
        print(f"[OK] Saved {len(stamps)} samples to {base_npz} "
              f"({'compressed' if args.npz_compress else 'uncompressed'})")

        if cam_node and args.save_frames_npz:
            for topic, frames in frames_dict.items():
                if len(frames) == 0:
                    continue
                safe_topic = topic.replace('/', '_').strip('_')
                frames_npz = os.path.join(os.path.dirname(base_npz), f"{base_name}_{safe_topic}.npz")
                np.savez(frames_npz, frames=np.stack(frames))
                print(f"[OK] Saved {len(frames)} frames from {topic} to {frames_npz}")

    # MP4 writer 닫기 전에 스핀 멈추고 join
    if cam_node:
        try:
            stop_event.set()
            if spin_thread:
                spin_thread.join(timeout=1.0)
        except Exception:
            pass

        # 이제 파일 핸들 안전하게 닫기
        if args.save_mp4:
            cam_node.close_writers()
            for topic, path_mp4 in topic2mp4.items():
                print(f"[OK] MP4 written: {topic} -> {path_mp4}")

        # executor/노드/ROS 종료
        try:
            if executor and cam_node:
                executor.remove_node(cam_node)
                executor.shutdown()
        except Exception:
            pass
        try:
            cam_node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
