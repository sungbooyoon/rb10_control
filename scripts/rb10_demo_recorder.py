import time
import rbpodo as rb
import sys, select, tty, termios
from datetime import datetime
import numpy as np
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ros2topic.api import get_topic_names_and_types

IP = "10.0.2.7"
HZ = 30.0
FIRST_TIMEOUT = 1.0
LOOP_TIMEOUT = 0.05

SAVE_MP4 = False  # Set True to save MP4 videos, False to skip

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
            self.create_subscription(Image, t, lambda msg, t=t: self.image_callback(msg, t), 10)

    def image_callback(self, msg, topic):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_frames[topic] = frame
        except Exception as e:
            self.get_logger().warn(f"[{topic}] Image conversion failed: {e}")

def detect_available_cameras():
    try:
        rclpy.init(args=None)
        tmp_node = Node("camera_topic_scanner")
        topics = [t[0] for t in get_topic_names_and_types(tmp_node, no_demangle=True) if 'image_raw' in t[0]]
        tmp_node.destroy_node()
        rclpy.shutdown()
        return topics
    except Exception as e:
        print(f"[WARN] Failed to detect camera topics: {e}")
        return []

def main():
    global SAVE_MP4

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

    print("Press ENTER to start recording; press ESC to stop and save (recording ALL packets).")

    available_cameras = detect_available_cameras()
    if available_cameras:
        rclpy.init(args=None)
        cam_node = MultiCameraRecorder(available_cameras)
        print(f"[INFO] Found {len(available_cameras)} camera(s): {available_cameras}")
        user_input = input("Save camera videos as MP4? (y/N): ").strip().lower()
        SAVE_MP4 = user_input == 'y'
        print(f"[INFO] MP4 recording {'enabled' if SAVE_MP4 else 'disabled'}.")
    else:
        cam_node = None
        print("[INFO] No camera topics found. Skipping camera recording.")

    if cam_node:
        rate = cam_node.create_rate(HZ)
    else:
        rclpy.init(args=None)
        tmp_node = Node("rate_controller")
        rate = tmp_node.create_rate(HZ)

    with KeyListener() as kl:
        # wait for ENTER to start
        while True:
            key = kl.get_key()
            if key == '\r' or key == '\n':
                break
            time.sleep(0.01)

        print("[INFO] Recording started. Logging ALL packets (with freedrive flag).")

        stamps, frees, jnt_angs, jnt_vels, tcp_poss, tcp_vels = [], [], [], [], [], []
        frames_dict = {t: [] for t in cam_node.latest_frames.keys()} if cam_node else {}

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
                print("[WARN] no packet this cycle.")
            else:
                s = data.sdata

                stamp = (cam_node.get_clock().now().nanoseconds * 1e-9) if cam_node else (tmp_node.get_clock().now().nanoseconds * 1e-9)
                now_pc = time.perf_counter()
                dt = now_pc - prev_pc if prev_pc is not None else None

                jnt_ang = as_list(s.jnt_ang, 6)
                tcp_pos = as_list(s.tcp_pos, 6)
                jnt_vel = diff_arr(prev_jnt_ang, jnt_ang, dt)
                tcp_vel = diff_arr(prev_tcp_pos, tcp_pos, dt)
                freedrive = int(getattr(s, "is_freedrive_mode", 0))

                # unit conversions
                jnt_ang_rad = np.deg2rad(jnt_ang).astype(np.float64)
                jnt_vel_rad = np.deg2rad(jnt_vel).astype(np.float64) if jnt_vel is not None else np.full(6, np.nan, dtype=np.float64)
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

                if cam_node:
                    for topic, frame in cam_node.latest_frames.items():
                        if frame is not None:
                            frames_dict[topic].append(frame.copy())

                # update prevs
                prev_pc = now_pc
                prev_jnt_ang = jnt_ang
                prev_tcp_pos = tcp_pos

            rate.sleep()

    os.makedirs("../dataset", exist_ok=True)
    filename = f"../dataset/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez_compressed(
        filename,
        stamp=np.asarray(stamps, dtype=np.float64),
        freedrive=np.asarray(frees, dtype=np.uint8),
        jnt_ang=np.vstack(jnt_angs),
        jnt_vel=np.vstack(jnt_vels),
        tcp_pos=np.vstack(tcp_poss),
        tcp_vel=np.vstack(tcp_vels),
    )
    print(f"[OK] Saved {len(stamps)} samples to {filename}")

    if cam_node and SAVE_MP4:
        for topic, frames in frames_dict.items():
            if len(frames) == 0:
                continue
            safe_topic = topic.replace('/', '_').strip('_')
            vid_filename = filename.replace('.npz', f'_{safe_topic}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w, _ = frames[0].shape
            out = cv2.VideoWriter(vid_filename, fourcc, HZ, (w, h))
            for f in frames:
                out.write(f)
            out.release()
            print(f"[OK] Saved {len(frames)} frames from {topic} to {vid_filename}")

    if cam_node:
        cam_node.destroy_node()
        rclpy.shutdown()
    else:
        tmp_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
