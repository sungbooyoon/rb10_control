#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export color/depth frames + camera_info json from a ROS2 rosbag2 (MCAP).

Input:
  - rosbag2 directory (e.g., /path/to/res_YYYYmmdd_HHMMSS/)
    which contains:
      metadata.yaml
      *.mcap

Reads:
  - /camera/camera/color/image_rect_raw
  - /camera/camera/depth/image_rect_raw
  - /camera/camera/color/camera_info

Outputs:
  <out_root>/<bag_dir_name>/
    camera_info.json
    color/  000000_1700000000.123456789.png ...
    depth/  000000_1700000000.123456789.png ...  (16-bit PNG if depth is 16UC1 / 32FC1)

Sampling:
  - target Hz (default 30)
  - uses color stream as reference; saves a pair when:
      * enough time passed since last save (>= 1/hz)
      * a new color and a new depth frame have both been received since last save
      * assumes perfect sync; no tolerance check
"""

import os
import json
import argparse
from typing import Optional, Tuple

import numpy as np
import cv2

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TOPIC_COLOR = "/camera/camera/color/image_rect_raw"
TOPIC_DEPTH = "/camera/camera/depth/image_rect_raw"
TOPIC_INFO  = "/camera/camera/color/camera_info"


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def image_msg_to_bgr8(msg):
    """Convert sensor_msgs/Image to BGR uint8 OpenCV image."""
    # Prefer cv_bridge if available
    try:
        from cv_bridge import CvBridge
        bridge = CvBridge()
        return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception:
        pass

    h = int(msg.height)
    w = int(msg.width)
    enc = str(msg.encoding).lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        img = data.reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "bgr8":
        return data.reshape(h, w, 3)
    if enc in ("mono8", "8uc1"):
        img = data.reshape(h, w)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported color encoding: {msg.encoding}")


def depth_msg_to_png16(msg) -> np.ndarray:
    """
    Convert depth Image msg to a PNG-writable array.
    - If 16UC1: returned as uint16 (assumed mm)
    - If 32FC1: meters float -> mm uint16 (clamped)
    """
    h = int(msg.height)
    w = int(msg.width)
    enc = str(msg.encoding).lower()

    if enc == "16uc1":
        d = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
        return d

    if enc == "32fc1":
        d = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
        mm = np.clip(d * 1000.0, 0.0, 65535.0)
        return mm.astype(np.uint16)

    raise ValueError(f"Unsupported depth encoding: {msg.encoding}")


def camera_info_to_dict(msg) -> dict:
    return {
        "header": {
            "frame_id": msg.header.frame_id,
            "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
        },
        "height": int(msg.height),
        "width": int(msg.width),
        "distortion_model": str(msg.distortion_model),
        "d": [float(x) for x in msg.d],
        "k": [float(x) for x in msg.k],
        "r": [float(x) for x in msg.r],
        "p": [float(x) for x in msg.p],
        "binning_x": int(msg.binning_x),
        "binning_y": int(msg.binning_y),
        "roi": {
            "x_offset": int(msg.roi.x_offset),
            "y_offset": int(msg.roi.y_offset),
            "height": int(msg.roi.height),
            "width": int(msg.roi.width),
            "do_rectify": bool(msg.roi.do_rectify),
        },
    }


def open_reader(bag_dir: str) -> SequentialReader:
    """
    Open rosbag2 directory recorded with MCAP.
    bag_dir should contain metadata.yaml and *.mcap.
    """
    if not os.path.isdir(bag_dir):
        raise FileNotFoundError(f"--bag must be an existing rosbag2 directory: {bag_dir}")

    metadata = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(metadata):
        raise FileNotFoundError(f"metadata.yaml not found in bag dir: {bag_dir}")

    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_dir, storage_id="mcap")
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    return reader


def get_topic_types(reader: SequentialReader) -> dict:
    tt = {}
    for t in reader.get_all_topics_and_types():
        tt[t.name] = t.type
    return tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="path to rosbag2 directory (MCAP), e.g., .../res_YYYYmmdd_HHMMSS/")
    ap.add_argument("--hz", type=float, default=30.0, help="target sampling Hz (default 30)")
    args = ap.parse_args()

    bag_dir = os.path.abspath(args.bag)
    
    hz = float(args.hz)
    if hz <= 0:
        raise ValueError("--hz must be > 0")
    period_ns = int(round(1e9 / hz))

    out_color = os.path.join(bag_dir, "color")
    out_depth = os.path.join(bag_dir, "depth")
    ensure_dir(out_color)
    ensure_dir(out_depth)

    reader = open_reader(bag_dir)
    topic_types = get_topic_types(reader)

    for need in (TOPIC_COLOR, TOPIC_DEPTH, TOPIC_INFO):
        if need not in topic_types:
            raise RuntimeError(
                f"Missing topic in bag: {need}\nAvailable topics:\n- " + "\n- ".join(sorted(topic_types.keys()))
            )

    ColorT = get_message(topic_types[TOPIC_COLOR])
    DepthT = get_message(topic_types[TOPIC_DEPTH])
    InfoT  = get_message(topic_types[TOPIC_INFO])

    latest_depth: Optional[Tuple[int, object]] = None  # (t_ns, msg)
    latest_color: Optional[Tuple[int, object]] = None  # (t_ns, msg)
    new_depth = False
    new_color = False
    camera_info_saved = False

    last_save_t_ns: Optional[int] = None
    saved = 0
    seen_color = 0
    seen_depth = 0

    print(f"[info] bag_dir={bag_dir}")
    print(f"[info] target_hz={hz:.3f}  period={period_ns/1e6:.2f} ms")
    print("[info] saving color/depth pairs triggered by color frames")
    print(f"[info] topics: color={TOPIC_COLOR}, depth={TOPIC_DEPTH}, info={TOPIC_INFO}")

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        if topic == TOPIC_INFO and not camera_info_saved:
            msg = deserialize_message(data, InfoT)
            info_path = os.path.join(bag_dir, "camera_info.json")
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(camera_info_to_dict(msg), f, indent=2)
            camera_info_saved = True
            continue

        if topic == TOPIC_DEPTH:
            latest_depth = (t_ns, deserialize_message(data, DepthT))
            new_depth = True
            seen_depth += 1
            continue

        if topic != TOPIC_COLOR:
            continue

        latest_color = (t_ns, deserialize_message(data, ColorT))
        new_color = True
        seen_color += 1

        if (latest_depth is None) or (latest_color is None) or (not (new_color and new_depth)):
            continue

        if last_save_t_ns is not None and (t_ns - last_save_t_ns) < period_ns:
            continue

        _, msg_color = latest_color
        _, msg_depth = latest_depth

        try:
            bgr = image_msg_to_bgr8(msg_color)
            dep16 = depth_msg_to_png16(msg_depth)
        except Exception:
            # If conversion fails for a frame, skip it
            continue

        sec = int(t_ns // 1_000_000_000)
        nsec = int(t_ns % 1_000_000_000)
        ts_str = f"{sec}.{nsec:09d}"

        fname = f"{saved:06d}_{ts_str}.png"
        cpath = os.path.join(out_color, fname)
        dpath = os.path.join(out_depth, fname)

        ok1 = cv2.imwrite(cpath, bgr)
        ok2 = cv2.imwrite(dpath, dep16)
        if not (ok1 and ok2):
            continue

        last_save_t_ns = t_ns
        saved += 1
        new_color = False
        new_depth = False

        if saved % 50 == 0:
            print(f"[info] saved {saved} pairs (seen_color={seen_color}, seen_depth={seen_depth})")

    if not camera_info_saved:
        print("[warn] camera_info topic existed but no messages were saved.")

    print(f"[done] saved_pairs={saved}  out={bag_dir}")


if __name__ == "__main__":
    main()
