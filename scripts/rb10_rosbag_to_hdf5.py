#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate multiple ROS2 bags (MCAP files or bag directories) into a single
robomimic-compatible HDF5 with demo_0, demo_1, ...

Changes vs. previous version:
- ACTION = Δee_pos (3) + 6D relative rotation (first two columns of R_rel)
- Saved as HDF5 dataset: actions (N,9)  [Δx,Δy,Δz, r11,r21,r31, r12,r22,r32]
- Optional per-demo normalization to [-1, 1]; scale is length 9

Requirements:
- rosbag2_py, rosidl_runtime_py
- numpy, h5py, opencv-python

Conventions:
- (Optional) keep only segments where /rb/freedrive == True if --freedrive-only.
- OBS = separate ee_pos (m) and ee_quat (xyzw).
- Per-demo action normalization via 99th percentile or user-provided scales.

Author: ChatGPT (Sungboo’s assistant)
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm

import numpy as np
import h5py
import cv2

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# ---------------- Topics ----------------
DEFAULT_TOPICS = {
    "joint_states": "/rb/joint_states",          # sensor_msgs/msg/JointState
    "tcp_pose": "/rb/tcp_pose",                  # geometry_msgs/msg/PoseStamped
    "freedrive": "/rb/freedrive",                # std_msgs/msg/Bool (optional unless --freedrive-only)
    "ee_wrench": "/rb/ee_wrench",                # geometry_msgs/msg/WrenchStamped (optional)
    "rgb": "/camera/camera/color/image_raw",     # sensor_msgs/msg/Image (optional)
}


# ---------------- Utils ----------------
def ns_to_sec(ns: int) -> float:
    return ns * 1e-9

def quat_normalize_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return (q / n).astype(np.float64)

def quat_multiply_xyzw(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x, y, z, w], dtype=np.float64)

def quat_conjugate_xyzw(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64)

def quat_relative_xyzw(q_next_xyzw: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """
    q_rel = q_next * conj(q_curr) in xyzw.
    """
    q_next = quat_normalize_xyzw(q_next_xyzw)
    q_curr = quat_normalize_xyzw(q_xyzw)
    return quat_multiply_xyzw(q_next, quat_conjugate_xyzw(q_curr))

def quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    """
    q: (4,) in xyzw. returns 3x3 rotation matrix.
    """
    x, y, z, w = quat_normalize_xyzw(q)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz),  1 - 2*(xx + zz),      2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),    1 - 2*(xx + yy)]
    ], dtype=np.float64)
    return R

def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """
    Zhou et al. CVPR'19 6D rep = first two columns of R (flattened col-wise):
    [r11,r21,r31, r12,r22,r32]
    """
    c1 = R[:, 0]
    c2 = R[:, 1]
    return np.concatenate([c1, c2], axis=0).astype(np.float64)

def rosimg_to_rgb_numpy(msg) -> np.ndarray:
    h, w = msg.height, msg.width
    data = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc == "rgb8":
        return data.reshape((h, w, 3))
    if enc == "bgr8":
        return cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_BGR2RGB)
    if enc == "rgba8":
        return cv2.cvtColor(data.reshape((h, w, 4)), cv2.COLOR_RGBA2RGB)
    if enc == "bgra8":
        return cv2.cvtColor(data.reshape((h, w, 4)), cv2.COLOR_BGRA2RGB)
    if enc == "mono8":
        return cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_GRAY2RGB)
    # fallback
    arr = data.reshape((h, w, -1))
    if arr.shape[2] == 3:
        return arr
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")

def resize_if_needed(img_rgb: np.ndarray, target_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    if not target_hw or target_hw[0] <= 0 or target_hw[1] <= 0:
        return img_rgb
    H, W = target_hw
    return cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

def finite_diff(x: np.ndarray, dt: np.ndarray) -> np.ndarray:
    N, D = x.shape
    v = np.zeros_like(x, dtype=np.float32)
    if N <= 1:
        return v
    dx = x[1:] - x[:-1]
    dtc = np.clip(dt, 1e-6, None)
    v[:-1] = (dx / dtc[:, None]).astype(np.float32)
    v[-1] = v[-2]
    return v

def nearest_indices(ref_ts: np.ndarray, other_ts: np.ndarray) -> np.ndarray:
    idxs = np.searchsorted(other_ts, ref_ts)
    idxs = np.clip(idxs, 0, len(other_ts) - 1)
    prev = np.clip(idxs - 1, 0, len(other_ts) - 1)
    choose_prev = np.abs(other_ts[prev] - ref_ts) < np.abs(other_ts[idxs] - ref_ts)
    return np.where(choose_prev, prev, idxs)

def filter_pairs_by_tol(ref_ts: np.ndarray, other_ts: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    j = nearest_indices(ref_ts, other_ts)
    mask = np.abs(other_ts[j] - ref_ts) <= tol
    return mask, j


# ---------------- Rosbag Reader ----------------
@dataclass
class TopicInfo:
    name: str
    type_str: str
    msg_type: object

class BagReader:
    def __init__(self, bag_uri: str):
        self.bag_uri = bag_uri
        self.reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id="")  # autodetect
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                                        output_serialization_format="cdr")
        self.reader.open(storage_options, converter_options)
        self.topic_map: Dict[str, TopicInfo] = {}
        for c in self.reader.get_all_topics_and_types():
            self.topic_map[c.name] = TopicInfo(c.name, c.type, get_message(c.type))

    def read_all(self, wanted_topics: List[str]) -> Dict[str, List[Tuple[float, object]]]:
        out = {t: [] for t in wanted_topics}
        while self.reader.has_next():
            topic, data, t_ns = self.reader.read_next()
            if topic not in out:
                continue
            info = self.topic_map[topic]
            msg = deserialize_message(data, info.msg_type)
            out[topic].append((ns_to_sec(t_ns), msg))
        for t in out:
            out[t].sort(key=lambda x: x[0])
        return out


# ---------------- Core conversion (single bag -> dict) ----------------
def process_single_bag(
    bag_uri: str,
    topics: Dict[str, str],
    target_hz: float,
    sync_tol: float,
    image_resize: Optional[Tuple[int, int]],
    normalize_actions: bool,
    action_scale_json: Optional[str],
    freedrive_only: bool,
) -> Optional[Dict]:
    """
    Returns a dict with fields necessary to write a demo, or None if skipped.
    """
    # open bag
    try:
        reader = BagReader(bag_uri)
    except Exception as e:
        print(f"[WARN] Failed to open bag: {bag_uri} ({e})", file=sys.stderr)
        return None

    # required topics
    required = [topics["joint_states"], topics["tcp_pose"]]
    if freedrive_only:
        required.append(topics["freedrive"])
    for rt in required:
        if rt not in reader.topic_map:
            print(f"[WARN] Missing required topic in {bag_uri}: {rt} -> SKIP", file=sys.stderr)
            return None

    # wanted topics
    wanted = [topics["joint_states"], topics["tcp_pose"]]
    if topics.get("freedrive") and topics["freedrive"] in reader.topic_map:
        wanted.append(topics["freedrive"])
    if topics.get("ee_wrench"):
        wanted.append(topics["ee_wrench"])
    if topics.get("rgb"):
        wanted.append(topics["rgb"])
    data = reader.read_all(wanted)

    # reference timeline from joint_states
    js_times = np.array([t for (t, _) in data[topics["joint_states"]]], dtype=np.float64)
    js_msgs = [m for (_, m) in data[topics["joint_states"]]]
    if js_times.size < 2:
        print(f"[WARN] Not enough joint_states in {bag_uri} -> SKIP", file=sys.stderr)
        return None

    t0, t1 = js_times[0], js_times[-1]
    ref_ts = np.arange(t0, t1 + 1e-9, 1.0 / target_hz, dtype=np.float64) if target_hz > 0 else js_times.copy()

    js_mask, js_idx = filter_pairs_by_tol(ref_ts, js_times, tol=sync_tol)
    ref_ts = ref_ts[js_mask]
    js_idx = js_idx[js_mask]
    if len(ref_ts) < 2:
        print(f"[WARN] Too few synchronized samples in {bag_uri} -> SKIP", file=sys.stderr)
        return None

    # joints
    q_list, name_order = [], None
    for i in js_idx:
        m = js_msgs[i]
        if name_order is None:
            name_order = list(m.name)
        if list(m.name) != name_order:
            mpos = {n: p for n, p in zip(m.name, m.position)}
            q_list.append([mpos[n] for n in name_order])
        else:
            q_list.append(list(m.position))
    q = np.asarray(q_list, dtype=np.float32)
    qd = finite_diff(q, np.diff(ref_ts))

    # freedrive alignment
    freedrive = np.ones(len(ref_ts), dtype=bool)  # default keep-all
    if topics.get("freedrive") in data:
        fd_times = np.array([t for (t, _) in data[topics["freedrive"]]], dtype=np.float64)
        fd_msgs = [m for (_, m) in data[topics["freedrive"]]]
        if fd_times.size > 0:
            fd_mask, fd_idx = filter_pairs_by_tol(ref_ts, fd_times, tol=sync_tol)
            freedrive = np.zeros(len(ref_ts), dtype=bool)
            valid = fd_mask.nonzero()[0]
            freedrive[valid] = np.array([bool(fd_msgs[j].data) for j in fd_idx[fd_mask]], dtype=bool)
        else:
            freedrive = np.zeros(len(ref_ts), dtype=bool)

        if freedrive_only and not np.any(freedrive):
            print(f"[WARN] No freedrive==True samples in {bag_uri} -> SKIP", file=sys.stderr)
            return None
    else:
        if freedrive_only:
            print(f"[WARN] freedrive_only=True but topic missing in {bag_uri} -> SKIP", file=sys.stderr)
            return None

    # tcp pose -> pos + quat(xyzw)
    tp_times = np.array([t for (t, _) in data[topics["tcp_pose"]]], dtype=np.float64)
    tp_msgs = [m for (_, m) in data[topics["tcp_pose"]]]
    tp_mask, tp_idx = filter_pairs_by_tol(ref_ts, tp_times, tol=sync_tol)
    pos = np.zeros((len(ref_ts), 3), dtype=np.float64)
    quat_xyzw = np.zeros((len(ref_ts), 4), dtype=np.float64)
    last_p = None
    last_q = None
    for i in range(len(ref_ts)):
        if tp_mask[i]:
            m = tp_msgs[tp_idx[i]]
            p = m.pose.position
            q_wxyz = (m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z)
            pos[i] = [p.x, p.y, p.z]
            quat_xyzw[i] = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)  # xyzw
            last_p, last_q = pos[i], quat_xyzw[i]
        else:
            if last_p is not None:
                pos[i], quat_xyzw[i] = last_p, last_q

    # optional wrench
    ee_wrench = None
    if topics.get("ee_wrench") and len(data.get(topics["ee_wrench"], [])) > 0:
        wr_times = np.array([t for (t, _) in data[topics["ee_wrench"]]], dtype=np.float64)
        wr_msgs = [m for (_, m) in data[topics["ee_wrench"]]]
        wr_mask, wr_idx = filter_pairs_by_tol(ref_ts, wr_times, tol=sync_tol)
        W = np.zeros((len(ref_ts), 6), dtype=np.float32)
        for i in range(len(ref_ts)):
            if wr_mask[i]:
                m = wr_msgs[wr_idx[i]]
                f = m.wrench.force
                t = m.wrench.torque
                W[i] = [f.x, f.y, f.z, t.x, t.y, t.z]
            else:
                if i > 0:
                    W[i] = W[i-1]
        ee_wrench = W

    # optional rgb
    rgb = None
    if topics.get("rgb") and len(data.get(topics["rgb"], [])) > 0:
        im_times = np.array([t for (t, _) in data[topics["rgb"]]], dtype=np.float64)
        im_msgs = [m for (_, m) in data[topics["rgb"]]]
        im_mask, im_idx = filter_pairs_by_tol(ref_ts, im_times, tol=sync_tol)
        imgs: List[np.ndarray] = []
        last_img = None
        for i in range(len(ref_ts)):
            if im_mask[i]:
                arr = rosimg_to_rgb_numpy(im_msgs[im_idx[i]])
                arr = resize_if_needed(arr, image_resize)
                last_img = arr
            imgs.append(last_img.copy() if last_img is not None else None)
        first_valid = next((k for k, v in enumerate(imgs) if v is not None), None)
        if first_valid is not None:
            for k in range(len(imgs)):
                if imgs[k] is None:
                    imgs[k] = imgs[first_valid]
            rgb = np.stack(imgs, axis=0).astype(np.uint8)

    # keep mask
    valid_pose = np.logical_and(np.isfinite(pos).all(axis=1), np.isfinite(quat_xyzw).all(axis=1))
    nonzero_pose = np.logical_or((np.linalg.norm(pos, axis=1) > 0.0), (np.linalg.norm(quat_xyzw, axis=1) > 0.0))
    keep = np.logical_and(valid_pose, nonzero_pose)
    if freedrive_only:
        keep = np.logical_and(keep, freedrive)

    if np.count_nonzero(keep) < 2:
        print(f"[WARN] Too few samples after filtering in {bag_uri} -> SKIP", file=sys.stderr)
        return None

    # apply mask
    ref_ts = ref_ts[keep]
    q = q[keep]
    qd = qd[keep]
    pos = pos[keep]
    quat_xyzw = quat_xyzw[keep]
    if ee_wrench is not None:
        ee_wrench = ee_wrench[keep]
    if rgb is not None:
        rgb = rgb[keep]

    # ----- actions = Δpos(3) + 6D rot (relative) -> flat 9D -----
    def compute_action_ee_delta_6d(pos: np.ndarray,
                                   quat_xyzw: np.ndarray,
                                   normalize: bool,
                                   given_scale9: Optional[np.ndarray]
                                   ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns:
          actions9: (N,9) [Δx,Δy,Δz, r11,r21,r31, r12,r22,r32]
          scale: (9,) if normalize else None
        """
        N = pos.shape[0]
        dpos  = np.zeros((N, 3), dtype=np.float64)
        r6d   = np.zeros((N, 6), dtype=np.float64)

        if N >= 2:
            dpos[:-1] = pos[1:] - pos[:-1]
            for i in range(N - 1):
                q_rel = quat_relative_xyzw(quat_xyzw[i + 1], quat_xyzw[i])
                R_rel = quat_to_rotmat_xyzw(q_rel)
                r6d[i] = rotmat_to_6d(R_rel)
            dpos[-1] = dpos[-2]
            r6d[-1]  = r6d[-2]

        actions = np.concatenate([dpos, r6d], axis=1)  # (N,9)

        if not normalize:
            return actions.astype(np.float32), None

        if given_scale9 is not None:
            s = np.clip(np.abs(given_scale9.astype(np.float32)), 1e-9, None)
            assert s.shape == (9,), "action-scale must be length 9"
        else:
            s = np.percentile(np.abs(actions), 99, axis=0).astype(np.float32)
            s = np.clip(s, 1e-9, None)

        actions_n = np.clip(actions / s[None, :], -1.0, 1.0).astype(np.float32)
        return actions_n, s

    given_scale = None
    if action_scale_json:
        given_scale = np.array(json.loads(action_scale_json), dtype=np.float32)

    actions9, action_scale = compute_action_ee_delta_6d(
        pos=pos, quat_xyzw=quat_xyzw,
        normalize=normalize_actions, given_scale9=given_scale
    )

    # ----- obs / next_obs -----
    def shift_next(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if x is None or len(x) <= 1:
            return x
        return np.concatenate([x[1:], x[-1:]], axis=0)

    ee_pos  = pos.astype(np.float32)          # (N,3)
    ee_quat = quat_xyzw.astype(np.float32)    # (N,4) xyzw

    demo = {
        "N": len(ref_ts),
        "actions": actions9.astype(np.float32),      # (N,9)
        "rewards": np.zeros((len(ref_ts),), dtype=np.float32),
        "dones": np.concatenate([np.zeros((len(ref_ts)-1,), dtype=np.uint8), np.array([1], dtype=np.uint8)], axis=0),
        "obs": {
            "joint_pos": q.astype(np.float32),
            "joint_vel": qd.astype(np.float32),
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "ee_wrench": ee_wrench.astype(np.float32) if ee_wrench is not None else None,
            "rgb": rgb.astype(np.uint8) if rgb is not None else None,
        },
        "next_obs": {
            "joint_pos": shift_next(q).astype(np.float32),
            "joint_vel": shift_next(qd).astype(np.float32),
            "ee_pos": shift_next(ee_pos).astype(np.float32),
            "ee_quat": shift_next(ee_quat).astype(np.float32),
            "ee_wrench": shift_next(ee_wrench).astype(np.float32) if ee_wrench is not None else None,
            "rgb": shift_next(rgb).astype(np.uint8) if rgb is not None else None,
        },
        "meta": {
            "topic_map": topics,
            "joint_names": name_order,
            "timeline_hz": target_hz,
            "sync_tol": sync_tol,
            "action_type": "ee_delta_xyz + rot6d_rel",
            "action_format": "actions = [Δx,Δy,Δz, r11,r21,r31, r12,r22,r32]",
            "action_dim": 9,
            "action_scale": action_scale.tolist() if action_scale is not None else None,  # len 9 or None
            "bag_uri": bag_uri,
            "freedrive_only": bool(freedrive_only),
        }
    }
    return demo


# ---------------- Writer ----------------
def write_many_demos(out_path: str, demos: List[Dict], env_name: str, env_type: str, env_kwargs: Dict):
    """
    Write demos into a robomimic HDF5 with tqdm progress.
    """
    if len(demos) == 0:
        raise RuntimeError("No valid demos to write.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with h5py.File(out_path, "w") as f:
        g_data = f.create_group("data")
        g_data.attrs["env_args"] = json.dumps({
            "name": env_name,
            "type": env_type,
            "env_name": env_name,
            "env_type": env_type,
            "env_kwargs": env_kwargs or {},
        })

        total = 0
        iterator = enumerate(demos)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(demos), desc="Writing demos", unit="demo")

        for k, demo in iterator:
            name = f"demo_{k}"
            g = g_data.create_group(name)
            N = int(demo["N"])

            # core datasets
            g.attrs["num_samples"] = N
            g.create_dataset("states", data=np.zeros((N, 0), dtype=np.float32))
            g.create_dataset("actions", data=demo["actions"])  # (N,9)
            g.create_dataset("rewards", data=demo["rewards"])
            g.create_dataset("dones", data=demo["dones"])

            # obs
            go = g.create_group("obs")
            go.create_dataset("joint_pos", data=demo["obs"]["joint_pos"])
            go.create_dataset("joint_vel", data=demo["obs"]["joint_vel"])
            go.create_dataset("ee_pos",  data=demo["obs"]["ee_pos"])     # (N,3)
            go.create_dataset("ee_quat", data=demo["obs"]["ee_quat"])    # (N,4) xyzw
            if demo["obs"]["ee_wrench"] is not None:
                go.create_dataset("ee_wrench", data=demo["obs"]["ee_wrench"])
            if demo["obs"]["rgb"] is not None:
                go.create_dataset("rgb", data=demo["obs"]["rgb"], compression="gzip", compression_opts=4)

            # next_obs
            gn = g.create_group("next_obs")
            gn.create_dataset("joint_pos", data=demo["next_obs"]["joint_pos"])
            gn.create_dataset("joint_vel", data=demo["next_obs"]["joint_vel"])
            gn.create_dataset("ee_pos",  data=demo["next_obs"]["ee_pos"])
            gn.create_dataset("ee_quat", data=demo["next_obs"]["ee_quat"])
            if demo["next_obs"]["ee_wrench"] is not None:
                gn.create_dataset("ee_wrench", data=demo["next_obs"]["ee_wrench"])
            if demo["next_obs"]["rgb"] is not None:
                gn.create_dataset("rgb", data=demo["next_obs"]["rgb"], compression="gzip", compression_opts=4)

            g.attrs["meta"] = json.dumps(demo["meta"])
            total += N

            if tqdm is not None:
                iterator.set_postfix_str(f"{name} N={N}")

        g_data.attrs["total"] = int(total)

    print(f"[OK] Wrote {len(demos)} demos to {out_path} (total samples={total})")


# ---------------- Helpers ----------------
def discover_bag_uris(folder: str) -> List[str]:
    uris: List[str] = []
    for root, dirs, files in os.walk(folder):
        if "metadata.yaml" in files:
            uris.append(root)
        for fn in files:
            if fn.endswith(".mcap") or fn.endswith(".db3"):
                uris.append(os.path.join(root, fn))
    uris = sorted(list(dict.fromkeys(uris)))
    return uris


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Convert all bags in a folder to a single robomimic HDF5 (demo_0, demo_1, ...)")
    p.add_argument("--folder", required=True, help="Folder containing ROS2 bag files (.mcap/.db3) or bag dirs (with metadata.yaml)")
    p.add_argument("--out", required=True, help="Output HDF5 file")
    p.add_argument("--hz", type=float, default=30.0, help="Target resample Hz (joint_states reference)")
    p.add_argument("--sync-tol", type=float, default=0.05, help="Max time diff (sec) for nearest sync")
    p.add_argument("--image-resize", nargs=2, type=int, default=None, metavar=("H", "W"),
                   help="Resize RGB to (H W); omit to keep original")
    p.add_argument("--normalize-actions", action="store_true", default=True)
    p.add_argument("--no-normalize-actions", action="store_false", dest="normalize_actions")
    p.add_argument("--action-scale-json", type=str, default=None,
                   help='JSON list of 9 scales for [dX,dY,dZ, rot6d(6)], e.g. "[0.02,0.02,0.02, 0.5,0.5,0.5, 0.5,0.5,0.5]"')
    p.add_argument("--topic-joint-states", type=str, default=DEFAULT_TOPICS["joint_states"])
    p.add_argument("--topic-tcp-pose", type=str, default=DEFAULT_TOPICS["tcp_pose"])
    p.add_argument("--topic-freedrive", type=str, default=DEFAULT_TOPICS["freedrive"])
    p.add_argument("--topic-ee-wrench", type=str, default=DEFAULT_TOPICS["ee_wrench"])
    p.add_argument("--topic-rgb", type=str, default=DEFAULT_TOPICS["rgb"])
    p.add_argument("--env-name", type=str, default="KinestheticTask")
    p.add_argument("--env-type", type=str, default="gym")
    p.add_argument("--env-kwargs", type=str, default=None, help="JSON string for env_kwargs")
    p.add_argument("--freedrive-only", action="store_true", default=False,
                   help="Use only samples where /rb/freedrive==True (requires freedrive topic).")
    p.add_argument("--no-freedrive-only", action="store_false", dest="freedrive_only",
                   help="Ignore freedrive and use all valid pose samples.")
    return p.parse_args()

def main():
    args = parse_args()
    topics = {
        "joint_states": args.topic_joint_states,
        "tcp_pose": args.topic_tcp_pose,
        "freedrive": args.topic_freedrive,
        "ee_wrench": args.topic_ee_wrench,
        "rgb": args.topic_rgb,
    }
    env_kwargs = json.loads(args.env_kwargs) if args.env_kwargs else {}

    uris = discover_bag_uris(args.folder)
    if len(uris) == 0:
        print(f"[ERR] No bag URIs found under: {args.folder}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(uris)} candidate bag(s).")
    demos: List[Dict] = []
    for uri in uris:
        print(f"[INFO] Processing: {uri}")
        demo = process_single_bag(
            bag_uri=uri,
            topics=topics,
            target_hz=args.hz,
            sync_tol=args.sync_tol,
            image_resize=tuple(args.image_resize) if args.image_resize else None,
            normalize_actions=args.normalize_actions,
            action_scale_json=args.action_scale_json,
            freedrive_only=args.freedrive_only,
        )
        if demo is not None:
            demos.append(demo)
        else:
            print(f"[INFO] Skipped: {uri}")

    if len(demos) == 0:
        print("[ERR] No valid demos produced.", file=sys.stderr)
        sys.exit(2)

    write_many_demos(args.out, demos, args.env_name, args.env_type, env_kwargs)

if __name__ == "__main__":
    main()
