#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read a ROS 2 rosbag2 MCAP file and compute pose of target frame(s) in source frame.

- Reads /tf and optionally /tf_static
- Maintains latest transforms in a graph
- For each timestamp, if a path exists source->target, composes transforms to get pose
- Saves results to CSV and NPZ

Requirements:
  pip install rosbags numpy

Usage examples:
  python extract_tf_link0_to_tag_from_mcap.py \
    --bag /path/to/demo_20260119_174306 \
    --source link0 \
    --target tag36h11:0 \
    --out_csv /tmp/link0_to_tag.csv \
    --out_npz /tmp/link0_to_tag.npz

  # If you want "any frame name containing tag36h11"
  python extract_tf_link0_to_tag_from_mcap.py \
    --bag /path/to/demo_20260119_174306 \
    --source link0 \
    --target_contains tag36h11
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Iterable

import numpy as np

from rosbags.highlevel import AnyReader


# ----------------------------
# Transform math (t + quat)
# ----------------------------

@dataclass(frozen=True)
class Tform:
    # translation (3,)
    t: np.ndarray
    # quaternion xyzw (4,)
    q: np.ndarray

def _q_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n

def _q_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

def _q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # Hamilton product, quats are xyzw
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return _q_normalize(np.array([x, y, z, w], dtype=np.float64))

def _q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # rotate vector v by quaternion q (xyzw)
    q = _q_normalize(q)
    v = np.asarray(v, dtype=np.float64)
    # v' = q * (v,0) * q_conj
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    return _q_mul(_q_mul(q, qv), _q_conj(q))[:3]

def tform_inv(T: Tform) -> Tform:
    qinv = _q_conj(_q_normalize(T.q))
    tinv = -_q_rotate(qinv, T.t)
    return Tform(t=tinv, q=qinv)

def tform_mul(A: Tform, B: Tform) -> Tform:
    # A ∘ B  (apply B then A)
    t = A.t + _q_rotate(A.q, B.t)
    q = _q_mul(A.q, B.q)
    return Tform(t=t, q=q)

IDENTITY = Tform(t=np.zeros(3, dtype=np.float64), q=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))


# ----------------------------
# TF graph buffer
# ----------------------------

class TfBufferGraph:
    """
    Stores latest transforms as directed edges parent->child.
    Allows querying composed transform source->target by finding a path
    in an undirected view of the graph (using inverse edges when traversing backward).
    """
    def __init__(self):
        self._edges: Dict[Tuple[str, str], Tform] = {}  # (parent, child) -> T(parent->child)

    def update(self, parent: str, child: str, T_parent_child: Tform):
        self._edges[(parent, child)] = T_parent_child

    def _neighbors(self, frame: str) -> Iterable[Tuple[str, Tform]]:
        # Return neighbors with transform from 'frame' to neighbor.
        # If we have (frame -> nb), use it.
        # If we have (nb -> frame), use inverse.
        for (p, c), T in self._edges.items():
            if p == frame:
                yield c, T
            elif c == frame:
                yield p, tform_inv(T)

    def lookup(self, source: str, target: str, max_hops: int = 64) -> Optional[Tform]:
        if source == target:
            return IDENTITY

        # BFS for path + transforms along path
        from collections import deque

        visited = set([source])
        q = deque()
        q.append((source, IDENTITY))  # current frame, T(source->current)

        hops = 0
        while q and hops < 200000:
            cur, T_source_cur = q.popleft()
            if cur == target:
                return T_source_cur

            for nb, T_cur_nb in self._neighbors(cur):
                if nb in visited:
                    continue
                visited.add(nb)

                # T(source->nb) = T(source->cur) ∘ T(cur->nb)
                T_source_nb = tform_mul(T_source_cur, T_cur_nb)
                q.append((nb, T_source_nb))

            hops += 1
            if len(visited) > max_hops * 500:  # safety guard
                break

        return None

    def known_frames(self) -> List[str]:
        frames = set()
        for (p, c) in self._edges.keys():
            frames.add(p)
            frames.add(c)
        return sorted(frames)


# ----------------------------
# Main bag reading
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Path to rosbag2 directory (containing metadata.yaml) or .mcap")
    ap.add_argument("--source", required=True, help="Source frame (e.g., link0)")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--target", help="Exact target frame name (e.g., tag36h11:0)")
    grp.add_argument("--target_contains", help="Substring match for target frames (e.g., tag36h11)")
    ap.add_argument("--tf_topic", default="/tf", help="TF topic (default: /tf)")
    ap.add_argument("--tf_static_topic", default="/tf_static", help="TF static topic (default: /tf_static)")
    ap.add_argument("--include_tf_static", action="store_true", help="Also ingest /tf_static")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--out_npz", required=True, help="Output NPZ path")
    ap.add_argument("--sample_every", type=int, default=1,
                    help="Record every N-th TFMessage (default: 1). Increase to downsample.")
    return ap.parse_args()

def _tform_from_ros_transform(transform_msg) -> Tform:
    t = np.array([transform_msg.translation.x,
                  transform_msg.translation.y,
                  transform_msg.translation.z], dtype=np.float64)
    q = np.array([transform_msg.rotation.x,
                  transform_msg.rotation.y,
                  transform_msg.rotation.z,
                  transform_msg.rotation.w], dtype=np.float64)
    return Tform(t=t, q=_q_normalize(q))

def main():
    args = parse_args()
    bag_path = Path(args.bag)
    out_csv = Path(args.out_csv)
    out_npz = Path(args.out_npz)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    buf = TfBufferGraph()

    rows = []  # (t_ns, target_frame, x,y,z,qx,qy,qz,qw)

    # AnyReader can open rosbag2 directory or single file storage (including mcap) depending on install.
    with AnyReader([bag_path]) as reader:
        # figure out available connections
        connections = list(reader.connections)
        tf_conns = [c for c in connections if c.topic == args.tf_topic]
        if args.include_tf_static:
            tf_conns += [c for c in connections if c.topic == args.tf_static_topic]

        if not tf_conns:
            available = sorted(set(c.topic for c in connections))
            raise RuntimeError(
                f"No TF connections found. Requested {args.tf_topic}"
                + (f" and {args.tf_static_topic}" if args.include_tf_static else "")
                + f"\nAvailable topics include:\n  " + "\n  ".join(available[:200])
            )

        # map typestore
        typestore = reader.typestore

        msg_idx = 0
        for conn, t_ns, raw in reader.messages(connections=tf_conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)

            # tf2_msgs/msg/TFMessage: transforms[]
            # Each element has header.stamp, header.frame_id, child_frame_id, transform
            if not hasattr(msg, "transforms"):
                continue

            # Update graph with all transforms in this message
            for tr in msg.transforms:
                parent = tr.header.frame_id.strip()
                child = tr.child_frame_id.strip()
                T = _tform_from_ros_transform(tr.transform)
                if parent and child:
                    buf.update(parent, child, T)

            if (msg_idx % max(args.sample_every, 1)) != 0:
                msg_idx += 1
                continue
            msg_idx += 1

            # Determine which target frames to compute
            if args.target:
                targets = [args.target]
            else:
                # substring match among known frames
                targets = [f for f in buf.known_frames() if args.target_contains in f]

            for tgt in targets:
                T_src_tgt = buf.lookup(args.source, tgt)
                if T_src_tgt is None:
                    continue
                x, y, z = T_src_tgt.t.tolist()
                qx, qy, qz, qw = T_src_tgt.q.tolist()
                rows.append((int(t_ns), tgt, x, y, z, qx, qy, qz, qw))

    # Save CSV
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ns", "target_frame", "x", "y", "z", "qx", "qy", "qz", "qw"])
        w.writerows(rows)

    # Save NPZ
    if rows:
        t_ns = np.array([r[0] for r in rows], dtype=np.int64)
        target = np.array([r[1] for r in rows], dtype=object)
        pose = np.array([[r[2], r[3], r[4], r[5], r[6], r[7], r[8]] for r in rows], dtype=np.float64)
    else:
        t_ns = np.zeros((0,), dtype=np.int64)
        target = np.zeros((0,), dtype=object)
        pose = np.zeros((0, 7), dtype=np.float64)

    np.savez(out_npz, t_ns=t_ns, target_frame=target, pose_xyzw=pose)

    print(f"Done. Wrote {len(rows)} poses.")
    print(f"CSV: {out_csv}")
    print(f"NPZ: {out_npz}")

if __name__ == "__main__":
    main()