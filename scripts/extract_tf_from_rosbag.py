#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract TF pose (parent -> child) from a ROS2 bag (MCAP) into CSV.
- Reads both /tf and /tf_static
- Writes time, translation, quaternion (xyzw)

Usage:
  python3 extract_tag_tf.py \
    --bag /path/to/demo_202601xx_xxxxxx \
    --parent link0 \
    --child tag36h11:1 \
    --out tag_pose.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# ----------------------------
# Transform math (t + quat xyzw)
# ----------------------------

@dataclass
class Tform:
    t: np.ndarray  # (3,)
    q: np.ndarray  # (4,) xyzw


def q_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def q_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # xyzw convention
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    ], dtype=np.float64)


def q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # v' = q * (v,0) * q_conj
    vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    return q_mul(q_mul(q, vq), q_conj(q))[:3]


def tf_inv(T: Tform) -> Tform:
    qi = q_conj(T.q)
    ti = -q_rotate(qi, T.t)
    return Tform(t=ti, q=qi)


def tf_mul(A: Tform, B: Tform) -> Tform:
    # A∘B
    t = A.t + q_rotate(A.q, B.t)
    q = q_mul(A.q, B.q)
    return Tform(t=t, q=q_normalize(q))


# ----------------------------
# TF graph lookup (chain parent->child)
# store edges as: parent -> child transform
# ----------------------------

def find_chain_transform(
    edges: Dict[Tuple[str, str], Tform],
    parent: str,
    child: str,
    max_depth: int = 50
) -> Optional[Tform]:
    """
    Find transform parent->child by BFS over directed edges and their inverses.
    edges contains parent->child.
    We allow traversing both directions by using inverse when needed.
    """
    if parent == child:
        return Tform(t=np.zeros(3), q=np.array([0.0, 0.0, 0.0, 1.0]))

    # adjacency: node -> list of (next_node, transform node->next_node)
    adj: Dict[str, List[Tuple[str, Tform]]] = {}
    for (p, c), T in edges.items():
        adj.setdefault(p, []).append((c, T))
        adj.setdefault(c, []).append((p, tf_inv(T)))  # reverse direction

    # BFS
    from collections import deque
    q = deque()
    q.append(parent)
    visited = {parent}
    prev: Dict[str, Tuple[str, Tform]] = {}  # node -> (prev_node, prev->node)

    depth = 0
    while q and depth < max_depth:
        for _ in range(len(q)):
            cur = q.popleft()
            if cur == child:
                # reconstruct
                T_acc = Tform(t=np.zeros(3), q=np.array([0.0, 0.0, 0.0, 1.0]))
                node = child
                while node != parent:
                    pnode, T_p_to_node = prev[node]
                    T_acc = tf_mul(T_p_to_node, T_acc)
                    node = pnode
                return T_acc
            for nxt, T_cur_to_nxt in adj.get(cur, []):
                if nxt in visited:
                    continue
                visited.add(nxt)
                prev[nxt] = (cur, T_cur_to_nxt)
                q.append(nxt)
        depth += 1

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, type=str, help="ROS2 bag directory (MCAP)")
    ap.add_argument("--parent", required=True, type=str)
    ap.add_argument("--child", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--sample_hz", type=float, default=0.0,
                    help="0이면 TF 메시지 도착 시점마다 저장. >0이면 해당 Hz로 리샘플링(간단).")
    args = ap.parse_args()

    bag_path = Path(args.bag)
    out_path = Path(args.out)

    reader = SequentialReader()
    storage_options = StorageOptions(uri=str(bag_path), storage_id="mcap")
    converter_options = ConverterOptions(input_serialization_format="cdr",
                                         output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topics}

    if "/tf" not in type_map and "/tf_static" not in type_map:
        raise RuntimeError("Bag에 /tf 또는 /tf_static이 없습니다. apriltag TF를 기록했는지 확인하세요.")

    TFMessage = get_message("tf2_msgs/msg/TFMessage")

    # current TF edges (static+dynamic merged; dynamic overwrites)
    edges: Dict[Tuple[str, str], Tform] = {}

    # for simple resampling
    next_sample_ns: Optional[int] = None
    if args.sample_hz and args.sample_hz > 0:
        period_ns = int(1e9 / args.sample_hz)
    else:
        period_ns = None

    rows = []
    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        if topic not in ("/tf", "/tf_static"):
            continue

        msg = deserialize_message(data, TFMessage)

        # update edges
        for tr in msg.transforms:
            p = tr.header.frame_id.strip()
            c = tr.child_frame_id.strip()
            # ROS TF frame_id sometimes starts with '/'
            if p.startswith("/"):
                p = p[1:]
            if c.startswith("/"):
                c = c[1:]
            T = Tform(
                t=np.array([tr.transform.translation.x,
                            tr.transform.translation.y,
                            tr.transform.translation.z], dtype=np.float64),
                q=q_normalize(np.array([tr.transform.rotation.x,
                                        tr.transform.rotation.y,
                                        tr.transform.rotation.z,
                                        tr.transform.rotation.w], dtype=np.float64))
            )
            edges[(p, c)] = T

        # decide whether to sample now
        do_sample = False
        if period_ns is None:
            do_sample = True
        else:
            if next_sample_ns is None:
                next_sample_ns = t_ns
                do_sample = True
            elif t_ns >= next_sample_ns:
                do_sample = True
                next_sample_ns += period_ns

        if not do_sample:
            continue

        T_pc = find_chain_transform(edges, args.parent, args.child)
        if T_pc is None:
            continue  # chain not available yet at this time

        stamp_sec = t_ns / 1e9
        rows.append([
            stamp_sec,
            float(T_pc.t[0]), float(T_pc.t[1]), float(T_pc.t[2]),
            float(T_pc.q[0]), float(T_pc.q[1]), float(T_pc.q[2]), float(T_pc.q[3]),
        ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "x", "y", "z", "qx", "qy", "qz", "qw"])
        w.writerows(rows)

    print(f"[OK] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
