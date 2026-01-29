#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract TF poses for ALL AprilTag frames (e.g. tag36h11:*)
from a ROS2 MCAP bag into CSV.

- Reads /tf and /tf_static
- Computes parent -> tag transform
- Writes tag_frame name into CSV

Usage:
python3 extract_all_tags_tf.py \
  --bag /path/to/demo_xxx \
  --parent link0 \
  --tag_prefix tag36h11: \
  --out tags_pose.csv
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


# =========================
# Transform math
# =========================

@dataclass
class Tform:
    t: np.ndarray  # (3,)
    q: np.ndarray  # (4,) xyzw


def q_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0, 0, 0, 1], dtype=np.float64)
    return q / n


def q_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    ], dtype=np.float64)


def q_rotate(q, v):
    return q_mul(q_mul(q, np.array([*v, 0.0])), q_conj(q))[:3]


def tf_inv(T):
    qi = q_conj(T.q)
    ti = -q_rotate(qi, T.t)
    return Tform(t=ti, q=qi)


def tf_mul(A, B):
    t = A.t + q_rotate(A.q, B.t)
    q = q_normalize(q_mul(A.q, B.q))
    return Tform(t=t, q=q)


# =========================
# TF graph search
# =========================

def find_chain(edges, parent, child, max_depth=50):
    if parent == child:
        return Tform(np.zeros(3), np.array([0, 0, 0, 1]))

    adj = {}
    for (p, c), T in edges.items():
        adj.setdefault(p, []).append((c, T))
        adj.setdefault(c, []).append((p, tf_inv(T)))

    from collections import deque
    q = deque([parent])
    visited = {parent}
    prev = {}

    while q:
        cur = q.popleft()
        if cur == child:
            Tacc = Tform(np.zeros(3), np.array([0, 0, 0, 1]))
            n = child
            while n != parent:
                p, Tpn = prev[n]
                Tacc = tf_mul(Tpn, Tacc)
                n = p
            return Tacc

        for nxt, Tcur in adj.get(cur, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            prev[nxt] = (cur, Tcur)
            q.append(nxt)

    return None


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--tag_prefix", default="tag36h11:")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=args.bag, storage_id="mcap"),
        ConverterOptions("cdr", "cdr")
    )

    TFMessage = get_message("tf2_msgs/msg/TFMessage")

    edges: Dict[Tuple[str, str], Tform] = {}

    rows = []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in ("/tf", "/tf_static"):
            continue

        msg = deserialize_message(data, TFMessage)

        # update TF edges
        for tr in msg.transforms:
            p = tr.header.frame_id.lstrip("/")
            c = tr.child_frame_id.lstrip("/")

            T = Tform(
                t=np.array([
                    tr.transform.translation.x,
                    tr.transform.translation.y,
                    tr.transform.translation.z
                ]),
                q=q_normalize(np.array([
                    tr.transform.rotation.x,
                    tr.transform.rotation.y,
                    tr.transform.rotation.z,
                    tr.transform.rotation.w
                ]))
            )
            edges[(p, c)] = T

        # collect all tag frames currently known
        tag_frames = sorted({
            c for (_, c) in edges.keys()
            if c.startswith(args.tag_prefix)
        })

        for tag in tag_frames:
            Tpt = find_chain(edges, args.parent, tag)
            if Tpt is None:
                continue

            rows.append([
                t_ns / 1e9,
                tag,
                Tpt.t[0], Tpt.t[1], Tpt.t[2],
                Tpt.q[0], Tpt.q[1], Tpt.q[2], Tpt.q[3]
            ])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t_sec", "tag_frame",
            "x", "y", "z",
            "qx", "qy", "qz", "qw"
        ])
        w.writerows(rows)

    print(f"[OK] {len(rows)} rows written → {out}")


if __name__ == "__main__":
    main()