#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster similar (x,y,z) rows in link0_to_tag.csv
and compute averaged pose per cluster.

- Clustering: DBSCAN on xyz
- Grouped per tag_frame
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def average_quaternion(quats: np.ndarray) -> np.ndarray:
    """
    Average quaternions using eigen method.
    quats: (N,4) array, xyzw
    """
    Q = np.zeros((4, 4))
    for q in quats:
        q = q / np.linalg.norm(q)
        Q += np.outer(q, q)
    Q /= len(quats)
    eigvals, eigvecs = np.linalg.eigh(Q)
    return eigvecs[:, np.argmax(eigvals)]


def process_tag(df_tag, eps, min_samples):
    xyz = df_tag[["x", "y", "z"]].to_numpy()

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(xyz)

    rows = []
    for label in sorted(set(labels)):
        if label == -1:
            continue  # noise 제거

        mask = labels == label
        cluster = df_tag[mask]

        xyz_mean = cluster[["x", "y", "z"]].mean().to_numpy()

        q_mean = average_quaternion(
            cluster[["qx", "qy", "qz", "qw"]].to_numpy()
        )

        rows.append({
            "tag_frame": cluster["tag_frame"].iloc[0],
            "count": len(cluster),
            "x": xyz_mean[0],
            "y": xyz_mean[1],
            "z": xyz_mean[2],
            "qx": q_mean[0],
            "qy": q_mean[1],
            "qz": q_mean[2],
            "qw": q_mean[3],
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input CSV (link0_to_tag.csv)")
    ap.add_argument("--eps", type=float, default=0.005, help="distance threshold in meters (default: 5mm)")
    ap.add_argument("--min_samples", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    all_rows = []

    for tag, df_tag in df.groupby("tag_frame"):
        rows = process_tag(df_tag, args.eps, args.min_samples)
        all_rows.extend(rows)

    df_out = pd.DataFrame(all_rows)
    df_out = df_out.sort_values(["tag_frame", "count"], ascending=[True, False])

    df_out.to_csv(args.out, index=False)
    print(f"[OK] {len(df_out)} averaged tag poses written → {args.out}")


if __name__ == "__main__":
    main()