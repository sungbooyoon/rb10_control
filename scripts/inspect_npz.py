#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect an .npz and guess which key stores skill labels.

Usage:
  python inspect_npz_keys.py --npz /home/sungboo/rb10_control/dataset/demo_20260122_final.npz
"""

from __future__ import annotations
import argparse
import numpy as np


def _pretty_shape(a):
    try:
        return str(a.shape)
    except Exception:
        return "<?>"

def _safe_unique(a, max_unique=30):
    try:
        u = np.unique(a)
        if u.size > max_unique:
            return np.concatenate([u[:max_unique], ["..."]]).tolist()
        return u.tolist()
    except Exception:
        return None

def _looks_like_skill_label(name: str, arr: np.ndarray) -> bool:
    n = name.lower()
    # name-based hints
    name_hit = any(k in n for k in [
        "skill", "label", "class", "mask", "seg", "phase", "primitive", "pa"
    ])
    # value-based hints: 1D int-ish labels aligned with steps or demos
    try:
        if arr.dtype == object:
            return name_hit
        if arr.ndim in (1, 2) and np.issubdtype(arr.dtype, np.integer):
            return name_hit or True
        # float labels also possible but rarer; still flag if name looks right
        if arr.ndim in (1, 2) and np.issubdtype(arr.dtype, np.floating) and name_hit:
            return True
    except Exception:
        pass
    return name_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)

    print("=" * 80)
    print(f"[NPZ] {args.npz}")
    print(f"[keys] {len(d.files)} keys")
    print("=" * 80)

    # print all keys sorted
    for k in sorted(d.files):
        a = d[k]
        dtype = getattr(a, "dtype", type(a))
        info = f"- {k:30s}  shape={_pretty_shape(a):12s}  dtype={dtype}"
        print(info)
        # show unique values if feasible
        # u = None
        # if isinstance(a, np.ndarray) and a.dtype != object and a.size <= 2_000_000:
        #     u = _safe_unique(a, max_unique=30)
        # if u is not None:
        #     print(f"  - {k:30s}  {info}  unique={u}")
        # else:
        #     print(f"  - {k:30s}  {info}")
    d.close()


if __name__ == "__main__":
    main()
