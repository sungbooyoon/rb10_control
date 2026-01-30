#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export ee_pos + ee_quat (7D) from a robomimic-style HDF5 into a single NPZ.

Output (A): concatenate all demos into one big array
- X        : (N_total, 7) float32   [ee_pos(3), ee_quat(4)]
- demo_id  : (N_total,)  int32      demo index per row
- skill_id : (N_total,)  int16      skill index per row (1..K), -1 if unknown
- t        : (N_total,)  int32      timestep within demo
- demo_ptr : (num_demos+1,) int64   prefix sums to slice each demo quickly
- demo_names : (num_demos,) object  demo key names (e.g., "demo_0")

Supports multiple ways skill mapping can appear:
1) Root groups: /skill_1, /skill_2, ... containing demo keys
2) Under /data/mask/skill_1 ... as groups containing demo keys
3) Per-demo goal string like "seam_17" (optional mapping to skill via --goal_to_skill)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np


SKILL_RE = re.compile(r"^skill_(\d+)$")
DEMO_RE = re.compile(r"^demo_(\d+)$")


def _sorted_demo_keys(data_group: h5py.Group) -> List[str]:
    keys = [k for k in data_group.keys() if k.startswith("demo_")]
    # sort by integer suffix
    def _idx(k: str) -> int:
        m = DEMO_RE.match(k)
        return int(m.group(1)) if m else 10**9
    return sorted(keys, key=_idx)


def _collect_skill_map_from_group(skill_group_parent: h5py.Group) -> Dict[str, int]:
    """
    Parse skill mapping in two formats:

    A) Group format:
        /mask/skill_1 (Group) contains keys: demo_0, demo_4, ...
    B) Dataset format:
        /mask/skill_1 (Dataset of object/bytes/str) contains values: ["demo_0", "demo_4", ...]
    Return: demo_key -> skill_id(int)
    """
    demo2skill: Dict[str, int] = {}

    for name, obj in skill_group_parent.items():
        m = SKILL_RE.match(name)
        if not m:
            continue
        skill_id = int(m.group(1))

        # ---- Case A: Group with demo_* keys
        if isinstance(obj, h5py.Group):
            for demo_key in obj.keys():
                if demo_key.startswith("demo_"):
                    demo2skill[demo_key] = skill_id
            continue

        # ---- Case B: Dataset listing demo keys
        if isinstance(obj, h5py.Dataset):
            arr = obj[()]  # could be scalar, list, np.ndarray

            # normalize to python list
            if np.isscalar(arr):
                arr_list = [arr]
            else:
                arr_list = list(arr)

            for v in arr_list:
                # decode bytes -> str
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="ignore")
                elif isinstance(v, np.bytes_):
                    v = v.tobytes().decode("utf-8", errors="ignore")

                v = str(v).strip()

                # sometimes stored like "b'demo_0'" or with quotes; keep robust
                v = v.strip(" \t\n\r'\"")

                if v.startswith("demo_"):
                    demo2skill[v] = skill_id

    return demo2skill


def _find_skill_mapping(f: h5py.File) -> Dict[str, int]:
    """
    Try multiple locations for skill->demo list mapping.
    """
    # 1) root: /skill_k
    demo2skill = _collect_skill_map_from_group(f)
    if demo2skill:
        return demo2skill

    # 2) /mask/skill_k 
    if "mask" in f:
        demo2skill = _collect_skill_map_from_group(f["mask"])
        if demo2skill:
            return demo2skill

    return {}


def _parse_goal_to_skill_map(spec: str) -> Dict[str, int]:
    """
    spec format example:
      "seam_1:1,seam_2:1,seam_17:3"
    Returns dict goal_str -> skill_id
    """
    out: Dict[str, int] = {}
    if not spec:
        return out
    items = [s.strip() for s in spec.split(",") if s.strip()]
    for it in items:
        if ":" not in it:
            raise ValueError(f"Bad --goal_to_skill item '{it}'. Use 'goal:skill'.")
        g, k = it.split(":", 1)
        out[g.strip()] = int(k.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, help="Input HDF5 path")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--data_key", default="data", help="Root group for demos (default: data)")
    ap.add_argument("--obs_pos", default="ee_pos", help="Observation key for position (default: ee_pos)")
    ap.add_argument("--obs_quat", default="ee_quat", help="Observation key for quaternion (default: ee_quat)")
    ap.add_argument(
        "--goal_to_skill",
        default="",
        help="Optional mapping 'goal:skill,goal:skill,...' (e.g., 'seam_1:1,seam_2:1') "
             "used only when skill mapping groups are absent.",
    )
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="X dtype")
    args = ap.parse_args()

    hdf5_path = Path(args.hdf5)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    goal2skill = _parse_goal_to_skill_map(args.goal_to_skill)

    with h5py.File(hdf5_path, "r") as f:
        if args.data_key not in f:
            raise KeyError(f"Group '{args.data_key}' not found in HDF5. Keys at root: {list(f.keys())}")

        data = f[args.data_key]
        demo_keys = _sorted_demo_keys(data)
        if not demo_keys:
            raise RuntimeError(f"No demo_* found under '{args.data_key}'")
        
        # --- skill mapping: demo_key -> skill_id
        demo2skill = _find_skill_mapping(f)

        # If still empty, try deriving from per-demo goal strings (optional)
        # demo has dataset 'goal' typically as bytes/string
        if not demo2skill and goal2skill:
            for dk in demo_keys:
                demo_grp = data[dk]
                if "goal" in demo_grp:
                    g = demo_grp["goal"][()]
                    if isinstance(g, bytes):
                        g = g.decode("utf-8", errors="ignore")
                    elif isinstance(g, np.ndarray) and g.dtype.kind in ("S", "U"):
                        # sometimes goal stored as array of strings
                        g = str(g.tolist())
                    else:
                        g = str(g)
                    if g in goal2skill:
                        demo2skill[dk] = int(goal2skill[g])

        # ---- precompute total length and demo_ptr
        lengths: List[int] = []
        for dk in demo_keys:
            obs = data[dk]["obs"]
            if args.obs_pos not in obs or args.obs_quat not in obs:
                raise KeyError(
                    f"{dk} missing obs/{args.obs_pos} or obs/{args.obs_quat}. "
                    f"Available obs keys: {list(obs.keys())}"
                )
            T = obs[args.obs_pos].shape[0]
            if obs[args.obs_quat].shape[0] != T:
                raise RuntimeError(f"{dk} pos len {T} != quat len {obs[args.obs_quat].shape[0]}")
            lengths.append(int(T))

        num_demos = len(demo_keys)
        demo_ptr = np.zeros((num_demos + 1,), dtype=np.int64)
        demo_ptr[1:] = np.cumsum(np.array(lengths, dtype=np.int64))
        N_total = int(demo_ptr[-1])

        X = np.empty((N_total, 7), dtype=np.float32 if args.dtype == "float32" else np.float64)
        demo_id = np.empty((N_total,), dtype=np.int32)
        skill_id = np.full((N_total,), -1, dtype=np.int16)
        t = np.empty((N_total,), dtype=np.int32)

        # ---- fill arrays
        cursor = 0
        for i, dk in enumerate(demo_keys):
            demo_grp = data[dk]
            obs = demo_grp["obs"]

            pos = obs[args.obs_pos][()]  # (T,3)
            quat = obs[args.obs_quat][()]  # (T,4)
            if pos.shape[1] != 3 or quat.shape[1] != 4:
                raise RuntimeError(f"{dk} expected pos(T,3) quat(T,4), got {pos.shape} {quat.shape}")

            T = pos.shape[0]
            sl = slice(cursor, cursor + T)

            X[sl, 0:3] = pos.astype(X.dtype, copy=False)
            X[sl, 3:7] = quat.astype(X.dtype, copy=False)

            demo_id[sl] = i
            t[sl] = np.arange(T, dtype=np.int32)

            if dk in demo2skill:
                skill_id[sl] = int(demo2skill[dk])

            cursor += T

        assert cursor == N_total

        demo_names = np.array(demo_keys, dtype=object)

    np.savez_compressed(
        out_path,
        X=X,
        demo_id=demo_id,
        skill_id=skill_id,
        t=t,
        demo_ptr=demo_ptr,
        demo_names=demo_names,
    )

    # quick summary
    unique_skills = sorted(set(int(s) for s in np.unique(skill_id) if s != -1))
    print(f"[saved] {out_path}")
    print(f"  X: {X.shape} dtype={X.dtype}")
    print(f"  demos: {len(demo_names)}  total_steps: {X.shape[0]}")
    print(f"  skills present: {unique_skills if unique_skills else '(none / all -1)'}")
    print("  slice demo i: X[demo_ptr[i]:demo_ptr[i+1]]")


if __name__ == "__main__":
    main()