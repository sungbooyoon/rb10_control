#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Safely remove demo_180 and shift demo indices >= 181 down by 1.
Uses temporary names to avoid HDF5 rename collisions.
Also fixes /data/mask/skill_* entries.
"""

import argparse
import h5py
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--remove", type=int, default=180)
    args = ap.parse_args()

    shutil.copy(args.hdf5, args.out)

    with h5py.File(args.out, "r+") as f:
        data = f["data"]

        demo_keys = sorted(
            [k for k in data.keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1])
        )

        remove_id = args.remove
        remove_key = f"demo_{remove_id}"

        # -------------------------
        # STEP 0: guard (only run if demo_216 exists)
        # -------------------------
        if "demo_216" not in data:
            print("[skip] demo_216 not found. "
                "Assuming file already processed or original demo count < 217.")
            return

        # -------------------------
        # STEP 1: delete demo_180
        # -------------------------
        if remove_key in data:
            print(f"[delete] {remove_key}")
            del data[remove_key]
        else:
            print(f"[warn] {remove_key} not found")

        # -------------------------
        # STEP 2: move demos >180 -> tmp_demo_x
        # -------------------------
        for dk in demo_keys:
            idx = int(dk.split("_")[1])
            if idx > remove_id and f"demo_{idx}" in data:
                tmp_name = f"tmp_demo_{idx}"
                print(f"[tmp] demo_{idx} -> {tmp_name}")
                data.move(f"demo_{idx}", tmp_name)

        # -------------------------
        # STEP 3: tmp_demo_x -> demo_{x-1}
        # -------------------------
        for dk in demo_keys:
            idx = int(dk.split("_")[1])
            if idx > remove_id:
                tmp_name = f"tmp_demo_{idx}"
                new_name = f"demo_{idx - 1}"
                if tmp_name in data:
                    print(f"[rename] {tmp_name} -> {new_name}")
                    data.move(tmp_name, new_name)

        # -------------------------
        # STEP 4: fix mask
        # -------------------------
        if "mask" in f:
            mask = f["mask"]
            for mk in list(mask.keys()):
                raw = mask[mk][...]
                new_list = []

                for s in raw:
                    demo = s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                    idx = int(demo.split("_")[1])
                    if idx == remove_id:
                        continue
                    elif idx > remove_id:
                        new_list.append(f"demo_{idx - 1}")
                    else:
                        new_list.append(demo)

                del mask[mk]
                mask.create_dataset(
                    mk,
                    data=[d.encode("utf-8") for d in new_list],
                    dtype=h5py.string_dtype("utf-8")
                )

    print("\n[DONE] demo index fixed safely using tmp rename.")


if __name__ == "__main__":
    main()
