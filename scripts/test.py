import h5py, shutil

src="/home/sungboo/rb10_control/data/demo_20260122_no_rgb.hdf5"
out="/home/sungboo/rb10_control/data/demo_20260122_no_rgb_new.hdf5"
shutil.copy(src, out)

with h5py.File(out, "r+") as f:
    if "data" not in f:
        raise SystemExit("Missing /data")
    data = f["data"]

    if "mask" not in data:
        print("No /data/mask found. Nothing to do.")
    else:
        # Ensure /mask exists
        if "mask" not in f:
            f.create_group("mask")

        # Copy each filter dataset under /data/mask/* to /mask/*
        for fk in data["mask"].keys():
            src_path = f"/data/mask/{fk}"
            dst_path = f"/mask/{fk}"

            if dst_path in f:
                del f[dst_path]
            f.copy(src_path, dst_path)
            print("[copy]", src_path, "->", dst_path)

        # Delete /data/mask
        del data["mask"]
        print("[delete] /data/mask")

print("wrote:", out)