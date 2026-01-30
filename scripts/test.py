import numpy as np
z = np.load("/home/sungboo/rb10_control/dataset/demo_20260122_final.npz", allow_pickle=True)

X = z["X_local_crop"]
ptr = z["demo_ptr"]

print("len(X) =", X.shape[0])
print("len(ptr) =", len(ptr), "=> D =", len(ptr)-1)
print("ptr[-1] =", ptr[-1], "(should equal len(X))")
print("has demo_names?", "demo_names" in z.files, "len(names) =", len(z["demo_names"]) if "demo_names" in z.files else None)