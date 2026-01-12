#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rainbow Robotics - Analytic IK (RB10-1300) | Python/Numpy port
- Input  : position [m], orientation [rad]
- Output : joint angles [deg]
NOTE:
- This is a direct port of the provided Octave script. It assumes the same
  kinematic conventions (link params, ZYX order, frame definitions).
"""

import numpy as np

D2R = np.pi / 180.0
R2D = 180.0 / np.pi

def clamp(x, lo=-1.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)

def rot_zyx(rz, ry, rx):
    """R = Rz(rz) * Ry(ry) * Rx(rx), angles in [rad]."""
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    Rz = np.array([[cz, -sz, 0.0],
                   [sz,  cz, 0.0],
                   [0.0, 0.0, 1.0]])
    Ry = np.array([[ cy, 0.0,  sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0,  cy]])
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cx, -sx],
                   [0.0,  sx,  cx]])
    return Rz @ Ry @ Rx

def T_from_R_P(R, P):
    """Homogeneous transform from rotation R (3x3) and position P (3,)"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = P
    return T

def ik_rb10_1300(pos, rot):
    """
    Args (m, rad): pos = [x, y, z], rot = [rx, ry, rz] (ZYX euler in rad; Z->Y'->X'')
    Returns (deg): th1..th6
    """
    # ---------------- Link Length parameter (RB10-1300) [converted to meters] ----------------
    d1 = 197.0 * 1e-3
    d2 = 187.5 * 1e-3
    d3 = 148.4 * 1e-3
    d4 = 117.15 * 1e-3
    d5 = 117.15 * 1e-3
    d6 = 115.3 * 1e-3
    a1 = 612.7 * 1e-3
    a2 = 570.15 * 1e-3

    # ---------------- Inputs ----------------
    x, y, z = map(float, pos)  # already in meters
    rx, ry, rz = map(float, rot)

    R = rot_zyx(rz, ry, rx)
    P06 = np.array([x, y, z], dtype=float)
    Y06 = R[:, 1]  # second column

    # ---------------- Wrist position P05 ----------------
    P05 = P06 + d6 * Y06

    # ---------------- th1, th5, th6 ----------------
    th1 = np.arctan2(P05[1], P05[0]) - np.arccos(clamp(d4 / np.sqrt(P05[1]**2 + P05[0]**2))) + 0.5*np.pi
    th5 = +np.arccos(clamp((np.sin(th1)*P06[0] - np.cos(th1)*P06[1] - d4) / d6))

    # to avoid division by zero, add small eps inside atan2 terms if needed
    eps = 1e-12
    num = -(-np.sin(th1)*R[0,0] + np.cos(th1)*R[1,0])
    den =  (-np.sin(th1)*R[0,2] + np.cos(th1)*R[1,2])
    th6 = np.arctan2(num / (np.sin(th5) + eps), den / (np.sin(th5) + eps)) + 0.5*np.pi

    # ---------------- Build auxiliary transforms ----------------
    def A01(th1):
        return np.array([
            [np.cos(th1), -np.cos(-np.pi*0.5)*np.sin(th1),  np.sin(-np.pi*0.5)*np.sin(th1), 0.0],
            [np.sin(th1),  np.cos(-np.pi*0.5)*np.cos(th1), -np.sin(-np.pi*0.5)*np.cos(th1), 0.0],
            [0.0,          np.sin(-np.pi*0.5),              np.cos(-np.pi*0.5),             d1],
            [0.0,          0.0,                             0.0,                            1.0]
        ])

    def A67():
        return np.array([
            [np.cos(0.0), -np.cos(np.pi*0.5)*np.sin(0.0),  np.sin(np.pi*0.5)*np.sin(0.0), 0.0],
            [np.sin(0.0),  np.cos(np.pi*0.5)*np.cos(0.0), -np.sin(np.pi*0.5)*np.cos(0.0), 0.0],
            [0.0,          np.sin(np.pi*0.5),              np.cos(np.pi*0.5),             0.0],
            [0.0,          0.0,                            0.0,                            1.0]
        ])

    def A78(th5):
        return np.array([
            [np.cos(th5), -np.cos(-np.pi*0.5)*np.sin(th5),  np.sin(-np.pi*0.5)*np.sin(th5), 0.0],
            [np.sin(th5),  np.cos(-np.pi*0.5)*np.cos(th5), -np.sin(-np.pi*0.5)*np.cos(th5), 0.0],
            [0.0,          np.sin(-np.pi*0.5),              np.cos(-np.pi*0.5),             d5],
            [0.0,          0.0,                             0.0,                            1.0]
        ])

    def A89(th6):
        return np.array([
            [np.cos(th6), -np.cos(np.pi*0.5)*np.sin(th6),  np.sin(np.pi*0.5)*np.sin(th6),  0.0],
            [np.sin(th6),  np.cos(np.pi*0.5)*np.cos(th6), -np.sin(np.pi*0.5)*np.cos(th6),  0.0],
            [0.0,          np.sin(np.pi*0.5),              np.cos(np.pi*0.5),             -d6],
            [0.0,          0.0,                            0.0,                            1.0]
        ])

    T06 = T_from_R_P(R, P06)
    A01_ = A01(th1)
    A67_ = A67()
    A78_ = A78(th5)
    A89_ = A89(th6)

    A17 = np.linalg.inv(A01_) @ T06 @ np.linalg.inv(A89_) @ np.linalg.inv(A78_) @ np.linalg.inv(A67_)
    P14 = A17[:3, 3].copy()

    # ---------------- th3, th2 ----------------
    # th3
    cos_th3 = clamp((P14[0]**2 + P14[1]**2 - a1**2 - a2**2) / (2.0*a1*a2))
    th3 = +np.arccos(cos_th3)

    # th2
    r = np.sqrt(P14[0]**2 + P14[1]**2)
    term = clamp(a2*np.sin(th3) / (r + eps))
    th2 = np.arctan2(P14[0], -P14[1]) - np.arcsin(term)

    # ---------------- Remaining transforms for th4 ----------------
    def A12(th2):
        return np.array([
            [np.cos(th2 - np.pi*0.5), -np.cos(0.0)*np.sin(th2 - np.pi*0.5),  np.sin(0.0)*np.sin(th2 - np.pi*0.5), 0.0],
            [np.sin(th2 - np.pi*0.5),  np.cos(0.0)*np.cos(th2 - np.pi*0.5), -np.sin(0.0)*np.cos(th2 - np.pi*0.5), 0.0],
            [0.0,                      np.sin(0.0),                         np.cos(0.0),                        -d2],
            [0.0,                      0.0,                                 0.0,                                 1.0]
        ])

    def A23():
        return np.array([
            [np.cos(0.0), -np.cos(0.0)*np.sin(0.0),  np.sin(0.0)*np.sin(0.0), a1*np.cos(0.0)],
            [np.sin(0.0),  np.cos(0.0)*np.cos(0.0), -np.sin(0.0)*np.cos(0.0), a1*np.sin(0.0)],
            [0.0,           np.sin(0.0),             np.cos(0.0),             0.0],
            [0.0,           0.0,                     0.0,                     1.0]
        ])

    def A34(th3):
        return np.array([
            [np.cos(th3), -np.cos(0.0)*np.sin(th3),  np.sin(0.0)*np.sin(th3), 0.0],
            [np.sin(th3),  np.cos(0.0)*np.cos(th3), -np.sin(0.0)*np.cos(th3), 0.0],
            [0.0,           np.sin(0.0),             np.cos(0.0),             d3],
            [0.0,           0.0,                     0.0,                     1.0]
        ])

    def A45():
        return np.array([
            [np.cos(0.0), -np.cos(0.0)*np.sin(0.0),  np.sin(0.0)*np.sin(0.0), a2*np.cos(0.0)],
            [np.sin(0.0),  np.cos(0.0)*np.cos(0.0), -np.sin(0.0)*np.cos(0.0), a2*np.sin(0.0)],
            [0.0,           np.sin(0.0),             np.cos(0.0),             0.0],
            [0.0,           0.0,                     0.0,                     1.0]
        ])

    A12_ = A12(th2)
    A23_ = A23()
    A34_ = A34(th3)
    A45_ = A45()

    A56_cal = np.linalg.inv(A45_) @ np.linalg.inv(A34_) @ np.linalg.inv(A23_) @ np.linalg.inv(A12_) @ np.linalg.inv(A01_) @ T06 @ np.linalg.inv(A89_) @ np.linalg.inv(A78_) @ np.linalg.inv(A67_)

    th4 = np.arctan2(A56_cal[1, 0], A56_cal[0, 0]) - 0.5*np.pi

    # ---------------- Return in degrees ----------------
    return tuple(angle * R2D for angle in (th1, th2, th3, th4, th5, th6))


if __name__ == "__main__":
    # Example with your inputs
    pos = [0.5, -0.1, 0.3]
    rot = [-43.47, 80.56, -60.88]
    th = ik_rb10_1300(pos, rot)
    print("---------------------------------")
    print("Inverse Kinematics Result (deg)")
    print("---------------------------------")
    print("th1 = {:.6f}".format(th[0]))
    print("th2 = {:.6f}".format(th[1]))
    print("th3 = {:.6f}".format(th[2]))
    print("th4 = {:.6f}".format(th[3]))
    print("th5 = {:.6f}".format(th[4]))
    print("th6 = {:.6f}".format(th[5]))