#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible script wrapper for the importable rb10_control module."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rb10_control.joint_trajectory_controller import (  # noqa: E402
    BASE_LINK,
    DEBUG,
    EE_LINK,
    JOINT_NAMES,
    JOINT_STATES_TOPIC,
    JTC_TOPIC,
    MAX_STEP_L2_RAD,
    MAX_STEP_PER_JOINT_RAD,
    RB10Controller,
    TASK_STOP_SERVICE,
    URDF_PATH,
    main,
)

__all__ = [
    "BASE_LINK",
    "DEBUG",
    "EE_LINK",
    "JOINT_NAMES",
    "JOINT_STATES_TOPIC",
    "JTC_TOPIC",
    "MAX_STEP_L2_RAD",
    "MAX_STEP_PER_JOINT_RAD",
    "RB10Controller",
    "TASK_STOP_SERVICE",
    "URDF_PATH",
    "main",
]


if __name__ == "__main__":
    main()
