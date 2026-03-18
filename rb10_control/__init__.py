"""Import-friendly RB10 control helpers.

Usage:
    from rb10_control import RB10Controller
    from rb10_control import JOINT_NAMES
"""

from .joint_trajectory_controller import (
    ACTIVE_LINKS_MASK,
    BASE_LINK,
    DEBUG,
    EE_LINK,
    IK_MAX_ITER,
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
    "ACTIVE_LINKS_MASK",
    "BASE_LINK",
    "DEBUG",
    "EE_LINK",
    "IK_MAX_ITER",
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
