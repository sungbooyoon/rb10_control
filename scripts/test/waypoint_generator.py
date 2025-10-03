# waypoint_generators.py
from typing import List, Tuple, Callable, Dict
from geometry_msgs.msg import Pose
import math

Quat = Tuple[float, float, float, float]
Vec3 = Tuple[float, float, float]

def circle_waypoints(center: Vec3,
                     radius: float,
                     n: int,
                     plane: str,
                     quat_xyzw: Quat) -> List[Pose]:
    if n <= 0:
        raise ValueError("n must be > 0")
    wps: List[Pose] = []
    for i in range(n):
        th = 2.0 * math.pi * i / n
        p = Pose()
        if plane == "xy":
            p.position.x = center[0] + radius * math.cos(th)
            p.position.y = center[1] + radius * math.sin(th)
            p.position.z = center[2]
        elif plane == "yz":
            p.position.x = center[0]
            p.position.y = center[1] + radius * math.cos(th)
            p.position.z = center[2] + radius * math.sin(th)
        elif plane == "xz":
            p.position.x = center[0] + radius * math.cos(th)
            p.position.y = center[1]
            p.position.z = center[2] + radius * math.sin(th)
        else:
            raise ValueError("plane must be 'xy', 'yz', or 'xz'")
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat_xyzw
        wps.append(p)
    return wps

def line_waypoints(start: Vec3, end: Vec3, n: int, quat_xyzw: Quat) -> List[Pose]:
    wps: List[Pose] = []
    if n <= 0:
        raise ValueError("n must be > 0")
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        p = Pose()
        p.position.x = (1 - t) * start[0] + t * end[0]
        p.position.y = (1 - t) * start[1] + t * end[1]
        p.position.z = (1 - t) * start[2] + t * end[2]
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat_xyzw
        wps.append(p)
    return wps

def rectangle_loop_waypoints(center: Vec3, size: Vec3, per_edge_points: int,
                             plane: str, quat_xyzw: Quat) -> List[Pose]:
    # ... (네 코드 그대로 유지)
    # [생략: 너가 올린 구현과 동일]
    ...
    
GENERATOR_REGISTRY: Dict[str, Callable] = {
    "circle": circle_waypoints,
    "line": line_waypoints,
    "rectangle": rectangle_loop_waypoints,
}
