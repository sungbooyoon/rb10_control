# collision_utils.py
from typing import Sequence, Optional
from rclpy.node import Node

VALID_SHAPES = {"box", "sphere", "cylinder", "cone"}
VALID_ACTIONS = {"add", "remove", "move"}


def _add_collision(moveit2,
                   shape: str,
                   object_id: str,
                   position: Sequence[float],
                   quat_xyzw: Sequence[float],
                   dimensions: Sequence[float]) -> None:
    if shape == "box":
        moveit2.add_collision_box(
            id=object_id, position=position, quat_xyzw=quat_xyzw, size=dimensions
        )
    elif shape == "sphere":
        moveit2.add_collision_sphere(
            id=object_id, position=position, radius=dimensions[0]
        )
    elif shape == "cylinder":
        moveit2.add_collision_cylinder(
            id=object_id,
            position=position,
            quat_xyzw=quat_xyzw,
            height=dimensions[0],
            radius=dimensions[1],
        )
    elif shape == "cone":
        moveit2.add_collision_cone(
            id=object_id,
            position=position,
            quat_xyzw=quat_xyzw,
            height=dimensions[0],
            radius=dimensions[1],
        )
    else:
        raise ValueError(f"Unknown shape '{shape}'")


def _remove_collision(moveit2, object_id: str) -> None:
    moveit2.remove_collision_object(id=object_id)


def _move_collision(moveit2,
                    object_id: str,
                    position: Sequence[float],
                    quat_xyzw: Sequence[float]) -> None:
    moveit2.move_collision(id=object_id, position=position, quat_xyzw=quat_xyzw)


def apply_collision_from_params(node: Node,
                                moveit2,
                                object_id: Optional[str] = None) -> str:
    """
    Read 'shape', 'action', 'position', 'quat_xyzw', 'dimensions' parameters from `node`,
    apply the requested collision operation on `moveit2`, and return the object_id used.

    If `object_id` is None, it defaults to the shape name.
    """
    log = node.get_logger()

    shape = node.get_parameter("shape").get_parameter_value().string_value
    action = node.get_parameter("action").get_parameter_value().string_value
    position = node.get_parameter("position").get_parameter_value().double_array_value
    quat_xyzw = node.get_parameter("quat_xyzw").get_parameter_value().double_array_value
    dimensions = node.get_parameter("dimensions").get_parameter_value().double_array_value

    if shape not in VALID_SHAPES:
        raise ValueError(f"Unknown shape '{shape}'. Valid: {sorted(VALID_SHAPES)}")
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown action '{action}'. Valid: {sorted(VALID_ACTIONS)}")

    oid = object_id or shape

    if action == "add":
        log.info(
            f"Adding collision primitive of type '{shape}' "
            f"{{position: {list(position)}, quat_xyzw: {list(quat_xyzw)}, dimensions: {list(dimensions)}}}"
        )
        _add_collision(moveit2, shape, oid, position, quat_xyzw, dimensions)

    elif action == "remove":
        log.info(f"Removing collision primitive with ID '{oid}'")
        _remove_collision(moveit2, oid)

    elif action == "move":
        log.info(
            f"Moving collision primitive with ID '{oid}' to "
            f"{{position: {list(position)}, quat_xyzw: {list(quat_xyzw)}}}"
        )
        _move_collision(moveit2, oid, position, quat_xyzw)

    return oid
