import omni.graph.core as og
import omni.usd
from isaacsim_msgs.srv import DeletePrims
from typing import Any
import os
import carb

from .utils import Service

_SENSOR_REGISTRY: dict[str, Any] = {}


def register_sensors(robot_prim_path: str, sensor_manager: Any) -> None:
    """Register per-robot sensor manager for robust reset handling.

    Call this once after Sensors.parse_gazebo() completes, passing the robot's
    prim path and the Sensors instance.  The ResetSensors service will then
    delegate to sensor_manager.reset_lidars() (fast path) instead of doing a
    full stage traversal (slow path).
    """
    _SENSOR_REGISTRY[str(robot_prim_path)] = sensor_manager
    carb.log_warn(f"[ResetSensors] Registered sensor manager for {robot_prim_path}")


def _reset_lidar_graph(stage, robot_prim_path: str) -> bool:
    """
    Slow-path fallback: traverse the stage and attempt to re-seat the
    cameraPrim input on every get_camera_prim node under robot_prim_path.

    This is used only when no Sensors manager was registered (i.e. the robot
    was spawned without going through Sensors.parse_gazebo).

    NOTE: This approach is inherently unreliable while the sim is paused
    because evaluate_sync does not fire OnPlaybackTick.  Prefer the fast path
    (register_sensors + Sensors.reset_lidars) whenever possible.
    """
    robot_root = str(os.path.dirname(robot_prim_path))
    found_any = False

    for prim in stage.Traverse():
        path_str = str(prim.GetPath())

        if robot_root not in path_str:
            continue
        if 'LidarPublisher' not in path_str:
            continue
        if not path_str.endswith('/get_camera_prim'):
            continue

        node_handle = og.Controller.node(path_str)
        if not node_handle.is_valid():
            carb.log_warn(f"[ResetSensors] Node not valid, skipping: {path_str}")
            continue

        lidar_prim_path = path_str.split('/LidarPublisher/')[0]

        carb.log_warn(
            f"[ResetSensors] Slow-path: re-seating get_camera_prim at {path_str} "
            f"-> {lidar_prim_path}"
        )

        try:
            attr = og.Controller.attribute(f"{path_str}.inputs:paths")

            # Clear first so OmniGraph registers an actual value change,
            # defeating its diff-and-skip optimisation.
            attr.set([])
            attr.set([lidar_prim_path])

            # Toggle render_product enabled to queue a recreation on next tick.
            rp_path = path_str.replace('/get_camera_prim', '/render_product')
            rp_handle = og.Controller.node(rp_path)
            if rp_handle.is_valid():
                og.Controller.attribute(f"{rp_path}.inputs:enabled").set(False)
                og.Controller.attribute(f"{rp_path}.inputs:enabled").set(True)

            # Best-effort synchronous evaluation — may be a no-op while paused.
            graph = node_handle.get_graph()
            if graph.is_valid():
                og.Controller.evaluate_sync(graph)

            found_any = True
        except Exception as e:
            carb.log_error(f"[ResetSensors] Failed to reset lidar graph at {path_str}: {e}")

    return found_any


def reset_sensors_callback(
    request: DeletePrims.Request,
    response: DeletePrims.Response,
):
    stage = omni.usd.get_context().get_stage()

    # Diagnostic: dump all LidarPublisher paths present in the stage.
    carb.log_warn("[ResetSensors] === Stage LidarPublisher paths ===")
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if 'LidarPublisher' in p:
            carb.log_warn(f"[ResetSensors]   FOUND: {p}")
    carb.log_warn(f"[ResetSensors] === Request paths: {list(request.names)} ===")

    results = []

    for robot_prim_path in request.names:
        success = False
        try:
            manager = _SENSOR_REGISTRY.get(str(robot_prim_path))

            if manager is not None and hasattr(manager, "reset_lidars"):
                # Fast path: delegate to Sensors.reset_lidars() which tears
                # down and rebuilds the OmniGraph pipeline from scratch.
                carb.log_warn(
                    f"[ResetSensors] Fast path: using registered sensor manager "
                    f"for {robot_prim_path}"
                )
                manager.reset_lidars()
                success = True
            else:
                # Slow path: in-place graph patch via stage traversal.
                carb.log_warn(
                    f"[ResetSensors] Slow path: no registered manager for "
                    f"{robot_prim_path}, falling back to stage traversal"
                )
                success = _reset_lidar_graph(stage, robot_prim_path)

                if not success:
                    carb.log_warn(
                        f"[ResetSensors] No LidarPublisher/get_camera_prim found "
                        f"under {robot_prim_path}. "
                        f"Is the robot spawned and sensors initialized?"
                    )

        except Exception as e:
            carb.log_error(f"[ResetSensors] Failed for {robot_prim_path}: {e}")
            success = False

        results.append(success)

    response.ret = results
    return response


reset_sensors_service = Service(
    srv_type=DeletePrims,
    srv_name='isaac/ResetSensors',
    callback=reset_sensors_callback
)

__all__ = ['reset_sensors_service', 'register_sensors']
