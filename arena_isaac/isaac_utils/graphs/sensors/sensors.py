import os
import xml.etree.ElementTree as ET

import carb
import omni.graph.core as og
import omni.usd
from isaac_utils.utils.geom import Rotation, Translation
from isaac_utils.utils.path import world_path

from .camera import SensorCamera, SensorCameraRGBD
from .contact import SensorContact
from .imu import SensorIMU
from .lidar import SensorLidar
from arena_isaac.services.ResetSensors import register_sensors


class Sensors:
    def __init__(
        self,
        prim_path: str,
        base_frame: str,
        base_topic: str,
    ):
        self.prim_path: str = prim_path
        self.robot_base_frame: str = base_frame
        self.robot_base_topic: str = base_topic

        # Keep references so reset_lidars() can rebuild the OmniGraph pipeline.
        self._lidars: list[SensorLidar] = []

    def parse_gazebo(self, urdf: str):
        """
        Parses the URDF string as an XML file and finds all <sensor> tags.
        Processes each sensor based on its type attribute.
        Args:
            urdf(str): The URDF string.
        """

        root = ET.fromstring(urdf)

        for gazebo in root.findall('.//gazebo'):
            reference = gazebo.get('reference')
            if reference is None:
                continue

            for sensor in gazebo.findall('.//sensor'):
                try:
                    sensor_type = sensor.get('type')
                    sensor_name = sensor.get('name')

                    if sensor_type is None:
                        continue
                    if sensor_name is None:
                        continue

                    pose = list(map(float, sensor.findtext('./pose', '0 0 0 0 0 0').split(' ')))
                    translation = Translation.parse(pose[:3])
                    rotation = Rotation.parse(pose[3:])

                    if sensor_type == 'gpu_lidar':
                        lidar = SensorLidar(
                            robot_base_frame=self.robot_base_frame,
                            parent_frame=reference,
                            name=sensor_name,
                            config=SensorLidar.Config.parse(sensor),
                            translation=translation,
                            rotation=rotation,
                        )
                        lidar.simulate(self.prim_path)
                        lidar.publish(self.robot_base_topic)
                        # Store reference for later resets.
                        self._lidars.append(lidar)

                    elif sensor_type == "imu":
                        imu = SensorIMU(
                            robot_base_frame=self.robot_base_frame,
                            config=SensorIMU.Config.parse(sensor),
                            name=sensor_name,
                            parent_frame=reference,
                        )
                        imu.simulate(self.prim_path)
                        imu.publish(self.robot_base_topic)

                    elif sensor_type == 'contact':
                        contact = SensorContact(
                            robot_base_frame=self.robot_base_frame,
                            config=SensorContact.Config.parse(sensor),
                            name=sensor_name,
                        )
                        contact.simulate(self.prim_path)
                        contact.publish(self.robot_base_topic)

                    elif sensor_type == 'camera':
                        camera = SensorCamera(
                            robot_base_frame=self.robot_base_frame,
                            parent_frame=reference,
                            config=SensorCamera.Config.parse(sensor),
                            name=sensor_name,
                            translation=translation,
                            rotation=rotation,
                        )
                        camera.simulate(self.prim_path)
                        camera.publish(self.robot_base_topic)

                    elif sensor_type == 'rgbd_camera':
                        camera = SensorCameraRGBD(
                            robot_base_frame=self.robot_base_frame,
                            parent_frame=reference,
                            config=SensorCamera.Config.parse(sensor),
                            name=sensor_name,
                            translation=translation,
                            rotation=rotation,
                        )
                        camera.simulate(self.prim_path)
                        camera.publish(self.robot_base_topic)

                except Exception as e:
                    raise

        # Register this Sensors instance so ResetSensors service can find it
        # by robot prim path and call reset_lidars() directly (fast path).
        if self._lidars:
            register_sensors(self.prim_path, self)
            carb.log_warn(
                f"[Sensors] Registered sensor manager for {self.prim_path} "
                f"with {len(self._lidars)} lidar(s)"
            )

    def reset_lidars(self) -> None:
        """
        Destroy and rebuild the OmniGraph LidarPublisher pipeline for every
        lidar owned by this Sensors instance.

        Called by ResetSensors service after robot teleportation.  Rebuilding
        the graph (rather than patching it in-place) guarantees that
        IsaacCreateRenderProduct acquires a fresh prim handle with the current
        world transform, which is the only reliable approach while the
        simulation is paused.
        """
        stage = omni.usd.get_context().get_stage()

        for lidar in self._lidars:
            if lidar.prim_path is None:
                carb.log_warn(
                    f"[Sensors] reset_lidars: lidar {lidar.name} has no prim_path, skipping"
                )
                continue

            graph_path = os.path.join(lidar.prim_path, "LidarPublisher")

            # --- Tear down: delete the existing OmniGraph prim ---
            graph_prim = stage.GetPrimAtPath(graph_path)
            if graph_prim.IsValid():
                carb.log_warn(f"[Sensors] reset_lidars: deleting graph at {graph_path}")
                try:
                    # The proper way to delete a graph/prim in Isaac Sim USD
                    stage.RemovePrim(graph_path)
                except Exception as e:
                    carb.log_error(
                        f"[Sensors] reset_lidars: failed to delete graph {graph_path}: {e}"
                    )

            # --- Rebuild: re-run publish() which rewires the full graph ---
            try:
                carb.log_warn(
                    f"[Sensors] reset_lidars: rebuilding graph for {lidar.prim_path}"
                )
                lidar.publish(self.robot_base_topic)
                carb.log_warn(
                    f"[Sensors] reset_lidars: rebuilt graph for {lidar.prim_path} OK"
                )
            except Exception as e:
                carb.log_error(
                    f"[Sensors] reset_lidars: failed to rebuild graph "
                    f"for {lidar.prim_path}: {e}"
                )