#!/usr/bin/env python3
#
# Copyright (c) 2024 University of Washington, All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# Neither the name of the copyright holder nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""ROS2 Node and entrypoint."""

from apriltag_msgs.msg import ApriltagDetectionArray
from blue_localization import PoseLocalizer
from bluerov_fiducial_localization.fiducial_map import FiducialMap
import cv2
from geometry_msgs.msg import PoseStamped
import numpy as np
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
import scipy.spatial.transform.Rotation as R
from sensor_msgs.msg import CameraInfo
from tf2 import TransformException


class FiducialLocalizationNode(PoseLocalizer):
    """ROS2 Node."""

    def __init__(self):
        """Initialize the node."""
        super().__init__("fiducial_localization")

        self.declare_parameter("fiducial_map_url", None)

        self.fiducial_sub = self.create_subscription(
            "/fiducials", ApriltagDetectionArray, self.fiducial_cb
        )

        self.camera_info_sub = self.create_subscription(
            "/camera_info",
            CameraInfo,
            self.camera_info_cb,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )

        fiducial_map_url = self.get_parameters("fiducial_map_url")
        self.fiducial_map = FiducialMap.load(fiducial_map_url)
        if self.fiducial_map is None:
            self.get_logger().fatal(
                f"Unable to load fiducial map from {fiducial_map_url}"
            )

        self.latest_camera_info = None

    def fiducial_cb(self, at_array: ApriltagDetectionArray) -> None:
        """Get the camera pose relative to the marker and send to the ArduSub EKF.

        Args:
            at_array: ROS2 ApriltagDetectionArray message
        """
        if self.latest_camera_info is None:
            self.get_logger().warn_once(
                "Have not received camera info, cannot calculate position"
            )
            return

        # Retain only the markers I recognize
        result = self.fiducial_map.match_detections(at_array.detections)

        if len(result.matches) < 4:
            pass

        object_points = [m.object for m in result.matches]
        image_points = [m.detection.centre for m in result.matches]

        pnp_method = cv2.SOLVEPNP_ITERATIVE
        rv = None
        tv = None
        retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            rv,
            tv,
            pnp_method,
        )

        # Isolate preferred rvec/tvec; this should be smarter
        best_rvec = rvecs[0]
        best_tvec = tvecs[0]

        # Convert the pose into a PoseStamped message
        pose = PoseStamped()

        pose.header.frame_id = "fiducial_map"
        pose.header.stamp = self.get_clock().now().to_msg()

        (
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ) = best_tvec.squeeze()

        rot_mat, _ = cv2.Rodrigues(best_rvec)

        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = R.from_matrix(rot_mat).as_quat()

        # Transform the pose from the `marker` frame to the `map` frame
        try:
            pose = self.tf_buffer.transform(pose, self.MAP_FRAME)
        except TransformException as e:
            self.get_logger().warning(
                f"Could not transform from fiducial map frame to map: {e}"
            )
            return

        # The pose now represents the transformation from the map frame to the
        # camera frame, but we need to publish the transformation from the map frame
        # to the base_link frame

        # Start by getting the camera to base_link transform
        try:
            tf_camera_to_base = self.tf_buffer.lookup_transform(
                self.CAMERA_FRAME, self.BASE_LINK_FRAME, Time()
            )
        except TransformException as e:
            self.get_logger().warning(f"Could not access transform: {e}")
            return

        # Convert the tf into a homogeneous tf matrix
        tf_camera_to_base_mat = np.eye(4)
        tf_camera_to_base_mat[:3, :3] = R.from_quat(
            [
                tf_camera_to_base.transform.rotation.x,
                tf_camera_to_base.transform.rotation.y,
                tf_camera_to_base.transform.rotation.z,
                tf_camera_to_base.transform.rotation.w,
            ]
        ).as_matrix()
        tf_camera_to_base_mat[:3, 3] = np.array(
            [
                tf_camera_to_base.transform.translation.x,
                tf_camera_to_base.transform.translation.y,
                tf_camera_to_base.transform.translation.z,
            ]
        )

        # Convert the pose back into a matrix
        tf_map_to_camera_mat = np.eye(4)
        tf_map_to_camera_mat[:3, :3] = R.from_quat(
            [
                pose.pose.orientation.x,  # type: ignore
                pose.pose.orientation.y,  # type: ignore
                pose.pose.orientation.z,  # type: ignore
                pose.pose.orientation.w,  # type: ignore
            ]
        ).as_matrix()
        tf_map_to_camera_mat[:3, 3] = np.array(
            [
                pose.pose.position.x,  # type: ignore
                pose.pose.position.y,  # type: ignore
                pose.pose.position.z,  # type: ignore
            ]
        )

        # Calculate the new transform
        tf_map_to_base_mat = tf_camera_to_base_mat @ tf_map_to_camera_mat

        # Update the pose using the new transform
        (
            pose.pose.position.x,  # type: ignore
            pose.pose.position.y,  # type: ignore
            pose.pose.position.z,  # type: ignore
        ) = tf_map_to_base_mat[3:, 3]

        (
            pose.pose.orientation.x,  # type: ignore
            pose.pose.orientation.y,  # type: ignore
            pose.pose.orientation.z,  # type: ignore
            pose.pose.orientation.w,  # type: ignore
        ) = R.from_matrix(tf_map_to_base_mat[:3, :3]).as_quat()

        self.state = pose

    def camera_info_cb(self, camera_info: CameraInfo) -> None:
        """Store the latest camera_info.

        Args:
            camera_info: A camera_info message.
        """
        self.latest_camera_info = camera_info

        self.camera_matrix = camera_info.K.reshape(3, 3)
        self.dist_coeffs = camera_info.D
