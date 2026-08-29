"""Compatibility launch: starts the unified vision node without the camera."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    vision_share = get_package_share_directory('vision_target_server')
    return LaunchDescription([
        Node(
            package='vision_target_server',
            executable='vision_node',
            name='vision_node',
            output='screen',
            parameters=[os.path.join(vision_share, 'config', 'vision.yaml')],
        )
    ])
