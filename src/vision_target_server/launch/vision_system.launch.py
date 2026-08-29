import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    orbbec_share = get_package_share_directory('orbbec_camera')
    vision_share = get_package_share_directory('vision_target_server')

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(orbbec_share, 'launch', 'gemini_330_series.launch.py')
        ),
        launch_arguments={
            'camera_name': 'camera',
            'enable_color': 'true',
            'enable_depth': 'true',
            'color_width': '1280',
            'color_height': '800',
            'color_fps': '30',
            'depth_width': '1280',
            'depth_height': '800',
            'depth_fps': '30',
            'depth_registration': 'true',
            'align_mode': 'SW',
            'align_target_stream': 'COLOR',
            'enable_frame_sync': 'true',
            'depth_precision': '1mm',
            'enable_point_cloud': 'true',
            'enable_colored_point_cloud': 'false',
        }.items(),
    )

    vision = Node(
        package='vision_target_server',
        executable='vision_node',
        name='vision_node',
        output='screen',
        parameters=[os.path.join(vision_share, 'config', 'vision.yaml')],
    )

    return LaunchDescription([camera, vision])
