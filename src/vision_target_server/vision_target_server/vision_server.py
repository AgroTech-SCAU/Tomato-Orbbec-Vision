"""ROS 2 RGB-D target service.

The node continuously caches synchronized RGB/depth/camera-info frames.
YOLO inference is executed only when /vision/get_targets is requested.
"""

from dataclasses import dataclass
import threading

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from vision_interfaces.msg import TargetPoint
from vision_interfaces.srv import GetTargets

from .detector import YoloDetector
from .rgbd_projector import depth_to_meters, project_pixel, sample_depth


@dataclass(frozen=True)
class Snapshot:
    """One synchronized RGB-D-camera-info snapshot."""

    sequence: int
    color: np.ndarray
    depth: np.ndarray
    camera_info: CameraInfo
    header: object
    received_ns: int


@dataclass(frozen=True)
class TargetFrame:
    """Projected detections for one synchronized camera frame."""

    sequence: int
    targets: tuple[TargetPoint, ...]
    skipped_depth: int


class VisionServer(Node):
    """Cache RGB-D frames and expose 3D detections through one ROS 2 service."""

    def __init__(self):
        super().__init__('vision_server')
        self._declare_parameters()

        self._bridge = CvBridge()
        self._snapshot_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._latest_snapshot = None
        self._latest_target_frame = None
        self._sequence = 0

        model_path = self.get_parameter('model_path').value
        device = self.get_parameter('device').value
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self._detector = YoloDetector(model_path, device)

        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value

        self._color_sub = Subscriber(
            self, Image, color_topic, qos_profile=qos_profile_sensor_data)
        self._depth_sub = Subscriber(
            self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self._info_sub = Subscriber(
            self, CameraInfo, info_topic, qos_profile=qos_profile_sensor_data)

        queue_size = int(self.get_parameter('sync_queue_size').value)
        slop = float(self.get_parameter('sync_slop_sec').value)
        self._sync = ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub, self._info_sub],
            queue_size=queue_size,
            slop=slop,
            allow_headerless=False,
        )
        self._sync.registerCallback(self._on_synced_rgbd)

        service_name = self.get_parameter('service_name').value
        self._service = self.create_service(
            GetTargets,
            service_name,
            self._on_get_targets,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info(
            f'Ready: service={service_name}; '
            f'color={color_topic}; depth={depth_topic}; info={info_topic}'
        )

    def _declare_parameters(self):
        defaults = {
            'color_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/depth/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'service_name': '/vision/get_targets',
            'model_path': 'yolov8n.pt',
            'device': 'cpu',
            'inference_confidence_floor': 0.10,
            'default_confidence': 0.40,
            'default_max_frame_age_sec': 0.50,
            'sync_queue_size': 10,
            'sync_slop_sec': 0.05,
            'depth_uint16_scale_m': 0.001,
            'min_depth_m': 0.25,
            'max_depth_m': 6.0,
            'max_depth_mad_m': 0.10,
            'min_depth_valid_ratio': 0.20,
            'min_depth_valid_samples': 5,
            'sample_radius_px': 7,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _on_synced_rgbd(self, color_msg, depth_msg, camera_info_msg):
        """Cache the newest synchronized RGB-D frame without running YOLO."""
        try:
            color = np.asarray(self._bridge.imgmsg_to_cv2(
                color_msg, desired_encoding='bgr8'))
            depth = np.asarray(self._bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough'))

            if color.shape[:2] != depth.shape[:2]:
                self.get_logger().warning(
                    f'RGB/depth size mismatch: {color.shape[:2]} vs '
                    f'{depth.shape[:2]}')
                return

            height, width = color.shape[:2]
            if (int(camera_info_msg.width) != width or
                    int(camera_info_msg.height) != height):
                self.get_logger().warning(
                    f'RGB/CameraInfo size mismatch: {width}x{height} vs '
                    f'{camera_info_msg.width}x{camera_info_msg.height}')
                return

            with self._snapshot_lock:
                self._sequence += 1
                self._latest_snapshot = Snapshot(
                    sequence=self._sequence,
                    color=color.copy(),
                    depth=depth.copy(),
                    camera_info=camera_info_msg,
                    header=color_msg.header,
                    received_ns=self.get_clock().now().nanoseconds,
                )
        except Exception as exc:
            self.get_logger().error(f'RGB-D conversion failed: {exc}')

    def _copy_snapshot(self):
        with self._snapshot_lock:
            snap = self._latest_snapshot
            if snap is None:
                return None
            return Snapshot(
                sequence=snap.sequence,
                color=snap.color.copy(),
                depth=snap.depth.copy(),
                camera_info=snap.camera_info,
                header=snap.header,
                received_ns=snap.received_ns,
            )

    def _infer_snapshot(self, snap: Snapshot) -> TargetFrame:
        """Run YOLO and RGB-D projection once for a camera snapshot."""
        cached = self._latest_target_frame
        if cached is not None and cached.sequence == snap.sequence:
            return cached

        with self._inference_lock:
            cached = self._latest_target_frame
            if cached is not None and cached.sequence == snap.sequence:
                return cached

            confidence_floor = float(
                self.get_parameter('inference_confidence_floor').value)
            detections = self._detector.predict(
                snap.color, confidence_floor=confidence_floor)

            depth_m = depth_to_meters(
                snap.depth,
                uint16_scale_m=float(
                    self.get_parameter('depth_uint16_scale_m').value),
            )

            radius = int(self.get_parameter('sample_radius_px').value)
            min_depth = float(self.get_parameter('min_depth_m').value)
            max_depth = float(self.get_parameter('max_depth_m').value)
            max_mad = float(self.get_parameter('max_depth_mad_m').value)
            min_samples = int(
                self.get_parameter('min_depth_valid_samples').value)
            min_ratio = float(
                self.get_parameter('min_depth_valid_ratio').value)

            targets: list[TargetPoint] = []
            skipped_depth = 0
            for detection in detections:
                xmin, ymin, xmax, ymax = detection.bbox_xyxy
                u = (xmin + xmax) // 2
                v = (ymin + ymax) // 2

                sampled = sample_depth(
                    depth_m,
                    u,
                    v,
                    radius_px=radius,
                    min_depth_m=min_depth,
                    max_depth_m=max_depth,
                    max_mad_m=max_mad,
                    min_valid_samples=min_samples,
                    min_valid_ratio=min_ratio,
                )
                if sampled is None:
                    skipped_depth += 1
                    continue

                x, y, z = project_pixel(
                    u, v, sampled.depth_m, snap.camera_info)

                target = TargetPoint()
                target.semantic_label = detection.semantic_label
                target.class_id = detection.class_id
                target.confidence = detection.confidence
                target.position = Point(x=x, y=y, z=z)
                target.bbox_xyxy = [xmin, ymin, xmax, ymax]
                target.pixel_u = u
                target.pixel_v = v
                target.depth_mad_m = sampled.mad_m
                targets.append(target)

            frame = TargetFrame(
                sequence=snap.sequence,
                targets=tuple(targets),
                skipped_depth=skipped_depth,
            )
            self._latest_target_frame = frame
            return frame

    @staticmethod
    def _filter_targets(targets, min_confidence: float, class_filter):
        wanted = {name.strip().lower() for name in class_filter if name.strip()}
        return [
            target for target in targets
            if float(target.confidence) >= min_confidence
            and (not wanted or target.semantic_label.lower() in wanted)
        ]

    @staticmethod
    def _set_error(response, code, message):
        response.success = False
        response.status_code = code
        response.message = message
        response.targets = []
        return response

    def _on_get_targets(self, request, response):
        """Return all valid 3D targets from the newest synchronized frame."""
        snap = self._copy_snapshot()
        if snap is None:
            return self._set_error(
                response,
                GetTargets.Response.NO_FRAME,
                'No synchronized RGB/depth/camera_info frame is available',
            )

        response.header = snap.header

        max_age = float(request.max_frame_age_sec)
        if max_age <= 0.0:
            max_age = float(
                self.get_parameter('default_max_frame_age_sec').value)

        age_sec = (
            self.get_clock().now().nanoseconds - snap.received_ns) / 1e9
        if age_sec > max_age:
            return self._set_error(
                response,
                GetTargets.Response.STALE_FRAME,
                f'Latest synchronized frame is stale: {age_sec:.3f}s',
            )

        confidence = float(request.min_confidence)
        if confidence <= 0.0:
            confidence = float(self.get_parameter('default_confidence').value)

        floor = float(self.get_parameter('inference_confidence_floor').value)
        if confidence < floor:
            confidence = floor

        try:
            frame = self._infer_snapshot(snap)
            targets = self._filter_targets(
                frame.targets, confidence, request.class_filter)

            response.success = True
            response.status_code = GetTargets.Response.OK
            response.targets = targets
            response.message = (
                f'OK: {len(targets)} target(s); '
                f'{frame.skipped_depth} detection(s) skipped for invalid depth')
            return response
        except Exception as exc:
            self.get_logger().exception('Inference request failed')
            return self._set_error(
                response,
                GetTargets.Response.INFERENCE_ERROR,
                str(exc),
            )


def main(args=None):
    rclpy.init(args=args)
    node = VisionServer()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
