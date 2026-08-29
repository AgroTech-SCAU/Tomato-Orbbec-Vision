"""Unified RGB-D target service/topic node."""

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
from visualization_msgs.msg import Marker, MarkerArray
from vision_interfaces.msg import TargetArray, TargetPoint
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


class VisionNode(Node):
    """Owns camera synchronization, one YOLO model, topic output and service."""

    def __init__(self):
        super().__init__('vision_node')
        self._declare_parameters()

        self._bridge = CvBridge()
        self._snapshot_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._latest_snapshot = None
        self._latest_target_frame = None
        self._sequence = 0
        self._last_no_frame_warning_ns = 0

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

        target_topic = self.get_parameter('target_array_topic').value
        marker_topic = self.get_parameter('marker_topic').value
        self._target_pub = self.create_publisher(
            TargetArray, target_topic, qos_profile_sensor_data)
        self._marker_pub = self.create_publisher(
            MarkerArray, marker_topic, qos_profile_sensor_data)

        service_name = self.get_parameter('service_name').value
        service_group = ReentrantCallbackGroup()
        self._service = self.create_service(
            GetTargets,
            service_name,
            self._on_get_targets,
            callback_group=service_group,
        )

        publish_period_sec = float(self.get_parameter('publish_period_sec').value)
        self._timer = self.create_timer(publish_period_sec, self._publish_latest)

        self.get_logger().info(
            f'Ready: service={service_name}; targets={target_topic}; '
            f'color={color_topic}; depth={depth_topic}; info={info_topic}'
        )

    def _declare_parameters(self):
        defaults = {
            'color_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/depth/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'service_name': '/vision/get_targets',
            'target_array_topic': '/vision/targets',
            'marker_topic': '/vision/debug/markers',
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
            'publish_period_sec': 0.10,
            'publish_markers': True,
            'marker_scale_m': 0.04,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _on_synced_rgbd(self, color_msg, depth_msg, camera_info_msg):
        try:
            color = self._bridge.imgmsg_to_cv2(
                color_msg, desired_encoding='bgr8')
            depth = self._bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
            color = np.asarray(color)
            depth = np.asarray(depth)

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

    def _publish_latest(self):
        snap = self._copy_snapshot()
        if snap is None:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_no_frame_warning_ns > 2_000_000_000:
                self.get_logger().warning('Waiting for synchronized RGB-D frames')
                self._last_no_frame_warning_ns = now_ns
            return

        try:
            frame = self._infer_snapshot(snap)
            confidence = float(self.get_parameter('default_confidence').value)
            targets = self._filter_targets(frame.targets, confidence, ())

            msg = TargetArray()
            msg.header = snap.header
            msg.targets = targets
            self._target_pub.publish(msg)

            if bool(self.get_parameter('publish_markers').value):
                self._marker_pub.publish(self._make_markers(msg))
        except Exception:
            self.get_logger().exception('Periodic inference/publish failed')

    def _make_markers(self, target_array: TargetArray) -> MarkerArray:
        markers = MarkerArray()

        clear = Marker()
        clear.header = target_array.header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        scale = float(self.get_parameter('marker_scale_m').value)
        for index, target in enumerate(target_array.targets):
            sphere = Marker()
            sphere.header = target_array.header
            sphere.ns = 'vision_targets'
            sphere.id = index
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = target.position
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = scale
            sphere.scale.y = scale
            sphere.scale.z = scale
            sphere.color.r = 1.0
            sphere.color.g = 0.2
            sphere.color.b = 0.1
            sphere.color.a = 0.9
            markers.markers.append(sphere)

            text = Marker()
            text.header = target_array.header
            text.ns = 'vision_target_labels'
            text.id = index
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = target.position
            text.pose.position.z += scale
            text.pose.orientation.w = 1.0
            text.scale.z = max(scale * 0.6, 0.02)
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = (
                f'{target.semantic_label} '
                f'{target.confidence:.2f} z={target.position.z:.3f}m')
            markers.markers.append(text)

        return markers

    @staticmethod
    def _set_error(response, code, message):
        response.success = False
        response.status_code = code
        response.message = message
        response.targets = []
        return response

    def _on_get_targets(self, request, response):
        snap = self._copy_snapshot()
        if snap is None:
            return self._set_error(
                response,
                GetTargets.Response.NO_FRAME,
                'No synchronized RGB/depth/camera_info frame is available',
            )

        response.header = snap.header
        age_sec = (
            self.get_clock().now().nanoseconds - snap.received_ns) / 1e9
        max_age = float(request.max_frame_age_sec)
        if max_age <= 0.0:
            max_age = float(
                self.get_parameter('default_max_frame_age_sec').value)
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
            self.get_logger().warning(
                f'Request min_confidence={confidence:.3f} is below '
                f'inference_confidence_floor={floor:.3f}; using {floor:.3f}')
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
                response, GetTargets.Response.INFERENCE_ERROR, str(exc))


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
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
