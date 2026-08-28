import threading
from dataclasses import dataclass
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
from ultralytics import YOLO
from vision_interfaces.msg import TargetPoint
from vision_interfaces.srv import GetTargets


@dataclass(frozen=True)
class Snapshot:
    color: np.ndarray
    depth: np.ndarray
    camera_info: CameraInfo
    header: object
    received_ns: int


class VisionServer(Node):
    def __init__(self):
        super().__init__('vision_server')
        self._declare_parameters()

        self.bridge = CvBridge()
        self.snapshot_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.latest_snapshot = None
        self.latest_camera_info = None

        model_path = self.get_parameter('model_path').value
        self.device = self.get_parameter('device').value
        self.get_logger().info(f'Loading model: {model_path}')
        self.model = YOLO(model_path)

        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value

        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self._on_camera_info,
            qos_profile_sensor_data)

        self.color_sub = Subscriber(self, Image, color_topic,
                                    qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(
            self, Image, depth_topic,
            qos_profile=qos_profile_sensor_data)

        queue_size = int(self.get_parameter('sync_queue_size').value)
        slop = float(self.get_parameter('sync_slop_sec').value)
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=queue_size, slop=slop,
            allow_headerless=False)
        self.sync.registerCallback(self._on_synced_images)

        service_group = ReentrantCallbackGroup()
        service_name = self.get_parameter('service_name').value
        self.service = self.create_service(
            GetTargets, service_name, self._on_get_targets,
            callback_group=service_group)

        self.get_logger().info(
            f'Ready: {service_name}; color={color_topic}; depth={depth_topic}')

    def _declare_parameters(self):
        defaults = {
            'color_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/depth/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'service_name': '/vision/get_targets',
            'model_path': 'yolov8n.pt',
            'device': 'cpu',
            'default_confidence': 0.40,
            'default_max_frame_age_sec': 0.50,
            'sync_queue_size': 10,
            'sync_slop_sec': 0.05,
            'min_depth_m': 0.25,
            'max_depth_m': 6.0,
            'max_depth_mad_m': 0.10,
            'sample_radius_px': 7,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _on_camera_info(self, msg):
        with self.snapshot_lock:
            self.latest_camera_info = msg

    def _on_synced_images(self, color_msg, depth_msg):
        try:
            color = self.bridge.imgmsg_to_cv2(
                color_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
            depth = np.asarray(depth)

            with self.snapshot_lock:
                info = self.latest_camera_info
                if info is None:
                    return
                self.latest_snapshot = Snapshot(
                    color=color.copy(), depth=depth.copy(),
                    camera_info=info, header=color_msg.header,
                    received_ns=self.get_clock().now().nanoseconds)
        except Exception as exc:
            self.get_logger().error(f'Image conversion failed: {exc}')

    def _copy_snapshot(self):
        with self.snapshot_lock:
            snap = self.latest_snapshot
            if snap is None:
                return None
            return Snapshot(
                color=snap.color.copy(), depth=snap.depth.copy(),
                camera_info=snap.camera_info, header=snap.header,
                received_ns=snap.received_ns)

    def _depth_to_meters(self, depth):
        if depth.dtype == np.uint16:
            # Camera launch explicitly sets depth_precision:=1mm.
            return depth.astype(np.float32) * 0.001
        if depth.dtype in (np.float32, np.float64):
            return depth.astype(np.float32)
        raise ValueError(f'Unsupported depth dtype: {depth.dtype}')

    def _sample_depth(self, depth_m, u, v):
        radius = int(self.get_parameter('sample_radius_px').value)
        min_d = float(self.get_parameter('min_depth_m').value)
        max_d = float(self.get_parameter('max_depth_m').value)

        h, w = depth_m.shape[:2]
        x0, x1 = max(0, u - radius), min(w, u + radius + 1)
        y0, y1 = max(0, v - radius), min(h, v + radius + 1)

        roi = depth_m[y0:y1, x0:x1]
        valid = roi[np.isfinite(roi) & (roi >= min_d) & (roi <= max_d)]
        if valid.size < 5:
            return None

        z = float(np.median(valid))
        mad = float(np.median(np.abs(valid - z)))
        max_mad = float(self.get_parameter('max_depth_mad_m').value)
        if mad > max_mad:
            return None
        return z, mad

    @staticmethod
    def _project(u, v, z, camera_info):
        fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
        cx, cy = float(camera_info.k[2]), float(camera_info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError('Invalid camera intrinsics')
        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy
        return x, y, z

    def _set_error(self, response, code, message):
        response.success = False
        response.status_code = code
        response.message = message
        response.targets = []
        return response

    def _on_get_targets(self, request, response):
        snap = self._copy_snapshot()
        if snap is None:
            return self._set_error(
                response, GetTargets.Response.NO_FRAME,
                'No synchronized RGB/depth frame is available')

        response.header = snap.header
        age = (self.get_clock().now().nanoseconds - snap.received_ns) / 1e9
        max_age = float(request.max_frame_age_sec)
        if max_age <= 0.0:
            max_age = float(self.get_parameter(
                'default_max_frame_age_sec').value)

        if age > max_age:
            return self._set_error(
                response, GetTargets.Response.STALE_FRAME,
                f'Latest synchronized frame is stale: {age:.3f}s')

        if snap.color.shape[:2] != snap.depth.shape[:2]:
            return self._set_error(
                response, GetTargets.Response.BAD_FRAME,
                f'RGB/depth size mismatch: '
                f'{snap.color.shape[:2]} vs {snap.depth.shape[:2]}')

        color_h, color_w = snap.color.shape[:2]
        if (int(snap.camera_info.width) != color_w or
                int(snap.camera_info.height) != color_h):
            return self._set_error(
                response, GetTargets.Response.BAD_FRAME,
                f'RGB/CameraInfo size mismatch: {color_w}x{color_h} vs '
                f'{snap.camera_info.width}x{snap.camera_info.height}')

        confidence = float(request.min_confidence)
        if confidence <= 0.0:
            confidence = float(self.get_parameter(
                'default_confidence').value)

        wanted = {x.strip().lower() for x in request.class_filter if x.strip()}

        try:
            depth_m = self._depth_to_meters(snap.depth)

            with self.model_lock:
                result = self.model.predict(
                    source=snap.color, conf=confidence,
                    device=self.device, verbose=False)[0]

            targets = []
            skipped_depth = 0

            if result.boxes is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()
                names = result.names

                h, w = snap.color.shape[:2]

                for box, class_id, score in zip(xyxy, classes, scores):
                    label = str(names[int(class_id)])
                    if wanted and label.lower() not in wanted:
                        continue

                    xmin, ymin, xmax, ymax = [int(round(x)) for x in box]
                    xmin, xmax = max(0, xmin), min(w - 1, xmax)
                    ymin, ymax = max(0, ymin), min(h - 1, ymax)

                    u, v = (xmin + xmax) // 2, (ymin + ymax) // 2

                    sampled = self._sample_depth(depth_m, u, v)
                    if sampled is None:
                        skipped_depth += 1
                        continue

                    z, mad = sampled
                    x, y, z = self._project(u, v, z, snap.camera_info)

                    target = TargetPoint()
                    target.semantic_label = label
                    target.class_id = int(class_id)
                    target.confidence = float(score)
                    target.position = Point(x=x, y=y, z=z)
                    target.bbox_xyxy = [xmin, ymin, xmax, ymax]
                    target.pixel_u = u
                    target.pixel_v = v
                    target.depth_mad_m = mad
                    targets.append(target)

            response.success = True
            response.status_code = GetTargets.Response.OK
            response.targets = targets
            response.message = (
                f'OK: {len(targets)} target(s); '
                f'{skipped_depth} detection(s) skipped for invalid depth')
            return response

        except Exception as exc:
            self.get_logger().exception('Inference request failed')
            return self._set_error(
                response, GetTargets.Response.INFERENCE_ERROR, str(exc))


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