import threading
import queue
from dataclasses import dataclass
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from ultralytics import YOLO
from vision_interfaces.msg import TargetPoint
from vision_interfaces.srv import GetTargets
import cv2


@dataclass(frozen=True)
class Snapshot:
    color: np.ndarray
    depth: np.ndarray
    camera_info: CameraInfo
    header: object
    received_ns: int


class VisionPublisher(Node):
    def __init__(self):
        super().__init__('vision_publisher')
        self._declare_parameters()

        self.bridge = CvBridge()
        self.snapshot_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.latest_snapshot = None
        self.latest_camera_info = None

        # ========== 新增：GUI 队列和线程 ==========
        self.gui_queue = queue.Queue(maxsize=1)  # 只保留最新图像
        self.gui_running = True
        self.gui_thread = threading.Thread(target=self._gui_loop, daemon=True)
        self.gui_thread.start()
        self.get_logger().info('GUI 线程已启动')
        # ========== 新增结束 ==========

        # 加载模型
        model_path = self.get_parameter('model_path').value
        self.device = self.get_parameter('device').value
        self.get_logger().info(f'Loading model: {model_path}')
        self.model = YOLO(model_path)

        # 订阅话题
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

        # 同步彩色图和深度图
        queue_size = int(self.get_parameter('sync_queue_size').value)
        slop = float(self.get_parameter('sync_slop_sec').value)
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=queue_size, slop=slop,
            allow_headerless=False)
        self.sync.registerCallback(self._on_synced_images)

        # 创建发布者（实时输出目标点）
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.target_pub = self.create_publisher(
            TargetPoint, '/vision/targets', qos)
        self.diagnostic_pub = self.create_publisher(
            String, '/vision/diagnostic', qos)

        # 定时器
        rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(rate, self._publish_targets)

        self.get_logger().info(
            f'Ready: publishing to /vision/targets every {rate}s; '
            f'color={color_topic}; depth={depth_topic}')

    def _declare_parameters(self):
        defaults = {
            'color_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/depth/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'model_path': '/home/ubuntu/orbbec_ros2_ws/best.pt',
            'device': '0',
            'default_confidence': 0.40,
            'sync_queue_size': 10,
            'sync_slop_sec': 0.05,
            'min_depth_m': 0.25,
            'max_depth_m': 6.0,
            'max_depth_mad_m': 0.10,
            'sample_radius_px': 7,
            'publish_rate': 0.1,
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

    # ========== 新增：GUI 线程方法 ==========
    def _gui_loop(self):
        self.get_logger().info('GUI 循环开始')
        while self.gui_running and rclpy.ok():
            try:
                # 阻塞等待，直到有图像
                annotated = self.gui_queue.get(timeout=0.1)
                if annotated is not None:
                    cv2.imshow('Vision Detection', annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.get_logger().info('按 Q 退出 GUI')
                        break
            except queue.Empty:
                pass
            except Exception as e:
                self.get_logger().error(f'GUI loop error: {e}')
        cv2.destroyAllWindows()
        self.get_logger().info('GUI 循环结束')
    # ========== 新增结束 ==========

    def _publish_targets(self):
        """定时发布目标点"""
        snap = self._copy_snapshot()
        if snap is None:
            self.get_logger().warn('snap is None，没有收到同步图像')
            return

        confidence = float(self.get_parameter('default_confidence').value)
        targets = []

        try:
            depth_m = self._depth_to_meters(snap.depth)

            with self.model_lock:
                result = self.model.predict(
                    source=snap.color, conf=confidence,
                    device=self.device, verbose=False)[0]

            # ========== 实时显示检测结果 ==========
            annotated = result.plot()
            
            if result.boxes is not None:
                target_count = len(result.boxes)
            else:
                target_count = 0
            
            cv2.putText(annotated, f"Targets: {target_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Conf: {confidence:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # ========== 修改：放入队列，由 GUI 线程显示 ==========
            try:
                self.gui_queue.put_nowait(annotated)
            except queue.Full:
                # 队列已满，丢弃旧图像
                pass
            # ========== 修改结束 ==========

            if result.boxes is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()
                names = result.names

                h, w = snap.color.shape[:2]

                for box, class_id, score in zip(xyxy, classes, scores):
                    label = str(names[int(class_id)])

                    xmin, ymin, xmax, ymax = [int(round(x)) for x in box]
                    xmin, xmax = max(0, xmin), min(w - 1, xmax)
                    ymin, ymax = max(0, ymin), min(h - 1, ymax)

                    u, v = (xmin + xmax) // 2, (ymin + ymax) // 2

                    sampled = self._sample_depth(depth_m, u, v)
                    if sampled is None:
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

            # 发布目标点
            for target in targets:
                self.target_pub.publish(target)

            self.get_logger().debug(f'Published {len(targets)} targets')

        except Exception as exc:
            self.get_logger().error(f'Publish targets failed: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = VisionPublisher()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        # 停止 GUI 线程
        node.gui_running = False
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()