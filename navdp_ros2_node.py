#!/usr/bin/env python3

"""
NavDP ROS2 Node
Migrated from ROS1 version for visual navigation using diffusion policy
"""

import threading
import time
import datetime
import numpy as np
import cv2
import imageio

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup

# ROS2 message imports
from sensor_msgs.msg import Image, CameraInfo  # CompressedImage
from geometry_msgs.msg import PointStamped, Twist, PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import Trigger
from std_msgs.msg import Float32MultiArray, Header

# TF2 imports
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import transforms3d as tf_trans  # Replacement for tf.transformations

# CvBridge (needs ROS2 version)
try:
    from cv_bridge import CvBridge, CvBridgeError
except ImportError:
    # Fallback for ROS2 (might need different import)
    try:
        from cv_bridge import CvBridge, CvBridgeError
    except:
        raise ImportError("cv_bridge not found. Install: sudo apt install ros-humble-cv-bridge")

# Project imports
from baselines.navdp.policy_agent import NavDP_Agent
from utils_tasks.tracking_utils import MPC_Controller


class NavDPNode(Node):
    """NavDP ROS2 Node for visual navigation using diffusion policy"""

    def __init__(self):
        super().__init__('navdp_node')

        # Use reentrant callback group for thread safety
        self.callback_group = ReentrantCallbackGroup()

        # Declare parameters (ROS2 style)
        self.declare_parameter('checkpoint', '/home/engineai/models/navdp/navdp-cross-modal.ckpt')
        self.declare_parameter('batch_size', 1)
        self.declare_parameter('stop_threshold', [-0.5])
        self.declare_parameter('desired_v', 0.4)
        self.declare_parameter('v_max', 1.0)  # Default to desired_v
        self.declare_parameter('w_max', 1.0)
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('world_frame', 'odom')
        # self.declare_parameter('camera_frame', 'head_camera_optical_link')
        self.declare_parameter('camera_frame', 'base_link')
        self.declare_parameter('intrinsic', [])

        # Get parameters
        self.checkpoint = self.get_parameter('checkpoint').value
        self.batch_size = int(self.get_parameter('batch_size').value)
        self.stop_threshold = np.array(self.get_parameter('stop_threshold').value, dtype=np.float32)
        self.desired_v = float(self.get_parameter('desired_v').value)
        self.v_max = float(self.get_parameter('v_max').value)
        self.w_max = float(self.get_parameter('w_max').value)

        self.robot_frame = self.get_parameter('robot_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value

        intrinsic_list = self.get_parameter('intrinsic').value
        if len(intrinsic_list) == 9:
            self.intrinsic = np.array(intrinsic_list, dtype=np.float32).reshape((3, 3))
        else:
            self.get_logger().warn("Intrinsic not provided or wrong size; using identity.")
            self.intrinsic = np.eye(3, dtype=np.float32)

        # Objects
        self.bridge = CvBridge()

        # TF2 setup
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.navdp_navigator = None
        self.navdp_fps_writer = None

        # Thread safety
        self._lock = threading.Lock()
        self.latest_image = None
        self.latest_depth = None
        self.latest_goal = None
        self.last_processed_ts = 0.0
        self._last_log_times = {}  # For throttled logging

        # QoS profiles
        # For images: best effort with small queue
        image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # For commands: reliable with small queue
        cmd_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Publishers
        self.path_pub = self.create_publisher(
            Path,
            '/navdp/trajectory_path',
            qos_profile=cmd_qos
        )

        self.values_pub = self.create_publisher(
            Float32MultiArray,
            '/navdp/all_values',
            qos_profile=cmd_qos
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            qos_profile=cmd_qos
        )

        # Subscribers with callback group for thread safety
        self.image_sub = self.create_subscription(
            Image,  # CompressedImage
            '/head_camera/rgb/image_raw',  # 'rgb/image_raw/compressed'
            self.image_cb,
            qos_profile=image_qos,
            callback_group=self.callback_group
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/head_camera/depth/image_raw',
            self.depth_cb,
            qos_profile=image_qos,
            callback_group=self.callback_group
        )

        self.goal_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.goal_cb,
            qos_profile=cmd_qos,
            callback_group=self.callback_group
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/head_camera/camera_info',
            self.camera_info_cb,
            qos_profile=cmd_qos,
            callback_group=self.callback_group
        )

        self.camera_info_color_sub = self.create_subscription(
            CameraInfo,
            '/head_camera/camera_info',
            self.camera_info_cb,
            qos_profile=cmd_qos,
            callback_group=self.callback_group
        )

        # Processing thread
        self._stop_event = threading.Event()
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

        # Auto-initialize after short delay (only once)
        self._initialized = False
        self.create_timer(0.2, self.auto_initialize_once)

        self.get_logger().info(
            f"NavDP ROS2 node initialized (robot_frame={self.robot_frame}, "
            f"world_frame={self.world_frame}, camera_frame={self.camera_frame})"
        )

    def log_throttle(self, key, message, level='warn', throttle_sec=5.0):
        """Throttled logging to avoid spam"""
        current_time = time.time()
        last_time = self._last_log_times.get(key, 0)

        if current_time - last_time >= throttle_sec:
            self._last_log_times[key] = current_time
            if level == 'warn':
                self.get_logger().warn(message)
            elif level == 'error':
                self.get_logger().error(message)
            elif level == 'info':
                self.get_logger().info(message)
            elif level == 'debug':
                self.get_logger().debug(message)
        # else: silently throttle

    def auto_initialize_once(self):
        """Auto-initialize NavDP after node startup (only once)"""
        if self._initialized:
            return
        try:
            self.initialize_navdp()
            self._initialized = True
            self.get_logger().info("NavDP initialized at startup.")
        except Exception as e:
            self.get_logger().warn(f"NavDP initialization failed: {e}")

    # ----------------- initialize -----------------
    def initialize_navdp(self):
        """Initialize NavDP agent"""
        if self.navdp_navigator is None:
            self.navdp_navigator = NavDP_Agent(
                self.intrinsic, image_size=224, memory_size=8, predict_size=24, temporal_depth=16,
                heads=8, token_dim=384, navi_model=self.checkpoint, device='cuda:0')

        self.navdp_navigator.reset(self.batch_size, self.stop_threshold)

        # FPS writer
        try:
            if self.navdp_fps_writer is not None:
                self.navdp_fps_writer.close()
        except Exception:
            pass

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.navdp_fps_writer = imageio.get_writer(f"{ts}_fps_pointgoal_ros2.mp4", fps=7)

        self.get_logger().info(f"navdp reset done: batch_size={self.batch_size}")

        # In ROS2, we return success directly since service response structure is different
        return True, "navdp reset"

    # ----------------- callbacks -----------------
    def image_cb(self, msg):
        """RGB image callback"""
        try:
            # Use imgmsg_to_cv2 for uncompressed Image messages
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            # For CompressedImage: img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
            with self._lock:
                self.latest_image = img
                # Convert timestamp to seconds
                self.latest_image_ts = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        except Exception as e:
            self.get_logger().error(f"CvBridge image error: {e}")

    def depth_cb(self, msg):
        """Depth image callback"""
        try:
            if msg.encoding in ('16UC1', 'mono16'):
                raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1').astype(np.float32)
                d = raw * 0.001   # RealSense: mm → m
            elif msg.encoding == '32FC1':
                d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1').astype(np.float32)
            else:
                return

            with self._lock:
                self.latest_depth = d
                self.latest_depth_ts = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        except Exception as e:
            self.get_logger().error(f"CvBridge depth error: {e}")

    def goal_cb(self, msg):
        """Goal point callback"""
        with self._lock:
            self.latest_goal = msg
            self.latest_goal_ts = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def camera_info_cb(self, msg):
        """Camera info callback"""
        try:
            K = np.array(msg.k, dtype=np.float32).reshape((3, 3))
            with self._lock:
                self.intrinsic = K
        except Exception as e:
            self.log_throttle("camera_info_error", f"Failed to read camera_info: {e}", 'warn', 5.0)

    # ----------------- transform helpers -----------------
    def transform_point(self, point, from_frame, to_frame):
        """Transform point between coordinate frames using TF2"""
        try:
            # Create transform stamped for lookup
            transform = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            # Build PointStamped so tf2 can access header.frame_id
            point_stamped = PointStamped()
            point_stamped.header.frame_id = from_frame
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point.x = float(point[0])
            point_stamped.point.y = float(point[1])
            point_stamped.point.z = float(point[2])

            # Transform using tf2_geometry_msgs
            transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return np.array([transformed.point.x, transformed.point.y, transformed.point.z])

        except tf2_ros.TransformException as e:
            self.log_throttle(f"tf_{from_frame}_{to_frame}", f"TF transform ({from_frame}->{to_frame}) failed: {e}", 'warn', 5.0)
            return None
        except Exception as e:
            self.log_throttle("tf_transform_error", f"Transform error: {e}", 'warn', 5.0)
            return None

    def get_robot_pose(self):
        """Get robot pose in world frame"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            # Extract translation
            trans = transform.transform.translation
            x, y = trans.x, trans.y

            # Extract yaw from quaternion
            rot = transform.transform.rotation
            q = [rot.x, rot.y, rot.z, rot.w]

            # ⚠️ transforms3d 需要 (w, x, y, z) 格式，不是 (x, y, z, w)！
            from transforms3d.euler import quat2euler
            q = [rot.w, rot.x, rot.y, rot.z]  # 改为 (w, x, y, z)
            euler = quat2euler(q, axes='sxyz')
            yaw = euler[2]  # yaw is the third element (z-axis rotation)
            
            self.get_logger().debug(f"Quat: {q}, Euler: {euler}, Yaw: {yaw:.4f} rad ({np.degrees(yaw):.2f}°)")

            return np.array([x, y, yaw], dtype=np.float32)

        except tf2_ros.TransformException as e:
            self.log_throttle("tf_robot_pose", f"TF lookup robot pose failed: {e}", 'warn', 5.0)
            return None
        except Exception as e:
            self.log_throttle("robot_pose_error", f"Robot pose error: {e}", 'warn', 5.0)
            return None

    # ----------------- processing loop -----------------
    def processing_loop(self):
        """Main processing loop at 15Hz"""
        rate = self.create_rate(15)  # 15Hz

        while rclpy.ok() and not self._stop_event.is_set():
            # Lock and copy latest data
            with self._lock:
                img = self.latest_image.copy() if self.latest_image is not None else None
                depth = self.latest_depth.copy() if self.latest_depth is not None else None
                goal = self.latest_goal

                # Get latest timestamp from sensor data only (not goal!)
                # Goal persists once set, so we don't need its timestamp for continuous processing
                ts = max(
                    getattr(self, 'latest_image_ts', 0.0),
                    getattr(self, 'latest_depth_ts', 0.0)
                )

            # Check if we have all required data
            if img is None or depth is None or goal is None:
                self.log_throttle("missing_data", 
                    f"Waiting for data... img: {img is not None}, depth: {depth is not None}, goal: {goal is not None}", 
                    'info', 2.0)
                rate.sleep()
                continue

            # Only skip if sensor data hasn't updated
            if ts <= self.last_processed_ts:
                rate.sleep()
                continue

            if self.navdp_navigator is None:
                self.log_throttle("navdp_not_init", "NavDP not initialized.", 'warn', 5.0)
                rate.sleep()
                continue

            # Preprocess
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image_input = bgr[np.newaxis, ...].astype(np.float32)
            depth_input = depth[..., np.newaxis][np.newaxis, ...].astype(np.float32)

            # Goal transform
            goal_z = getattr(goal.point, 'z', 0.0)
            goal_cam = self.transform_point(
                [goal.point.x, goal.point.y, goal_z],
                self.world_frame,
                self.camera_frame
            )

            if goal_cam is None:
                rate.sleep()
                continue

            goal_input = np.array([[goal_cam[0], goal_cam[1], 0]], dtype=np.float32)
            # goal_input = np.array([[2.0, 0.0, 0]], dtype=np.float32)  # DEBUG: fixed goal in front
            # goal_input = np.array([[0.0, 2.0, 0]], dtype=np.float32)  # DEBUG: fixed goal to the left
            # goal_input = np.array([[0.0, -2.0, 0]], dtype=np.float32)  # DEBUG: fixed goal to the right
            # goal_input = np.array([[-2.0, 0.0, 0]], dtype=np.float32)  # DEBUG: fixed goal close in front

            print(f"Goal in camera frame: {goal_input}")
            # NavDP step
            execute_traj, all_traj, all_values, traj_mask = self.navdp_navigator.step_pointgoal(
                goal_input, image_input, depth_input
            )

            # FPS video
            if self.navdp_fps_writer is not None and traj_mask is not None:
                # imageio expects uint8; cast to suppress float->uint8 warning
                mask_u8 = np.clip(traj_mask, 0, 255).astype(np.uint8)
                self.navdp_fps_writer.append_data(mask_u8)

            # Publish path
            path_msg = Path()
            now = self.get_clock().now()
            path_msg.header = Header(
                stamp=now.to_msg(),
                frame_id=self.world_frame
            )

            # Handle different trajectory dimensions
            if execute_traj.ndim == 3:
                traj_points = execute_traj[0]
            else:
                traj_points = execute_traj

            for pt in traj_points:
                world_pt = self.transform_point(pt[:3], self.camera_frame, self.world_frame)
                if world_pt is not None:
                    pose = PoseStamped()
                    pose.header = path_msg.header
                    pose.pose.position.x = world_pt[0]
                    pose.pose.position.y = world_pt[1]
                    pose.pose.position.z = world_pt[2]
                    path_msg.poses.append(pose)

            if path_msg.poses:
                self.path_pub.publish(path_msg)

            # Publish values
            values_msg = Float32MultiArray()
            values_msg.data = np.array(all_values, dtype=np.float32).ravel().tolist()
            self.values_pub.publish(values_msg)

            # Controller
            robot_pose = self.get_robot_pose()
            twist = Twist()

            if robot_pose is not None and execute_traj.size > 0:
                # Transform trajectory to world frame
                traj_world = []
                for pt in (execute_traj[0] if execute_traj.ndim == 3 else execute_traj):
                    world_pt = self.transform_point(pt[:3], self.camera_frame, self.world_frame)
                    if world_pt is not None and not np.any(np.isnan(world_pt)):
                        traj_world.append(world_pt)

                traj_world = np.array(traj_world)

                if traj_world.size > 0:
                    mpc = MPC_Controller(traj_world, desired_v=self.desired_v,
                                        v_max=self.v_max, w_max=self.w_max)
                    opt_u, _ = mpc.solve(robot_pose)

                    if opt_u.shape[0] > 1:
                        v_cmd, w_cmd = opt_u[1]
                    else:
                        v_cmd, w_cmd = opt_u[0]

                    twist.linear.x = np.clip(v_cmd, -self.v_max, self.v_max)
                    twist.angular.z = np.clip(w_cmd, -self.w_max, self.w_max)

            self.cmd_pub.publish(twist)
            self.last_processed_ts = ts

            rate.sleep()

    # ----------------- shutdown -----------------
    def shutdown(self):
        """Clean shutdown"""
        self._stop_event.set()

        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        try:
            if self.navdp_fps_writer:
                self.navdp_fps_writer.close()
        except Exception as e:
            self.get_logger().error(f"Error closing FPS writer: {e}")


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    try:
        node = NavDPNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    except Exception as e:
        node.get_logger().error(f"Node error: {e}")
    finally:
        if 'node' in locals():
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()