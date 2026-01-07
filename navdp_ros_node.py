import rospy
import threading
import time
import datetime
import numpy as np
import cv2
import imageio

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PointStamped, Twist, PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Float32MultiArray, Header
from cv_bridge import CvBridge, CvBridgeError
import tf
from tf import transformations as tf_trans

from baselines.navdp.policy_agent import NavDP_Agent
from utils_tasks.tracking_utils import MPC_Controller

class NavDPNode:
    def __init__(self):
        rospy.init_node('navdp_node')

        # params
        self.checkpoint = rospy.get_param('~checkpoint', '/home/doge/models/navdp/navdp-cross-modal.ckpt')
        self.batch_size = int(rospy.get_param('~batch_size', 1))
        self.stop_threshold = np.array(rospy.get_param('~stop_threshold', [0.5]), dtype=np.float32)
        self.desired_v = float(rospy.get_param('~desired_v', 0.5))
        self.v_max = float(rospy.get_param('~v_max', self.desired_v))
        self.w_max = float(rospy.get_param('~w_max', 1.0))

        self.robot_frame = rospy.get_param('~robot_frame', 'body')
        self.world_frame = rospy.get_param('~world_frame', 'camera_init')
        self.camera_frame = rospy.get_param('~camera_frame', 'body')

        intrinsic_list = rospy.get_param('~intrinsic', [])
        if len(intrinsic_list) == 9:
            self.intrinsic = np.array(intrinsic_list, dtype=np.float32).reshape((3,3))
        else:
            rospy.logwarn("Intrinsic not provided or wrong size; using identity.")
            self.intrinsic = np.eye(3, dtype=np.float32)

        # objects
        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()
        self.navdp_navigator = None
        self.navdp_fps_writer = None

        self._lock = threading.Lock()
        self.latest_image = None
        self.latest_depth = None
        self.latest_goal = None
        self.last_processed_ts = 0.0

        # publishers
        self.path_pub = rospy.Publisher('/navdp/trajectory_path', Path, queue_size=5)
        self.values_pub = rospy.Publisher('/navdp/all_values', Float32MultiArray, queue_size=5)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # subscribers
        rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.image_cb, queue_size=1)
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_cb, queue_size=1)
        rospy.Subscriber('/nav/goal', PointStamped, self.goal_cb, queue_size=1)
        rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.camera_info_cb, queue_size=1)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_cb, queue_size=1)

        # processing thread
        self._stop_event = threading.Event()
        threading.Thread(target=self.processing_loop, daemon=True).start()

        # auto-init
        rospy.sleep(0.2)
        try:
            self.initialize_navdp()
            rospy.loginfo("NavDP initialized at startup.")
        except Exception as e:
            rospy.logwarn(f"NavDP initialization failed: {e}")

        rospy.loginfo(f"NavDP ROS node initialized (robot_frame={self.robot_frame}, world_frame={self.world_frame}, camera_frame={self.camera_frame})")

    # ----------------- initialize -----------------
    def initialize_navdp(self):
        if self.navdp_navigator is None:
            self.navdp_navigator = NavDP_Agent(
                self.intrinsic, image_size=224, memory_size=8, predict_size=24, temporal_depth=16,
                heads=8, token_dim=384, navi_model=self.checkpoint, device='cuda:0')
        self.navdp_navigator.reset(self.batch_size, self.stop_threshold)

        # fps writer
        try:
            if self.navdp_fps_writer is not None:
                self.navdp_fps_writer.close()
        except Exception:
            pass
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.navdp_fps_writer = imageio.get_writer(f"{ts}_fps_pointgoal_ros.mp4", fps=7)
        rospy.loginfo(f"navdp reset done: batch_size={self.batch_size}")
        return TriggerResponse(success=True, message="navdp reset")

    # ----------------- callbacks -----------------
    def image_cb(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
            with self._lock:
                self.latest_image = img
                self.latest_image_ts = msg.header.stamp.to_sec()
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge image error: {e}")

    def depth_cb(self, msg):
        try:
            if msg.encoding in ('16UC1', 'mono16'):
                d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1').astype(np.float32)/10000.0
            else:
                d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            with self._lock:
                self.latest_depth = d
                self.latest_depth_ts = msg.header.stamp.to_sec()
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge depth error: {e}")

    def goal_cb(self, msg: PointStamped):
        with self._lock:
            self.latest_goal = msg
            self.latest_goal_ts = msg.header.stamp.to_sec()

    def camera_info_cb(self, msg: CameraInfo):
        try:
            K = np.array(msg.K, dtype=np.float32).reshape((3,3))
            with self._lock:
                self.intrinsic = K
        except Exception as e:
            rospy.logwarn_throttle(5, f"Failed to read camera_info: {e}")

    # ----------------- transform helpers -----------------
    def transform_point(self, point, from_frame, to_frame):
        try:
            self.tf_listener.waitForTransform(to_frame, from_frame, rospy.Time(0), rospy.Duration(0.5))
            (trans, rot) = self.tf_listener.lookupTransform(to_frame, from_frame, rospy.Time(0))
            T = tf_trans.quaternion_matrix(rot)
            T[0:3, 3] = np.array(trans)
            p_h = np.append(point, 1.0)
            return T.dot(p_h)[:3]
        except Exception as e:
            rospy.logwarn_throttle(5, f"tf transform ({from_frame}->{to_frame}) failed: {e}")
            return None

    def get_robot_pose(self):
        try:
            self.tf_listener.waitForTransform(self.world_frame, self.robot_frame, rospy.Time(0), rospy.Duration(0.5))
            (trans, rot) = self.tf_listener.lookupTransform(self.world_frame, self.robot_frame, rospy.Time(0))
            yaw = tf_trans.euler_from_quaternion(rot)[2]
            return np.array([trans[0], trans[1], yaw], dtype=np.float32)
        except Exception as e:
            rospy.logwarn_throttle(5, f"tf lookup robot pose failed: {e}")
            return None

    # ----------------- processing loop -----------------
    def processing_loop(self):
        rate = rospy.Rate(15)
        while not rospy.is_shutdown() and not self._stop_event.is_set():
            with self._lock:
                img = self.latest_image.copy() if self.latest_image is not None else None
                depth = self.latest_depth.copy() if self.latest_depth is not None else None
                goal = self.latest_goal
                ts = max(getattr(self, 'latest_image_ts', 0.0),
                         getattr(self, 'latest_depth_ts', 0.0),
                         getattr(self, 'latest_goal_ts', 0.0))

            if img is None or depth is None or goal is None or ts <= self.last_processed_ts:
                rate.sleep()
                continue
            if self.navdp_navigator is None:
                rospy.logwarn_throttle(10, "NavDP not initialized. Call /navdp/reset service.")
                rate.sleep()
                continue

            # preprocess
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image_input = bgr[np.newaxis, ...].astype(np.float32)
            depth_input = depth[..., np.newaxis][np.newaxis, ...].astype(np.float32)

            # goal transform
            goal_cam = self.transform_point([goal.point.x, goal.point.y, getattr(goal.point,'z',0.0)], self.world_frame, self.camera_frame)
            if goal_cam is None:
                rate.sleep()
                continue
            goal_input = np.array([[goal_cam[0], goal_cam[1], 0]], dtype=np.float32)

            # NavDP step
            execute_traj, all_traj, all_values, traj_mask = self.navdp_navigator.step_pointgoal(goal_input, image_input, depth_input)

            # fps video
            if self.navdp_fps_writer is not None and traj_mask is not None:
                self.navdp_fps_writer.append_data(traj_mask)

            # publish path
            path_msg = Path()
            path_msg.header = Header(stamp=rospy.Time.now(), frame_id=self.world_frame)
            for pt in execute_traj[0] if execute_traj.ndim==3 else execute_traj:
                world_pt = self.transform_point(pt[:3], self.camera_frame, self.world_frame)
                if world_pt is not None:
                    pose = PoseStamped()
                    pose.header = path_msg.header
                    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = world_pt
                    path_msg.poses.append(pose)
            if path_msg.poses:
                self.path_pub.publish(path_msg)

            # publish values
            self.values_pub.publish(Float32MultiArray(data=np.array(all_values, dtype=np.float32).ravel().tolist()))

            # controller
            robot_pose = self.get_robot_pose()
            twist = Twist()
            if robot_pose is not None and execute_traj.size>0:
                traj_world = np.array([self.transform_point(pt[:3], self.camera_frame, self.world_frame) for pt in (execute_traj[0] if execute_traj.ndim==3 else execute_traj)])
                traj_world = traj_world[~np.isnan(traj_world).any(axis=1)]
                if traj_world.size>0:
                    mpc = MPC_Controller(traj_world, desired_v=self.desired_v, v_max=self.v_max, w_max=self.w_max)
                    opt_u, _ = mpc.solve(robot_pose)
                    v_cmd, w_cmd = opt_u[1] if opt_u.shape[0]>1 else opt_u[0]
                    twist.linear.x = np.clip(v_cmd, -self.v_max, self.v_max)
                    twist.angular.z = np.clip(w_cmd, -self.w_max, self.w_max)
            self.cmd_pub.publish(twist)
            self.last_processed_ts = ts
            rate.sleep()

    # ----------------- shutdown -----------------
    def shutdown(self):
        self._stop_event.set()
        try:
            if self.navdp_fps_writer:
                self.navdp_fps_writer.close()
        except Exception:
            pass

if __name__ == '__main__':
    node = NavDPNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    node.shutdown()
