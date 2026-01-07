import rospy
import threading
import time
import datetime
import json

import numpy as np
import cv2
import imageio

from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PointStamped, Twist, PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Float32MultiArray, Header
from cv_bridge import CvBridge, CvBridgeError

import tf
from tf import transformations as tf_trans
from sensor_msgs.msg import CameraInfo

from baselines.navdp.policy_agent import NavDP_Agent
from utils_tasks.tracking_utils import MPC_Controller

class NavDPNode(object):
    def __init__(self):
        rospy.init_node('navdp_node', anonymous=False)

        # Params
        self.checkpoint = rospy.get_param('~checkpoint', '/home/doge/models/navdp/navdp-weights.ckpt')
        intrinsic_list = rospy.get_param('~intrinsic', [])
        if len(intrinsic_list) == 9:
            self.intrinsic = np.array(intrinsic_list, dtype=np.float32).reshape((3,3))
        else:
            rospy.logwarn("Intrinsic not provided or wrong size; using identity.")
            self.intrinsic = np.eye(3, dtype=np.float32)
        self.batch_size = int(rospy.get_param('~batch_size', 1))
        self.stop_threshold = np.array(rospy.get_param('~stop_threshold', [0.5]), dtype=np.float32)

        # controller params
        self.desired_v = float(rospy.get_param('~desired_v', 0.5))
        self.v_max = float(rospy.get_param('~v_max', self.desired_v))
        self.w_max = float(rospy.get_param('~w_max', 1.0))

        # frames: robot frame provided by tf (robot_id) and world frame (world_id)
        # Per request: robot_id = 'body', world_id = 'camera_init'
        self.robot_frame = rospy.get_param('~robot_frame', 'body')
        self.world_frame = rospy.get_param('~world_frame', 'camera_init')
        # camera frame (frame of images)
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_link')

        # objects
        self.navdp_navigator = None
        self.navdp_fps_writer = None
        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()

        # latest data
        self._lock = threading.Lock()
        self.latest_image = None
        self.latest_depth = None
        self.latest_goal = None          # PointStamped; goal is published in world_frame
        self.last_processed_ts = 0.0

        # pubs/subs/srv
        self.path_pub = rospy.Publisher('/navdp/trajectory_path', Path, queue_size=5)
        self.values_pub = rospy.Publisher('/navdp/all_values', Float32MultiArray, queue_size=5)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.image_cb, queue_size=1)
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_cb, queue_size=1)
        # goal is published in world coords (camera_init)
        rospy.Subscriber('/nav/goal', PointStamped, self.goal_cb, queue_size=1)
        # subscribe camera_info to read intrinsics
        rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.depth_camera_info_cb, queue_size=1)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.color_camera_info_cb, queue_size=1)

        self.reset_srv = rospy.Service('/navdp/reset', Trigger, self.handle_reset)

        # processing thread
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self.processing_loop)
        self._thread.daemon = True
        self._thread.start()

        # auto-initialize navigator on startup (will wait briefly for camera_info)
        try:
            rospy.sleep(0.2)
            self.initialize_navdp()
            rospy.loginfo("NavDP initialized at startup.")
        except Exception as e:
            rospy.logwarn("NavDP initialization failed: %s" % str(e))

        rospy.loginfo("NavDP ROS node initialized (robot_frame=%s, world_frame=%s, camera_frame=%s)" %
                      (self.robot_frame, self.world_frame, self.camera_frame))

    def handle_reset(self, req):
        # (re)create navigator
        self.batch_size = int(rospy.get_param('~batch_size', self.batch_size))
        stop_threshold = rospy.get_param('~stop_threshold', list(self.stop_threshold))
        self.stop_threshold = np.array(stop_threshold, dtype=np.float32)

        if self.navdp_navigator is None:
            self.navdp_navigator = NavDP_Agent(self.intrinsic,
                                              image_size=224,
                                              memory_size=8,
                                              predict_size=24,
                                              temporal_depth=16,
                                              heads=8,
                                              token_dim=384,
                                              navi_model=self.checkpoint,
                                              device='cuda:0')
        self.navdp_navigator.reset(self.batch_size, self.stop_threshold)

        # reopen fps writer for debugging/visualization (optional)
        try:
            if self.navdp_fps_writer is not None:
                self.navdp_fps_writer.close()
        except Exception:
            pass
        ts = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
        self.navdp_fps_writer = imageio.get_writer("{}_fps_pointgoal_ros.mp4".format(ts), fps=7)

        rospy.loginfo("navdp reset done: batch_size=%d" % self.batch_size)
        return TriggerResponse(success=True, message="navdp reset")

    def image_cb(self, msg):
        try:
            # msg is sensor_msgs/CompressedImage
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except CvBridgeError as e:
            rospy.logerr("CvBridge compressed image error: %s" % e)
            return
        with self._lock:
            self.latest_image = img.copy()
            self.latest_image_ts = msg.header.stamp.to_sec()

    def depth_cb(self, msg):
        try:
            # support 16UC1 or 32FC1
            if msg.encoding in ('16UC1','mono16'):
                d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                d = np.array(d, dtype=np.uint16).astype(np.float32) / 10000.0
            else:
                d = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                d = np.array(d, dtype=np.float32)
        except CvBridgeError as e:
            rospy.logerr("CvBridge depth error: %s" % e)
            return
        with self._lock:
            self.latest_depth = d.copy()
            self.latest_depth_ts = msg.header.stamp.to_sec()

    def goal_cb(self, msg: PointStamped):
        # Goal is defined in world_frame (camera_init)
        with self._lock:
            self.latest_goal = msg
            self.latest_goal_ts = msg.header.stamp.to_sec()

    def depth_camera_info_cb(self, msg: CameraInfo):
        # use depth camera_info to set intrinsics if not set
        try:
            K = np.array(msg.K, dtype=np.float32).reshape((3,3))
            with self._lock:
                self.intrinsic = K
                rospy.loginfo_throttle(5, "Depth camera_info received -> intrinsic updated")
        except Exception as e:
            rospy.logwarn_throttle(5, "Failed to read depth camera_info: %s" % str(e))

    def color_camera_info_cb(self, msg: CameraInfo):
        # use color camera_info to set intrinsics if not set; prefer color for RGB pipeline
        try:
            K = np.array(msg.K, dtype=np.float32).reshape((3,3))
            with self._lock:
                self.intrinsic = K
                rospy.loginfo_throttle(5, "Color camera_info received -> intrinsic updated")
        except Exception as e:
            rospy.logwarn_throttle(5, "Failed to read color camera_info: %s" % str(e))

    def transform_point_camera_to_world(self, point_cam):
        # Convert point in camera_frame to world_frame (camera_init)
        try:
            self.tf_listener.waitForTransform(self.world_frame, self.camera_frame, rospy.Time(0), rospy.Duration(0.5))
            (trans, rot) = self.tf_listener.lookupTransform(self.world_frame, self.camera_frame, rospy.Time(0))
            T = tf_trans.quaternion_matrix(rot)
            T[0:3, 3] = np.array(trans)
            p_cam_h = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
            p_world_h = T.dot(p_cam_h)
            return p_world_h[0:3]
        except Exception as e:
            rospy.logwarn_throttle(5, "tf transform (camera->world) failed: %s" % str(e))
            return None

    def transform_point_world_to_camera(self, point_world):
        # Convert point in world_frame to camera_frame
        try:
            self.tf_listener.waitForTransform(self.camera_frame, self.world_frame, rospy.Time(0), rospy.Duration(0.5))
            (trans, rot) = self.tf_listener.lookupTransform(self.camera_frame, self.world_frame, rospy.Time(0))
            T = tf_trans.quaternion_matrix(rot)
            T[0:3, 3] = np.array(trans)
            p_world_h = np.array([point_world[0], point_world[1], point_world[2], 1.0])
            p_cam_h = T.dot(p_world_h)
            return p_cam_h[0:3]
        except Exception as e:
            rospy.logwarn_throttle(5, "tf transform (world->camera) failed: %s" % str(e))
            return None

    def get_robot_pose_in_world(self):
        # get robot pose (x,y,yaw) in world_frame via tf between world_frame and robot_frame
        try:
            self.tf_listener.waitForTransform(self.world_frame, self.robot_frame, rospy.Time(0), rospy.Duration(0.5))
            (trans, rot) = self.tf_listener.lookupTransform(self.world_frame, self.robot_frame, rospy.Time(0))
            yaw = tf_trans.euler_from_quaternion(rot)[2]
            return np.array([trans[0], trans[1], yaw], dtype=np.float32)
        except Exception as e:
            rospy.logwarn_throttle(5, "tf lookup robot pose failed: %s" % str(e))
            return None

    def processing_loop(self):
        rate = rospy.Rate(15)
        while not rospy.is_shutdown() and not self._stop_event.is_set():
            try:
                with self._lock:
                    img = None if self.latest_image is None else self.latest_image.copy()
                    depth = None if self.latest_depth is None else self.latest_depth.copy()
                    goal_msg = None if self.latest_goal is None else self.latest_goal
                    ts = max(getattr(self, 'latest_image_ts', 0.0),
                             getattr(self, 'latest_depth_ts', 0.0),
                             getattr(self, 'latest_goal_ts', 0.0))
                if img is None or depth is None or goal_msg is None:
                    rate.sleep()
                    continue

                if ts <= self.last_processed_ts:
                    rate.sleep()
                    continue

                if self.navdp_navigator is None:
                    rospy.logwarn_throttle(10, "NavDP not initialized. Call /navdp/reset service.")
                    rate.sleep()
                    continue

                # Preprocess to NavDP format (batch_size assumed 1)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                H = bgr.shape[0]
                image_input = bgr.reshape((self.batch_size, -1, H, 3))

                if depth.ndim == 2:
                    depth_c = depth[:, :, np.newaxis]
                else:
                    depth_c = depth
                depth_input = depth_c.astype(np.float32).reshape((self.batch_size, -1, depth_c.shape[1], 1))

                # Goal is in world_frame; transform it to camera_frame for NavDP input
                gx_w = float(goal_msg.point.x)
                gy_w = float(goal_msg.point.y)
                gz_w = float(goal_msg.point.z) if hasattr(goal_msg.point, 'z') else 0.0
                p_cam = self.transform_point_world_to_camera([gx_w, gy_w, gz_w])
                if p_cam is None:
                    rospy.logwarn_throttle(5, "Cannot transform goal into camera frame; skipping this loop.")
                    rate.sleep()
                    continue
                # NavDP expects goal as (x,y,0) in camera coords (z ignored)
                goal_np = np.stack((np.array([p_cam[0]]), np.array([p_cam[1]]), np.zeros(1)), axis=1)  # (1,3)

                # Run NavDP
                start_t = time.time()
                execute_trajectory, all_trajectory, all_values, trajectory_mask = self.navdp_navigator.step_pointgoal(goal_np, image_input, depth_input)
                rospy.logdebug("NavDP step time: %.3f s" % (time.time() - start_t))

                # Optionally save visualization mask frames
                try:
                    if self.navdp_fps_writer is not None and trajectory_mask is not None:
                        self.navdp_fps_writer.append_data(trajectory_mask)
                except Exception:
                    pass

                # Publish execute_trajectory as nav_msgs/Path in world_frame.
                try:
                    traj = np.array(execute_trajectory)
                    if traj.ndim == 3:
                        traj_pts_cam = traj[0]
                    else:
                        traj_pts_cam = traj
                    path_msg = Path()
                    path_msg.header = Header(stamp=rospy.Time.now(), frame_id=self.world_frame)
                    for pt in traj_pts_cam:
                        x_cam = float(pt[0]); y_cam = float(pt[1]); z_cam = float(pt[2]) if pt.shape[0]>2 else 0.0
                        p_world = self.transform_point_camera_to_world([x_cam, y_cam, z_cam])
                        if p_world is None:
                            continue
                        pose = PoseStamped()
                        pose.header = path_msg.header
                        pose.pose.position.x = float(p_world[0])
                        pose.pose.position.y = float(p_world[1])
                        pose.pose.position.z = float(p_world[2])
                        path_msg.poses.append(pose)
                    if len(path_msg.poses) > 0:
                        self.path_pub.publish(path_msg)
                except Exception as e:
                    rospy.logerr("Failed to publish Path: %s" % str(e))

                # Publish all_values
                try:
                    values_arr = np.array(all_values, dtype=np.float32).ravel()
                    fa = Float32MultiArray()
                    fa.data = values_arr.tolist()
                    self.values_pub.publish(fa)
                except Exception as e:
                    rospy.logerr("Failed to publish all_values: %s" % str(e))

                # Controller: compute v,w using MPC based on path in world_frame and robot pose from tf
                try:
                    robot_pose = self.get_robot_pose_in_world()
                    if robot_pose is None:
                        rospy.logwarn_throttle(5, "No robot pose from tf; publishing zero cmd.")
                        twist = Twist()
                        self.cmd_pub.publish(twist)
                    else:
                        # prepare trajectory in world frame for MPC
                        traj_world = []
                        traj = np.array(execute_trajectory)
                        if traj.ndim == 3:
                            traj_cam = traj[0]
                        else:
                            traj_cam = traj
                        for pt in traj_cam:
                            p_world = self.transform_point_camera_to_world([float(pt[0]), float(pt[1]), float(pt[2]) if pt.shape[0]>2 else 0.0])
                            if p_world is not None:
                                traj_world.append(p_world[:2])
                        traj_world = np.array(traj_world)
                        if traj_world.size == 0:
                            twist = Twist()
                            self.cmd_pub.publish(twist)
                        else:
                            # initial state x0: [x, y, yaw] in world frame
                            x0 = robot_pose  # [x,y,yaw]
                            mpc = MPC_Controller(traj_world, desired_v=self.desired_v, v_max=self.v_max, w_max=self.w_max)
                            # Many MPC implementations expect (x,y,yaw) as initial
                            opt_u, opt_x = mpc.solve(x0)  
                            # opt_u shape: (H, 2) where columns are [v, w]
                            if opt_u.shape[0] > 1:
                                v_cmd = float(opt_u[1, 0])
                                w_cmd = float(opt_u[1, 1])
                            else:
                                v_cmd = float(opt_u[0, 0])
                                w_cmd = float(opt_u[0, 1])
                            # clip
                            v_cmd = np.clip(v_cmd, -self.v_max, self.v_max)
                            w_cmd = np.clip(w_cmd, -self.w_max, self.w_max)
                            twist = Twist()
                            twist.linear.x = v_cmd
                            twist.angular.z = w_cmd
                            self.cmd_pub.publish(twist)
                except Exception as e:
                    rospy.logerr("Controller error: %s" % str(e))

                self.last_processed_ts = ts

            except Exception as e:
                rospy.logerr("Processing loop error: %s" % str(e))
            rate.sleep()

    def shutdown(self):
        self._stop_event.set()
        try:
            if self.navdp_fps_writer is not None:
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