#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped as GeometryPoseStamped, PointStamped, Quaternion, Point
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
import tf2_ros
from tf2_geometry_msgs import do_transform_point
import tf_transformations
import traceback

from std_srvs.srv import SetBool

class ROILidarFusionNode(Node):
    def __init__(self):
        super().__init__('roi_lidar_fusion_node_activated')
        
        self.declare_parameter('input_image_topic', 'camera/image')
        self.declare_parameter('input_pc_topic', 'scan_02/points')
        self.declare_parameter('output_window', 'Fused View')
        self.declare_parameter('output_selected_pc_topic', 'selected_lidar_points')
        self.declare_parameter('output_corrected_goal_topic', '/corrected_local_goal')
        self.declare_parameter('navigation_frame', 'map')
        self.declare_parameter('img_w', 1920)
        self.declare_parameter('img_h', 1200)
        self.img_w_param = self.get_parameter('img_w').get_parameter_value().integer_value
        self.img_h_param = self.get_parameter('img_h').get_parameter_value().integer_value

        default_roi_x_start = int(self.img_w_param * 0 / 100.0)
        default_roi_y_start = int(self.img_h_param * 19 / 100.0)
        default_roi_x_end = int(self.img_w_param * 100 / 100.0)
        default_roi_y_end = int(self.img_h_param * 76 / 100.0)

        self.declare_parameter('roi_x_start', default_roi_x_start)
        self.declare_parameter('roi_y_start', default_roi_y_start)
        self.declare_parameter('roi_x_end', default_roi_x_end)
        self.declare_parameter('roi_y_end', default_roi_y_end)
        self.declare_parameter('enable_black_segmentation', True)
        for name, default in [('black_h_min', 0), ('black_s_min', 0), ('black_v_min', 0),
                              ('black_h_max', 180), ('black_s_max', 255), ('black_v_max', 50)]:
            self.declare_parameter(name, default)
        self.declare_parameter('point_display_mode', 2)
        self.declare_parameter('hfov', 1.25)
        self.declare_parameter('min_dist_colorize', 1.5)
        self.declare_parameter('max_dist_colorize', 5.0)
        self.declare_parameter('point_radius_viz', 2)
        self.declare_parameter('colormap_viz', cv2.COLORMAP_JET)
        self.declare_parameter('camera_optical_frame', 'front_camera_link_optical')
        self.declare_parameter('lidar_optical_frame', 'front_lidar_link_optical')
        self.declare_parameter('static_transform_lookup_timeout_sec', 5.0)

        self.input_image_topic_ = self.get_parameter('input_image_topic').value
        self.input_pc_topic_ = self.get_parameter('input_pc_topic').value
        self.output_window_ = self.get_parameter('output_window').value
        self.output_selected_pc_topic_ = self.get_parameter('output_selected_pc_topic').value
        self.output_corrected_goal_topic_ = self.get_parameter('output_corrected_goal_topic').value
        self.navigation_frame_ = self.get_parameter('navigation_frame').value
        self.roi_x_start = self.get_parameter('roi_x_start').value
        self.roi_y_start = self.get_parameter('roi_y_start').value
        self.roi_x_end = self.get_parameter('roi_x_end').value
        self.roi_y_end = self.get_parameter('roi_y_end').value
        self.enable_seg_param_ = self.get_parameter('enable_black_segmentation').value
        self.h_min, self.s_min, self.v_min = self.get_parameter('black_h_min').value, self.get_parameter('black_s_min').value, self.get_parameter('black_v_min').value
        self.h_max, self.s_max, self.v_max = self.get_parameter('black_h_max').value, self.get_parameter('black_s_max').value, self.get_parameter('black_v_max').value
        self.point_display_mode_ = self.get_parameter('point_display_mode').value
        self.hfov_ = self.get_parameter('hfov').value
        self.min_dist_colorize_ = self.get_parameter('min_dist_colorize').value
        self.max_dist_colorize_ = self.get_parameter('max_dist_colorize').value
        self.point_radius_viz_ = self.get_parameter('point_radius_viz').value
        self.colormap_viz_ = self.get_parameter('colormap_viz').value
        self.camera_optical_frame_ = self.get_parameter('camera_optical_frame').value
        self.lidar_optical_frame_ = self.get_parameter('lidar_optical_frame').value
        self.tf_timeout_sec_ = self.get_parameter('static_transform_lookup_timeout_sec').value

        self.roi_x_start = max(0, self.roi_x_start); self.roi_y_start = max(0, self.roi_y_start)
        self.roi_x_end = min(self.img_w_param, self.roi_x_end); self.roi_y_end = min(self.img_h_param, self.roi_y_end)
        if not (self.roi_x_start < self.roi_x_end and self.roi_y_start < self.roi_y_end):
            self.get_logger().error("Invalid ROI, defaulting to full image.")
            self.roi_x_start, self.roi_y_start = 0, 0; self.roi_x_end, self.roi_y_end = self.img_w_param, self.img_h_param
        self.get_logger().info(f"ROI: x=[{self.roi_x_start},{self.roi_x_end}), y=[{self.roi_y_start},{self.roi_y_end})")

        self.fx_ = self.img_w_param / (2 * math.tan(self.hfov_ / 2.0)); self.fy_ = self.fx_
        self.cx_ = self.img_w_param / 2.0; self.cy_ = self.img_h_param / 2.0
        self.get_logger().info(f"Cam Intrinsics: fx={self.fx_:.2f}, fy={self.fy_:.2f}, cx={self.cx_:.2f}, cy={self.cy_:.2f}")

        self.bridge_ = CvBridge(); self.latest_image_ = None; self.latest_pc_ = None
        self.tf_buffer_ = tf2_ros.Buffer(); self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)
        self.static_transform_lidar_to_cam_ = None
        self.process_and_publish_target_ = False

        if self.output_window_: cv2.namedWindow(self.output_window_, cv2.WINDOW_NORMAL); cv2.resizeWindow(self.output_window_, 800, 600)

        self.selected_points_publisher_ = self.create_publisher(PointCloud2, self.output_selected_pc_topic_, 10)
        _qos_reliable_transient = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.corrected_goal_publisher_ = self.create_publisher(GeometryPoseStamped, self.output_corrected_goal_topic_, _qos_reliable_transient)
        
        qos_sensor = rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(Image, self.input_image_topic_, self.image_cb, qos_sensor)
        self.create_subscription(PointCloud2, self.input_pc_topic_, self.pc_cb, qos_sensor)
        self.activation_service_ = self.create_service(SetBool, '~/activate_segmentation', self.handle_activation_request)
        
        self.timer_ = self.create_timer(1.0 / 15.0, self.timer_cb)
        self.get_logger().info('ROI-LiDAR Fusion Node (with activation) started. Target processing INACTIVE.')

    def handle_activation_request(self, request: SetBool.Request, response: SetBool.Response):
        self.process_and_publish_target_ = request.data
        response.message = f"Target processing and goal publishing {'ACTIVATED' if request.data else 'DEACTIVATED'}."
        self.get_logger().info(response.message)
        if not self.process_and_publish_target_: # If deactivating, publish an invalid goal
            self._publish_invalid_goal("Deactivated by service call.")
        response.success = True
        return response

    def _publish_invalid_goal(self, reason=""):
        invalid_goal_pose = GeometryPoseStamped()
        invalid_goal_pose.header.stamp = self.get_clock().now().to_msg()
        invalid_goal_pose.header.frame_id = "" # Empty frame_id signifies invalid/no target
        self.corrected_goal_publisher_.publish(invalid_goal_pose)
        self.get_logger().debug(f"Published invalid goal. Reason: {reason}")

    def image_cb(self, msg: Image):
        try:
            img = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
            if img.shape[1] != self.img_w_param or img.shape[0] != self.img_h_param:
                self.latest_image_ = cv2.resize(img, (self.img_w_param, self.img_h_param))
            else: self.latest_image_ = img
        except CvBridgeError as e: self.get_logger().error(f'Image conversion error: {e}')

    def pc_cb(self, msg: PointCloud2): self.latest_pc_ = msg

    def crop_and_segment(self, img_to_process):
        roi_content = img_to_process[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        segmentation_mask_in_roi = None
        if self.enable_seg_param_ and roi_content.size > 0:
            hsv_roi = cv2.cvtColor(roi_content, cv2.COLOR_BGR2HSV)
            lower_hsv = np.array([self.h_min, self.s_min, self.v_min])
            upper_hsv = np.array([self.h_max, self.s_max, self.v_max])
            segmentation_mask_in_roi = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
        return roi_content, segmentation_mask_in_roi, (self.roi_x_start, self.roi_y_start)

    def _get_static_transform_lidar_to_cam(self):
        try:
            transform_stamped = self.tf_buffer_.lookup_transform(
                self.camera_optical_frame_, self.lidar_optical_frame_, RclpyTime(),
                timeout=RclpyDuration(seconds=self.tf_timeout_sec_))
            q = transform_stamped.transform.rotation; t = transform_stamped.transform.translation
            T_mat = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            T_mat[:3, 3] = [t.x, t.y, t.z]
            self.static_transform_lidar_to_cam_ = T_mat
        except Exception as e:
            self.get_logger().warn(f"TF Fail (Lidar->Cam): {e}. Will retry.", throttle_duration_sec=5.0)
            self.static_transform_lidar_to_cam_ = None
            
    def process_lidar_and_image_data(self, base_image_for_viz, segmentation_mask_roi, roi_top_left_offset):
        points_for_viz_cloud = []; points_on_target_lidar_coords = []
        if self.latest_pc_ is None or self.static_transform_lidar_to_cam_ is None:
            return base_image_for_viz, points_for_viz_cloud, points_on_target_lidar_coords

        pts_raw_from_msg = pc2.read_points_numpy(self.latest_pc_, field_names=('x','y','z'), skip_nans=True)
        if pts_raw_from_msg.size == 0: return base_image_for_viz, points_for_viz_cloud, points_on_target_lidar_coords

        pts_lidar_frame = np.vstack((pts_raw_from_msg['x'], pts_raw_from_msg['y'], pts_raw_from_msg['z'])).T.astype(np.float32) \
            if pts_raw_from_msg.dtype.names else pts_raw_from_msg.reshape(-1,3).astype(np.float32)
        pts_lidar_frame = pts_lidar_frame[np.isfinite(pts_lidar_frame).all(axis=1)]
        if pts_lidar_frame.shape[0] == 0: return base_image_for_viz, points_for_viz_cloud, points_on_target_lidar_coords

        pts_h_lidar = np.hstack((pts_lidar_frame, np.ones((pts_lidar_frame.shape[0], 1))))
        pts_in_cam_optical_frame = (self.static_transform_lidar_to_cam_ @ pts_h_lidar.T).T[:, :3]
        
        Z_depth_proj = pts_in_cam_optical_frame[:,0]; X_px_val = -pts_in_cam_optical_frame[:,1]; Y_px_val = -pts_in_cam_optical_frame[:,2]
        valid_depth_mask = Z_depth_proj > 0.01
        X_px_val_vd, Y_px_val_vd, Z_depth_proj_vd = X_px_val[valid_depth_mask], Y_px_val[valid_depth_mask], Z_depth_proj[valid_depth_mask]
        original_pts_lidar_frame_vd = pts_lidar_frame[valid_depth_mask]
        if X_px_val_vd.size == 0: return base_image_for_viz, points_for_viz_cloud, points_on_target_lidar_coords

        u_img_coords = (self.fx_ * X_px_val_vd / Z_depth_proj_vd + self.cx_).astype(int)
        v_img_coords = (self.fy_ * Y_px_val_vd / Z_depth_proj_vd + self.cy_).astype(int)
        valid_proj_mask = (u_img_coords >= 0) & (u_img_coords < self.img_w_param) & (v_img_coords >= 0) & (v_img_coords < self.img_h_param)
        u_img_coords_valid, v_img_coords_valid = u_img_coords[valid_proj_mask], v_img_coords[valid_proj_mask]
        pts_lidar_for_valid_pixels = original_pts_lidar_frame_vd[valid_proj_mask]
        
        distances_from_lidar_origin = np.linalg.norm(pts_lidar_for_valid_pixels, axis=1)
        norm_dist_colorize = np.clip((distances_from_lidar_origin - self.min_dist_colorize_) / (self.max_dist_colorize_ - self.min_dist_colorize_), 0, 1)
        intensity_colorize = ((1.0 - norm_dist_colorize) * 255).astype(np.uint8)
        colors_for_viz = cv2.applyColorMap(intensity_colorize.reshape(-1,1), self.colormap_viz_).reshape(-1,3)
        if u_img_coords_valid.size == 0: return base_image_for_viz, points_for_viz_cloud, points_on_target_lidar_coords

        fused_image_display = base_image_for_viz.copy()
        for idx in range(len(u_img_coords_valid)):
            full_img_u, full_img_v = u_img_coords_valid[idx], v_img_coords_valid[idx]
            roi_relative_u, roi_relative_v = full_img_u - roi_top_left_offset[0], full_img_v - roi_top_left_offset[1]
            current_3d_point_in_lidar_frame = list(pts_lidar_for_valid_pixels[idx])
            is_on_segmented_target = self.enable_seg_param_ and segmentation_mask_roi is not None and \
                                     0 <= roi_relative_v < segmentation_mask_roi.shape[0] and \
                                     0 <= roi_relative_u < segmentation_mask_roi.shape[1] and \
                                     segmentation_mask_roi[roi_relative_v, roi_relative_u] > 0
            if is_on_segmented_target:
                points_for_viz_cloud.append(current_3d_point_in_lidar_frame)
                if self.process_and_publish_target_: points_on_target_lidar_coords.append(current_3d_point_in_lidar_frame)

            point_color_for_viz = tuple(map(int, colors_for_viz[idx]))
            if self.point_display_mode_ == 2 or (self.point_display_mode_ == 1 and is_on_segmented_target):
                if is_on_segmented_target and self.point_display_mode_ == 2: point_color_for_viz = (0,0,255) # Highlight red if all shown
                elif is_on_segmented_target and self.point_display_mode_ == 1: point_color_for_viz = (0,0,255) # Red for selected only
                if self.output_window_: cv2.circle(fused_image_display, (full_img_u, full_img_v), self.point_radius_viz_, point_color_for_viz, -1)
        return fused_image_display, points_for_viz_cloud, points_on_target_lidar_coords

    def calculate_and_publish_corrected_goal(self, points_on_target_lidar_frame: list, original_pc_header: Header):
        self.get_logger().info(f"FINAL CORRECTED GOAL (map frame): X={corrected_goal_pose.pose.position.x:.3f}, Y={corrected_goal_pose.pose.position.y:.3f}, Z={corrected_goal_pose.pose.position.z:.3f}")
        self.corrected_goal_publisher_.publish(corrected_goal_pose)
        if not self.process_and_publish_target_: return

        if not points_on_target_lidar_frame:
            self._publish_invalid_goal("No target points found to calculate corrected goal.")
            return

        target_points_np_lidar = np.array(points_on_target_lidar_frame, dtype=np.float32)
        if target_points_np_lidar.shape[0] == 0:
            self._publish_invalid_goal("Converted target points array is empty.")
            return
            
        centroid_lidar_frame = np.mean(target_points_np_lidar, axis=0)
        pt_stamped_lidar = PointStamped(header=original_pc_header, point=Point(x=float(centroid_lidar_frame[0]), y=float(centroid_lidar_frame[1]), z=float(centroid_lidar_frame[2])))

        try:
            tf_stamp_to_use = pt_stamped_lidar.header.stamp
            if tf_stamp_to_use.sec == 0 and tf_stamp_to_use.nanosec == 0:
                 tf_stamp_to_use = RclpyTime() # Use rclpy.time.Time() which defaults to node's clock
                 self.get_logger().warn(f"Lidar PC header stamp was zero, using current time for TF: {tf_stamp_to_use.nanoseconds}")


            transform_to_nav_frame = self.tf_buffer_.lookup_transform(
                self.navigation_frame_, pt_stamped_lidar.header.frame_id,
                tf_stamp_to_use, timeout=RclpyDuration(seconds=0.5))
            
            pt_stamped_nav_frame = do_transform_point(pt_stamped_lidar, transform_to_nav_frame)
            corrected_goal_pose = GeometryPoseStamped(header=Header(stamp=self.get_clock().now().to_msg(), frame_id=self.navigation_frame_))
            corrected_goal_pose.pose.position = pt_stamped_nav_frame.point
            q_identity = tf_transformations.quaternion_from_euler(0,0,0) 
            corrected_goal_pose.pose.orientation = Quaternion(x=q_identity[0],y=q_identity[1],z=q_identity[2],w=q_identity[3])
            self.corrected_goal_publisher_.publish(corrected_goal_pose)
            self.get_logger().info(f"Published corrected local goal in '{self.navigation_frame_}': P({corrected_goal_pose.pose.position.x:.2f}, {corrected_goal_pose.pose.position.y:.2f})")
        except Exception as ex:
            self.get_logger().warn(f"TF/GoalPub Error: {ex}", throttle_duration_sec=2.0)
            self._publish_invalid_goal(f"TF Error: {ex}")

    def timer_cb(self):
        if self.latest_image_ is None: return
        if self.static_transform_lidar_to_cam_ is None:
            self._get_static_transform_lidar_to_cam()
            if self.static_transform_lidar_to_cam_ is None:
                if self.output_window_: cv2.imshow(self.output_window_, cv2.resize(self.latest_image_.copy(), (800,600)))
                return

        _, segmentation_mask_in_roi, roi_top_left_offset = self.crop_and_segment(self.latest_image_)
        segmentation_found_target = self.enable_seg_param_ and segmentation_mask_in_roi is not None and cv2.countNonZero(segmentation_mask_in_roi) > 0
        image_for_visualization = self.latest_image_.copy()
        points_for_viz_topic, points_on_target_for_goal_calc = [], []

        if self.latest_pc_ is not None:
            image_for_visualization, points_for_viz_topic, points_on_target_for_goal_calc = \
                self.process_lidar_and_image_data(self.latest_image_, segmentation_mask_in_roi, roi_top_left_offset)
        
        if self.enable_seg_param_ and points_for_viz_topic and self.latest_pc_:
            selected_cloud_msg = pc2.create_cloud_xyz32(self.latest_pc_.header, points_for_viz_topic)
            self.selected_points_publisher_.publish(selected_cloud_msg)

        if self.process_and_publish_target_:
            if points_on_target_for_goal_calc and self.latest_pc_:
                self.calculate_and_publish_corrected_goal(points_on_target_for_goal_calc, self.latest_pc_.header)
            else: 
                self._publish_invalid_goal("No target points for goal in current frame while active.")

        if self.output_window_:
            if self.enable_seg_param_:
                roi_color = (0, 255, 0) if segmentation_found_target else (255, 191, 0)
                cv2.rectangle(image_for_visualization, (self.roi_x_start, self.roi_y_start), (self.roi_x_end - 1, self.roi_y_end - 1), roi_color, 1)
            cv2.imshow(self.output_window_, cv2.resize(image_for_visualization, (800,600)))
            if cv2.waitKey(1) & 0xFF == ord('q'): self.destroy_node(); rclpy.shutdown()

    def destroy_node(self):
        self.get_logger().info("Destroying ROILidarFusionNode.")
        if self.output_window_: cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try: node = ROILidarFusionNode(); rclpy.spin(node)
    except KeyboardInterrupt: pass
    except Exception as e: node.get_logger().error(f"Unhandled exception: {e}\n{traceback.format_exc()}") if node else print(f"Exc: {e}")
    finally:
        if node and rclpy.ok(): node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        if hasattr(node, 'output_window_') and node.output_window_: cv2.destroyAllWindows()

if __name__ == '__main__':
    main()