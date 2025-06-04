#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclpyTime # Explicit import
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header # For creating headers if needed
from geometry_msgs.msg import PoseStamped as GeometryPoseStamped, PointStamped, Quaternion
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
import tf2_ros
from tf2_geometry_msgs import do_transform_point
import tf_transformations
import traceback

from std_srvs.srv import SetBool # Standard service for activation

class ROILidarFusionNode(Node):
    def __init__(self):
        super().__init__('roi_lidar_fusion_node_activated') # Changed node name for clarity
        
        # --- Parameters ---
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

        self.declare_parameter('enable_black_segmentation', True) # General toggle for segmentation logic
        for name, default in [('black_h_min', 0), ('black_s_min', 0), ('black_v_min', 0),
                              ('black_h_max', 180), ('black_s_max', 255), ('black_v_max', 50)]: # Adjusted v_max
            self.declare_parameter(name, default)

        self.declare_parameter('point_display_mode', 2) # 0:None, 1:Selected, 2:All
        self.declare_parameter('hfov', 1.25) # Horizontal Field of View for camera
        self.declare_parameter('min_dist_colorize', 1.5) # Min distance for point cloud colorization
        self.declare_parameter('max_dist_colorize', 5.0) # Max distance for point cloud colorization
        self.declare_parameter('point_radius_viz', 2) # Radius for drawing points in visualization
        self.declare_parameter('colormap_viz', cv2.COLORMAP_JET)
        self.declare_parameter('camera_optical_frame', 'front_camera_link_optical') # TF frame for camera intrinsics
        self.declare_parameter('lidar_optical_frame', 'front_lidar_link_optical')   # TF frame of incoming Lidar points
        self.declare_parameter('static_transform_lookup_timeout_sec', 5.0)

        # Get parameters
        self.input_image_topic_ = self.get_parameter('input_image_topic').get_parameter_value().string_value
        self.input_pc_topic_ = self.get_parameter('input_pc_topic').get_parameter_value().string_value
        self.output_window_ = self.get_parameter('output_window').get_parameter_value().string_value
        self.output_selected_pc_topic_ = self.get_parameter('output_selected_pc_topic').get_parameter_value().string_value
        self.output_corrected_goal_topic_ = self.get_parameter('output_corrected_goal_topic').get_parameter_value().string_value
        self.navigation_frame_ = self.get_parameter('navigation_frame').get_parameter_value().string_value

        self.roi_x_start = self.get_parameter('roi_x_start').get_parameter_value().integer_value
        self.roi_y_start = self.get_parameter('roi_y_start').get_parameter_value().integer_value
        self.roi_x_end = self.get_parameter('roi_x_end').get_parameter_value().integer_value
        self.roi_y_end = self.get_parameter('roi_y_end').get_parameter_value().integer_value
        self.enable_seg_param_ = self.get_parameter('enable_black_segmentation').get_parameter_value().bool_value
        self.h_min = self.get_parameter('black_h_min').get_parameter_value().integer_value
        self.s_min = self.get_parameter('black_s_min').get_parameter_value().integer_value
        self.v_min = self.get_parameter('black_v_min').get_parameter_value().integer_value
        self.h_max = self.get_parameter('black_h_max').get_parameter_value().integer_value
        self.s_max = self.get_parameter('black_s_max').get_parameter_value().integer_value
        self.v_max = self.get_parameter('black_v_max').get_parameter_value().integer_value
        self.point_display_mode_ = self.get_parameter('point_display_mode').get_parameter_value().integer_value
        self.hfov_ = self.get_parameter('hfov').get_parameter_value().double_value
        self.min_dist_colorize_ = self.get_parameter('min_dist_colorize').get_parameter_value().double_value
        self.max_dist_colorize_ = self.get_parameter('max_dist_colorize').get_parameter_value().double_value
        self.point_radius_viz_ = self.get_parameter('point_radius_viz').get_parameter_value().integer_value
        self.colormap_viz_ = self.get_parameter('colormap_viz').get_parameter_value().integer_value
        self.camera_optical_frame_ = self.get_parameter('camera_optical_frame').get_parameter_value().string_value
        self.lidar_optical_frame_ = self.get_parameter('lidar_optical_frame').get_parameter_value().string_value
        self.tf_timeout_sec_ = self.get_parameter('static_transform_lookup_timeout_sec').get_parameter_value().double_value

        # Validate and clamp ROI
        self.roi_x_start = max(0, self.roi_x_start)
        self.roi_y_start = max(0, self.roi_y_start)
        self.roi_x_end = min(self.img_w_param, self.roi_x_end)
        self.roi_y_end = min(self.img_h_param, self.roi_y_end)
        if not (self.roi_x_start < self.roi_x_end and self.roi_y_start < self.roi_y_end):
            self.get_logger().error("Invalid ROI, defaulting to full image.")
            self.roi_x_start, self.roi_y_start = 0, 0
            self.roi_x_end, self.roi_y_end = self.img_w_param, self.img_h_param
        self.get_logger().info(f"Using ROI: x=[{self.roi_x_start},{self.roi_x_end}), y=[{self.roi_y_start},{self.roi_y_end})")

        # Camera intrinsics based on parameters
        self.fx_ = self.img_w_param / (2 * math.tan(self.hfov_ / 2.0))
        self.fy_ = self.fx_ # Assuming square pixels
        self.cx_ = self.img_w_param / 2.0
        self.cy_ = self.img_h_param / 2.0
        self.get_logger().info(f"Cam Intrinsics: fx={self.fx_:.2f}, fy={self.fy_:.2f}, cx={self.cx_:.2f}, cy={self.cy_:.2f}")

        self.bridge_ = CvBridge()
        self.latest_image_ = None
        self.latest_pc_ = None
        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)
        self.static_transform_lidar_to_cam_ = None # Stores the numpy matrix

        if self.output_window_:
            cv2.namedWindow(self.output_window_, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.output_window_, 800, 600)

        # --- New state variable for activation ---
        self.process_and_publish_target_ = False # Controlled by service, start as False

        # Publishers
        self.selected_points_publisher_ = self.create_publisher(PointCloud2, self.output_selected_pc_topic_, 10)
        _qos_reliable_transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, 
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.corrected_goal_publisher_ = self.create_publisher(
            GeometryPoseStamped, self.output_corrected_goal_topic_, _qos_reliable_transient)
        self.get_logger().info(f"Publishing corrected local goals to: {self.output_corrected_goal_topic_}")

        # Subscribers (using SENSOR_DATA QoS for potentially high-rate image/lidar)
        qos_sensor = rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(Image, self.input_image_topic_, self.image_cb, qos_sensor)
        self.create_subscription(PointCloud2, self.input_pc_topic_, self.pc_cb, qos_sensor)

        # Service Server for activation
        self.activation_service_ = self.create_service(
            SetBool, '~/activate_segmentation', self.handle_activation_request)
        self.get_logger().info(f"Activation service '~/activate_segmentation' is ready.")

        self.timer_ = self.create_timer(1.0 / 15.0, self.timer_cb) # Process at 15Hz
        self.get_logger().info('ROI-LiDAR Fusion Node (with activation) started. Target processing INACTIVE.')

    def handle_activation_request(self, request: SetBool.Request, response: SetBool.Response):
        self.process_and_publish_target_ = request.data
        if self.process_and_publish_target_:
            response.message = "Target processing and goal publishing ACTIVATED."
        else:
            response.message = "Target processing and goal publishing DEACTIVATED."
            # Publish an empty/invalid PoseStamped to signal no active target
            empty_pose = GeometryPoseStamped()
            empty_pose.header.stamp = self.get_clock().now().to_msg()
            empty_pose.header.frame_id = "" # Explicitly empty or clearly invalid
            self.corrected_goal_publisher_.publish(empty_pose)
        self.get_logger().info(response.message)
        response.success = True
        return response

    def image_cb(self, msg: Image):
        try:
            img = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
            # Resize if incoming image dimensions differ from configured ones
            if img.shape[1] != self.img_w_param or img.shape[0] != self.img_h_param:
                self.latest_image_ = cv2.resize(img, (self.img_w_param, self.img_h_param))
            else:
                self.latest_image_ = img
        except CvBridgeError as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def pc_cb(self, msg: PointCloud2):
        self.latest_pc_ = msg

    def crop_and_segment(self, img_to_process):
        # Assumes img_to_process is already at self.img_w_param x self.img_h_param
        roi_content = img_to_process[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        segmentation_mask_in_roi = None # Mask relative to ROI
        if self.enable_seg_param_ and roi_content.size > 0:
            hsv_roi = cv2.cvtColor(roi_content, cv2.COLOR_BGR2HSV)
            lower_hsv = np.array([self.h_min, self.s_min, self.v_min])
            upper_hsv = np.array([self.h_max, self.s_max, self.v_max])
            segmentation_mask_in_roi = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
        
        # Offset of the ROI's top-left corner in the full image
        roi_top_left_offset = (self.roi_x_start, self.roi_y_start)
        return roi_content, segmentation_mask_in_roi, roi_top_left_offset

    def _get_static_transform_lidar_to_cam(self):
        try:
            transform_stamped = self.tf_buffer_.lookup_transform(
                self.camera_optical_frame_, self.lidar_optical_frame_, RclpyTime(), # Get latest
                timeout=RclpyDuration(seconds=self.tf_timeout_sec_)
            )
            q = transform_stamped.transform.rotation
            t = transform_stamped.transform.translation
            T_mat = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            T_mat[:3, 3] = [t.x, t.y, t.z]
            self.static_transform_lidar_to_cam_ = T_mat
            self.get_logger().debug(f"Fetched static transform from {self.lidar_optical_frame_} to {self.camera_optical_frame_}")
        except Exception as e:
            self.get_logger().warn(f"TF Fail (Lidar->Cam): {e}. Will retry.")
            self.static_transform_lidar_to_cam_ = None
            
    def process_lidar_and_image_data(self, base_image_for_viz, segmentation_mask_roi, roi_top_left_offset):
        points_for_viz_cloud = [] # Points that fall in segmented region (for selected_lidar_points topic)
        points_on_target_lidar_coords = [] # 3D points (in lidar frame) that are on the segmented target

        if self.latest_pc_ is None or self.static_transform_lidar_to_cam_ is None:
            return base_image_for_viz, points_for_viz_cloud, points_on_target_lidar_coords

        pts_raw_from_msg = pc2.read_points_numpy(self.latest_pc_, field_names=('x','y','z'), skip_nans=True)
        if pts_raw_from_msg.size == 0: return base_image_for_viz, points_for_viz_cloud, points_on_target_lidar_coords

        # Ensure pts_lidar_frame is a simple Nx3 array of floats
        if pts_raw_from_msg.dtype.names: # If it's a structured array
            pts_lidar_frame = np.vstack((pts_raw_from_msg['x'], pts_raw_from_msg['y'], pts_raw_from_msg['z'])).T.astype(np.float32)
        else: # If it's already a plain array (e.g., from some preprocessing)
            pts_lidar_frame = pts_raw_from_msg.reshape(-1,3).astype(np.float32)
        
        pts_lidar_frame = pts_lidar_frame[np.isfinite(pts_lidar_frame).all(axis=1)]
        if pts_lidar_frame.shape[0] == 0: return base_image_for_viz, points_for_viz_cloud, points_on_target_lidar_coords

        # Transform points from lidar_optical_frame to camera_optical_frame
        pts_h_lidar = np.hstack((pts_lidar_frame, np.ones((pts_lidar_frame.shape[0], 1))))
        pts_in_cam_optical_frame = (self.static_transform_lidar_to_cam_ @ pts_h_lidar.T).T[:, :3]

        # Projection logic (adjust based on your actual camera_optical_frame definition relative to lidar_optical_frame)
        # Common convention for camera_optical_frame: Z forward, X right, Y down
        # If lidar_optical_frame is X forward, Y left, Z up, then after transform to camera_optical:
        # Cam's Z (depth) = Lidar's X (if aligned)
        # Cam's X (u-coord) = Lidar's -Y (if aligned)
        # Cam's Y (v-coord) = Lidar's -Z (if aligned)
        # Your original projection: X_cam_proj=-cam_coords[:,1], Y_cam_proj=-cam_coords[:,2], Z_cam_depth=cam_coords[:,0]
        # This implies Z_depth_for_projection = pts_in_cam_optical_frame[:,0]
        #             X_pixel_direction_val = -pts_in_cam_optical_frame[:,1]
        #             Y_pixel_direction_val = -pts_in_cam_optical_frame[:,2]
        
        Z_depth_proj = pts_in_cam_optical_frame[:,0] 
        X_px_val = -pts_in_cam_optical_frame[:,1]
        Y_px_val = -pts_in_cam_optical_frame[:,2]

        valid_depth_mask = Z_depth_proj > 0.01 # Points in front of camera
        X_px_val_vd = X_px_val[valid_depth_mask]
        Y_px_val_vd = Y_px_val[valid_depth_mask]
        Z_depth_proj_vd = Z_depth_proj[valid_depth_mask]
        original_pts_lidar_frame_vd = pts_lidar_frame[valid_depth_mask] # Keep original lidar coords

        if X_px_val_vd.size == 0: return base_image_for_viz, points_for_viz_cloud, points_on_target_lidar_coords

        # Calculate pixel coordinates
        u_img_coords = (self.fx_ * X_px_val_vd / Z_depth_proj_vd + self.cx_).astype(int)
        v_img_coords = (self.fy_ * Y_px_val_vd / Z_depth_proj_vd + self.cy_).astype(int)

        # Filter points that project within the full image bounds
        valid_proj_mask = (u_img_coords >= 0) & (u_img_coords < self.img_w_param) & \
                          (v_img_coords >= 0) & (v_img_coords < self.img_h_param)
        
        u_img_coords_valid = u_img_coords[valid_proj_mask]
        v_img_coords_valid = v_img_coords[valid_proj_mask]
        # Keep original lidar coordinates of points that project validly onto the full image
        pts_lidar_for_valid_pixels = original_pts_lidar_frame_vd[valid_proj_mask]
        
        # --- Colorization for visualization ---
        distances_from_lidar_origin = np.linalg.norm(pts_lidar_for_valid_pixels, axis=1)
        norm_dist_colorize = np.clip((distances_from_lidar_origin - self.min_dist_colorize_) / \
                                     (self.max_dist_colorize_ - self.min_dist_colorize_), 0, 1)
        intensity_colorize = ((1.0 - norm_dist_colorize) * 255).astype(np.uint8) # Closer is hotter (red in JET)
        colors_for_viz = cv2.applyColorMap(intensity_colorize.reshape(-1,1), self.colormap_viz_).reshape(-1,3)
        # --- End Colorization ---

        if u_img_coords_valid.size == 0: return base_image_for_viz, points_for_viz_cloud, points_on_target_lidar_coords

        fused_image_display = base_image_for_viz.copy() # Image for drawing points on

        for idx in range(len(u_img_coords_valid)):
            full_img_u, full_img_v = u_img_coords_valid[idx], v_img_coords_valid[idx]
            
            # Coordinates relative to the ROI's top-left corner
            roi_relative_u = full_img_u - roi_top_left_offset[0]
            roi_relative_v = full_img_v - roi_top_left_offset[1]
            
            current_3d_point_in_lidar_frame = list(pts_lidar_for_valid_pixels[idx])

            is_on_segmented_target = False
            if self.enable_seg_param_ and segmentation_mask_roi is not None and \
               0 <= roi_relative_v < segmentation_mask_roi.shape[0] and \
               0 <= roi_relative_u < segmentation_mask_roi.shape[1] and \
               segmentation_mask_roi[roi_relative_v, roi_relative_u] > 0:
                is_on_segmented_target = True
                points_for_viz_cloud.append(current_3d_point_in_lidar_frame)
                if self.process_and_publish_target_: # Only collect for goal if master switch is on
                    points_on_target_lidar_coords.append(current_3d_point_in_lidar_frame)

            # Visualization drawing logic
            draw_this_point_for_viz = False
            point_color_for_viz = tuple(map(int, colors_for_viz[idx]))

            if self.point_display_mode_ == 2: # Display all projected points
                draw_this_point_for_viz = True
                if is_on_segmented_target: point_color_for_viz = (0, 0, 255) # Highlight segmented points in red
            elif self.point_display_mode_ == 1 and is_on_segmented_target: # Display only selected (segmented)
                draw_this_point_for_viz = True
                point_color_for_viz = (0, 0, 255) # Color selected points red

            if draw_this_point_for_viz and self.output_window_:
                cv2.circle(fused_image_display, (full_img_u, full_img_v), self.point_radius_viz_, point_color_for_viz, -1)
                        
        return fused_image_display, points_for_viz_cloud, points_on_target_lidar_coords

    def calculate_and_publish_corrected_goal(self, points_on_target_lidar_frame: list, original_pc_header: Header):
        if not self.process_and_publish_target_ or not points_on_target_lidar_frame:
            return

        target_points_np_lidar = np.array(points_on_target_lidar_frame, dtype=np.float32)
        centroid_lidar_frame = np.mean(target_points_np_lidar, axis=0)

        pt_stamped_lidar = PointStamped()
        pt_stamped_lidar.header = original_pc_header # Use stamp and frame_id from source Lidar PC
        pt_stamped_lidar.point.x = float(centroid_lidar_frame[0])
        pt_stamped_lidar.point.y = float(centroid_lidar_frame[1])
        pt_stamped_lidar.point.z = float(centroid_lidar_frame[2])

        try:
            transform_to_nav_frame = self.tf_buffer_.lookup_transform(
                self.navigation_frame_, pt_stamped_lidar.header.frame_id,
                pt_stamped_lidar.header.stamp if (pt_stamped_lidar.header.stamp.sec > 0 or pt_stamped_lidar.header.stamp.nanosec > 0) else RclpyTime(),
                timeout=RclpyDuration(seconds=0.5)
            )
            pt_stamped_nav_frame = do_transform_point(pt_stamped_lidar, transform_to_nav_frame)

            corrected_goal_pose = GeometryPoseStamped()
            corrected_goal_pose.header.stamp = self.get_clock().now().to_msg()
            corrected_goal_pose.header.frame_id = self.navigation_frame_
            corrected_goal_pose.pose.position = pt_stamped_nav_frame.point
            q_identity = tf_transformations.quaternion_from_euler(0,0,0) # Assuming target is on ground, orientation might not matter much for position
            corrected_goal_pose.pose.orientation = Quaternion(x=q_identity[0],y=q_identity[1],z=q_identity[2],w=q_identity[3])
            
            self.corrected_goal_publisher_.publish(corrected_goal_pose)
            self.get_logger().info(f"Published corrected local goal in '{self.navigation_frame_}': "
                                   f"P({corrected_goal_pose.pose.position.x:.2f}, {corrected_goal_pose.pose.position.y:.2f})")
        except Exception as ex:
            self.get_logger().warn(f"TF/GoalPub Error: {ex}\n{traceback.format_exc()}", throttle_duration_sec=2.0)

    def timer_cb(self):
        if self.latest_image_ is None: 
            self.get_logger().debug("Timer CB: No latest image.", throttle_duration_sec=5.0)
            return

        if self.static_transform_lidar_to_cam_ is None:
            self._get_static_transform_lidar_to_cam()
            if self.static_transform_lidar_to_cam_ is None:
                self.get_logger().warn("Timer CB: Waiting for Lidar->Cam TF. Skipping fusion.", throttle_duration_sec=2.0)
                if self.output_window_: cv2.imshow(self.output_window_, cv2.resize(self.latest_image_.copy(), (800,600)))
                return

        # Perform cropping and segmentation on the latest image
        _, segmentation_mask_in_roi, roi_top_left_offset = self.crop_and_segment(self.latest_image_)
        
        segmentation_found_target = False
        if self.enable_seg_param_ and segmentation_mask_in_roi is not None and cv2.countNonZero(segmentation_mask_in_roi) > 0:
            segmentation_found_target = True

        # Start with the raw camera image for display
        image_for_visualization = self.latest_image_.copy()
        
        points_for_viz_topic, points_on_target_for_goal_calc = [], []

        if self.latest_pc_ is not None:
            # This function handles projection for visualization AND extracts points for goal calculation
            image_for_visualization, points_for_viz_topic, points_on_target_for_goal_calc = \
                self.process_lidar_and_image_data(self.latest_image_, segmentation_mask_in_roi, roi_top_left_offset)
        
        # Publish the point cloud for visualization (points that fell in segmented region)
        if self.enable_seg_param_ and points_for_viz_topic and self.latest_pc_:
            pc_header_for_viz = self.latest_pc_.header # Use original lidar PC header
            selected_cloud_msg = pc2.create_cloud_xyz32(pc_header_for_viz, points_for_viz_topic)
            self.selected_points_publisher_.publish(selected_cloud_msg)

        # If target processing is active AND we found points on the target, calculate and publish the goal
        if self.process_and_publish_target_ and points_on_target_for_goal_calc and self.latest_pc_:
            self.calculate_and_publish_corrected_goal(points_on_target_for_goal_calc, self.latest_pc_.header)
        elif self.process_and_publish_target_ and not points_on_target_for_goal_calc:
            # Active, but no target points found in this frame.
            # Deactivation (publishing empty pose) is handled by the service call to SetBool=False.
            # So, if process_and_publish_target_ is True but no points, we just don't publish a new goal.
            self.get_logger().debug("Target processing active, but no target points found in current frame to form a goal.")


        if self.output_window_:
            # Draw ROI border on the visualization image
            if self.enable_seg_param_:
                roi_color = (0, 255, 0) if segmentation_found_target else (255, 191, 0)
                cv2.rectangle(image_for_visualization, 
                              (self.roi_x_start, self.roi_y_start), 
                              (self.roi_x_end - 1, self.roi_y_end - 1), 
                              roi_color, 1)
            
            display_img_resized = cv2.resize(image_for_visualization, (800, 600))
            cv2.imshow(self.output_window_, display_img_resized)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Shutdown requested by Q key')
                self.destroy_node() # This should trigger on_shutdown if overridden
                rclpy.shutdown()
                if self.output_window_: cv2.destroyAllWindows()

    def destroy_node(self): # Override to ensure windows close
        self.get_logger().info("Destroying ROILidarFusionNode.")
        if self.output_window_:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ROILidarFusionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info('Keyboard interrupt, shutting down ROILidarFusionNode...')
    except Exception as e:
        if node: node.get_logger().error(f"Unhandled exception in ROILidarFusionNode: {e}\n{traceback.format_exc()}")
        else: print(f"Unhandled exception during ROILidarFusionNode init: {e}\n{traceback.format_exc()}")
    finally:
        if node and rclpy.ok(): # Ensure node exists and rclpy is still ok
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        # Ensure OpenCV windows are closed, even if destroy_node wasn't called or window wasn't set
        if hasattr(node, 'output_window_') and node.output_window_:
            cv2.destroyAllWindows()
        elif cv2.getWindowProperty("Fused View", 0) >= 0 : # Check if window exists by name
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()