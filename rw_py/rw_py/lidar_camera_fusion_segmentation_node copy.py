#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
import tf2_ros
import tf_transformations # For quaternion_matrix

class ROILidarFusionNode(Node):
    def __init__(self):
        super().__init__('roi_lidar_fusion_node')
        # --- Parameters ---
        self.declare_parameter('input_image_topic', 'camera/image')
        self.declare_parameter('input_pc_topic', 'scan_02/points')
        self.declare_parameter('output_window', 'Fused View')

        # Declare image dimensions first, as their defaults are used for ROI default calculations
        self.declare_parameter('img_w', 1920)
        self.declare_parameter('img_h', 1200)
        # Get the configured (or default) image dimensions.
        # These self.img_w_param and self.img_h_param will be the basis for calculations.
        self.img_w_param = self.get_parameter('img_w').get_parameter_value().integer_value
        self.img_h_param = self.get_parameter('img_h').get_parameter_value().integer_value

        # Calculate default pixel values for ROI based on percentages of the configured/default image size
        # For Y: y_min=19%, y_max=76% of self.img_h_param
        # For X: x_min=0%, x_max=100% of self.img_w_param (full width)
        default_roi_x_start = int(self.img_w_param * 0 / 100.0)    # Default: 0
        default_roi_y_start = int(self.img_h_param * 19 / 100.0)   # Default: 1200 * 0.19 = 228
        default_roi_x_end = int(self.img_w_param * 100 / 100.0)  # Default: self.img_w_param (e.g., 1920)
        default_roi_y_end = int(self.img_h_param * 76 / 100.0)   # Default: 1200 * 0.76 = 912

        # Now declare the ROI parameters with these calculated defaults
        self.declare_parameter('roi_x_start', default_roi_x_start)
        self.declare_parameter('roi_y_start', default_roi_y_start)
        self.declare_parameter('roi_x_end', default_roi_x_end)
        self.declare_parameter('roi_y_end', default_roi_y_end)

        self.declare_parameter('enable_black_segmentation', True)
        for name, default in [('black_h_min', 0), ('black_s_min', 0), ('black_v_min', 0),
                              ('black_h_max', 180), ('black_s_max', 255), ('black_v_max', 50)]:
            self.declare_parameter(name, default)

        self.declare_parameter('point_display_mode', 1)

        self.declare_parameter('hfov', 1.25)
        self.declare_parameter('min_dist', 1.5)
        self.declare_parameter('max_dist', 5.0)
        self.declare_parameter('point_radius', 2)
        self.declare_parameter('colormap', cv2.COLORMAP_JET)
        self.declare_parameter('target_frame', 'front_camera_link_optical')
        self.declare_parameter('source_frame', 'front_lidar_link_optical')
        self.declare_parameter('static_transform_lookup_timeout_sec', 5.0)

        # Get parameters
        it = self.get_parameter('input_image_topic').get_parameter_value().string_value
        pt = self.get_parameter('input_pc_topic').get_parameter_value().string_value

        # Get the ROI parameters (they will have the calculated defaults if not overridden)
        self.roi_x_start = self.get_parameter('roi_x_start').get_parameter_value().integer_value
        self.roi_y_start = self.get_parameter('roi_y_start').get_parameter_value().integer_value
        self.roi_x_end = self.get_parameter('roi_x_end').get_parameter_value().integer_value
        self.roi_y_end = self.get_parameter('roi_y_end').get_parameter_value().integer_value

        # Validate and clamp ROI coordinates against actual image dimensions (self.img_w_param, self.img_h_param)
        self.roi_x_start = max(0, self.roi_x_start)
        self.roi_y_start = max(0, self.roi_y_start)
        self.roi_x_end = min(self.img_w_param, self.roi_x_end)
        self.roi_y_end = min(self.img_h_param, self.roi_y_end)

        if not (self.roi_x_start < self.roi_x_end and self.roi_y_start < self.roi_y_end):
            self.get_logger().error(
                f"Invalid ROI coordinates after clamping or due to override: "
                f"x_start={self.roi_x_start}, y_start={self.roi_y_start}, "
                f"x_end={self.roi_x_end}, y_end={self.roi_y_end} "
                f"for image {self.img_w_param}x{self.img_h_param}. "
                f"Defaulting ROI to full image."
            )
            self.roi_x_start = 0; self.roi_y_start = 0
            self.roi_x_end = self.img_w_param; self.roi_y_end = self.img_h_param
        self.get_logger().info(f"Using ROI: x=[{self.roi_x_start},{self.roi_x_end}), y=[{self.roi_y_start},{self.roi_y_end}) on {self.img_w_param}x{self.img_h_param} image")

        self.enable_seg = self.get_parameter('enable_black_segmentation').get_parameter_value().bool_value
        self.h_min = self.get_parameter('black_h_min').get_parameter_value().integer_value
        self.s_min = self.get_parameter('black_s_min').get_parameter_value().integer_value
        self.v_min = self.get_parameter('black_v_min').get_parameter_value().integer_value
        self.h_max = self.get_parameter('black_h_max').get_parameter_value().integer_value
        self.s_max = self.get_parameter('black_s_max').get_parameter_value().integer_value
        self.v_max = self.get_parameter('black_v_max').get_parameter_value().integer_value

        self.point_display_mode = self.get_parameter('point_display_mode').get_parameter_value().integer_value
        if self.point_display_mode not in [0, 1, 2]:
            self.get_logger().warn(f"Invalid point_display_mode: {self.point_display_mode}. Defaulting to 2 (All Projected).")
            self.point_display_mode = 2
        self.get_logger().info(f"Point display mode: {self.point_display_mode} (0:None, 1:Selected, 2:All)")


        hfov = self.get_parameter('hfov').get_parameter_value().double_value
        self.min_dist = self.get_parameter('min_dist').get_parameter_value().double_value
        self.max_dist = self.get_parameter('max_dist').get_parameter_value().double_value
        self.point_radius = self.get_parameter('point_radius').get_parameter_value().integer_value
        self.colormap = self.get_parameter('colormap').get_parameter_value().integer_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.source_frame = self.get_parameter('source_frame').get_parameter_value().string_value
        self.static_transform_lookup_timeout_sec = self.get_parameter('static_transform_lookup_timeout_sec').get_parameter_value().double_value

        # Camera intrinsics are based on self.img_w_param, self.img_h_param
        self.fx = self.img_w_param / (2 * math.tan(hfov/2))
        self.fy = self.fx
        self.cx = self.img_w_param / 2.0
        self.cy = self.img_h_param / 2.0
        self.get_logger().info(f"Camera Intrinsics (for {self.img_w_param}x{self.img_h_param} image): fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_pc = None
        self.create_subscription(Image, it, self.image_cb, rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.create_subscription(PointCloud2, pt, self.pc_cb, rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_transform_matrix = None

        self.window = self.get_parameter('output_window').get_parameter_value().string_value
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 800, 600)

        self.create_timer(1.0/30.0, self.timer_cb)
        self.get_logger().info('ROI-LiDAR Fusion Node started')

    def image_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if img.shape[1] != self.img_w_param or img.shape[0] != self.img_h_param:
                self.latest_image = cv2.resize(img, (self.img_w_param, self.img_h_param))
            else:
                self.latest_image = img
        except CvBridgeError as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def pc_cb(self, msg: PointCloud2):
        self.latest_pc = msg

    def crop_and_segment(self, img):
        # img is already resized to self.img_w_param x self.img_h_param
        # self.roi_x_start etc. are already validated pixel coordinates
        roi = img[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        segmentation_mask_roi = None
        if self.enable_seg:
            if roi.size == 0: # Should not happen if ROI validation is correct
                self.get_logger().warn("ROI is empty during crop_and_segment, cannot perform segmentation.")
            else:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                lower = np.array([self.h_min, self.s_min, self.v_min])
                upper = np.array([self.h_max, self.s_max, self.v_max])
                segmentation_mask_roi = cv2.inRange(hsv, lower, upper)
        offset = (self.roi_x_start, self.roi_y_start)
        return roi, segmentation_mask_roi, offset

    def _get_static_transform(self):
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                self.target_frame, self.source_frame, Time(),
                timeout=Duration(seconds=self.static_transform_lookup_timeout_sec)
            )
            q = transform_stamped.transform.rotation; t = transform_stamped.transform.translation
            T_mat = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            T_mat[:3, 3] = [t.x, t.y, t.z]
            self.static_transform_matrix = T_mat
            self.get_logger().info(f"Successfully fetched static transform from {self.source_frame} to {self.target_frame}")
        except Exception as e:
            self.get_logger().warn(f"Could not get static transform: {e}. Will retry.")
            self.static_transform_matrix = None

    def project_and_color(self, base_img, segmentation_mask_roi, roi_offset):
        if self.latest_pc is None: return base_img
        if self.static_transform_matrix is None: return base_img

        pts_raw = point_cloud2.read_points_numpy(self.latest_pc, field_names=('x','y','z'), skip_nans=True)
        if pts_raw.size == 0: return base_img

        pts = np.vstack((pts_raw['x'], pts_raw['y'], pts_raw['z'])).T if pts_raw.dtype.names else pts_raw.reshape(-1,3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] == 0: return base_img

        pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
        cam_coords = (self.static_transform_matrix @ pts_h.T).T[:, :3]

        X_cam_proj, Y_cam_proj, Z_cam_depth = -cam_coords[:,1], -cam_coords[:,2], cam_coords[:,0]
        valid_depth_mask = Z_cam_depth > 0.01
        X_cam_proj = X_cam_proj[valid_depth_mask]; Y_cam_proj = Y_cam_proj[valid_depth_mask]; Z_cam_depth = Z_cam_depth[valid_depth_mask]
        original_pts_filtered = pts[valid_depth_mask]
        if X_cam_proj.size == 0: return base_img

        distances = np.linalg.norm(original_pts_filtered, axis=1)
        norm_dist = np.clip((distances - self.min_dist) / (self.max_dist - self.min_dist), 0, 1)
        intensity = ((1.0 - norm_dist) * 255).astype(np.uint8)
        colors_map = cv2.applyColorMap(intensity.reshape(-1, 1), self.colormap).reshape(-1, 3)

        u_px = (self.fx * X_cam_proj / Z_cam_depth + self.cx).astype(int)
        v_px = (self.fy * Y_cam_proj / Z_cam_depth + self.cy).astype(int)

        valid_proj_mask = (u_px >= 0) & (u_px < self.img_w_param) & (v_px >= 0) & (v_px < self.img_h_param)
        u_px_valid = u_px[valid_proj_mask]; v_px_valid = v_px[valid_proj_mask]; colors_valid = colors_map[valid_proj_mask]
        if u_px_valid.size == 0: return base_img

        fused_img = base_img.copy()
        roi_x_abs_offset, roi_y_abs_offset = roi_offset

        for idx in range(len(u_px_valid)):
            img_u, img_v = u_px_valid[idx], v_px_valid[idx]
            roi_u, roi_v = img_u - roi_x_abs_offset, img_v - roi_y_abs_offset

            is_in_segmented_region = False
            if self.enable_seg and segmentation_mask_roi is not None and \
               0 <= roi_v < segmentation_mask_roi.shape[0] and \
               0 <= roi_u < segmentation_mask_roi.shape[1] and \
               segmentation_mask_roi[roi_v, roi_u] > 0:
                is_in_segmented_region = True

            draw_this_point = False
            point_color_to_draw = tuple(map(int, colors_valid[idx]))

            if self.point_display_mode == 2:
                draw_this_point = True
                if is_in_segmented_region: point_color_to_draw = (0, 0, 255)
            elif self.point_display_mode == 1:
                if is_in_segmented_region:
                    draw_this_point = True
                    point_color_to_draw = (0, 0, 255)

            if draw_this_point:
                for dx_p in range(-self.point_radius, self.point_radius + 1):
                    for dy_p in range(-self.point_radius, self.point_radius + 1):
                        draw_x = np.clip(img_u + dx_p, 0, self.img_w_param - 1)
                        draw_y = np.clip(img_v + dy_p, 0, self.img_h_param - 1)
                        fused_img[draw_y, draw_x] = point_color_to_draw
        return fused_img

    def timer_cb(self):
        if self.latest_image is None: return

        if self.static_transform_matrix is None:
            self._get_static_transform()
            if self.static_transform_matrix is None:
                self.get_logger().info("Waiting for static transform...")
                display_img_resized = cv2.resize(self.latest_image, (800,600))
                cv2.imshow(self.window, display_img_resized)
                cv2.waitKey(1)
                return

        roi_image_content, segmentation_mask_for_roi, roi_absolute_offset = self.crop_and_segment(self.latest_image)

        segmentation_found_something = False
        if self.enable_seg and segmentation_mask_for_roi is not None and cv2.countNonZero(segmentation_mask_for_roi) > 0:
            segmentation_found_something = True

        output_display_image = self.latest_image.copy()

        should_process_and_project_pc = False
        if self.point_display_mode == 2: should_process_and_project_pc = True
        elif self.point_display_mode == 1:
            if segmentation_found_something: should_process_and_project_pc = True

        if should_process_and_project_pc:
            if self.latest_pc is not None:
                output_display_image = self.project_and_color(
                    self.latest_image, segmentation_mask_for_roi, roi_absolute_offset)
            # else: self.get_logger().debug("Point cloud display requested, but no PC data yet.")


        if self.enable_seg :
            roi_color = (0, 255, 0) if segmentation_found_something else (255, 191, 0) # Green if found, light blue otherwise
            
            # Draw top horizontal line of ROI
            cv2.line(output_display_image, 
                     (self.roi_x_start, self.roi_y_start),  # Start point (left)
                     (self.roi_x_end -1 , self.roi_y_start),  # End point (right)
                     roi_color, 1)
            
            # Draw bottom horizontal line of ROI
            # Note: y_end is exclusive in slicing, so for drawing it's the coordinate of the line
            # If roi_y_end is the height, then roi_y_end - 1 is the last drawable row index
            y_bottom_line = self.roi_y_end -1 
            if y_bottom_line >= self.roi_y_start: # Ensure bottom line is not above or at top line
                cv2.line(output_display_image,
                         (self.roi_x_start, y_bottom_line), # Start point (left)
                         (self.roi_x_end -1, y_bottom_line), # End point (right)
                         roi_color, 1)

        display_img_resized = cv2.resize(output_display_image, (800,600))
        cv2.imshow(self.window, display_img_resized)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Shutdown requested')
            self.destroy_node()
            rclpy.shutdown()
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = ROILidarFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down...')
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()