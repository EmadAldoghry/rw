#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 # For read_points_numpy and create_cloud_xyz32
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
import tf2_ros
import tf_transformations # For quaternion_matrix
# from std_msgs.msg import Header # Not strictly needed if we copy and modify existing header

class ROILidarFusionNode(Node):
    def __init__(self):
        super().__init__('roi_lidar_fusion_node')
        # --- Parameters ---
        self.declare_parameter('input_image_topic', 'camera/image')
        self.declare_parameter('input_pc_topic', 'scan_02/points')
        self.declare_parameter('output_window', 'Fused View')
        self.declare_parameter('output_selected_pc_topic', 'selected_lidar_points') # New parameter

        # Declare image dimensions first, as their defaults are used for ROI default calculations
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
                              ('black_h_max', 180), ('black_s_max', 255), ('black_v_max', 0)]:
            self.declare_parameter(name, default)

        self.declare_parameter('point_display_mode', 2)
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
        output_selected_pc_topic_name = self.get_parameter('output_selected_pc_topic').get_parameter_value().string_value

        self.roi_x_start = self.get_parameter('roi_x_start').get_parameter_value().integer_value
        self.roi_y_start = self.get_parameter('roi_y_start').get_parameter_value().integer_value
        self.roi_x_end = self.get_parameter('roi_x_end').get_parameter_value().integer_value
        self.roi_y_end = self.get_parameter('roi_y_end').get_parameter_value().integer_value

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

        # New publisher for selected points
        self.selected_points_publisher = self.create_publisher(PointCloud2, output_selected_pc_topic_name, 10)
        self.get_logger().info(f"Publishing selected LiDAR points to: {output_selected_pc_topic_name}")


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
        roi = img[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        segmentation_mask_roi = None
        if self.enable_seg:
            if roi.size == 0:
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
        selected_points_for_publishing = [] # List to store [x,y,z] of selected points
        
        if self.latest_pc is None: return base_img, selected_points_for_publishing
        if self.static_transform_matrix is None: return base_img, selected_points_for_publishing

        pts_raw = point_cloud2.read_points_numpy(self.latest_pc, field_names=('x','y','z'), skip_nans=True)
        if pts_raw.size == 0: return base_img, selected_points_for_publishing

        # pts are in original LiDAR frame (self.source_frame)
        pts = np.vstack((pts_raw['x'], pts_raw['y'], pts_raw['z'])).T if pts_raw.dtype.names else pts_raw.reshape(-1,3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] == 0: return base_img, selected_points_for_publishing

        pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
        # cam_coords are in camera optical frame (self.target_frame)
        cam_coords = (self.static_transform_matrix @ pts_h.T).T[:, :3]

        # Standard camera projection: X right, Y down, Z forward
        # LiDAR points might be X forward, Y left, Z up (ROS standard)
        # Transformation to camera optical: X right, Y down, Z forward from camera perspective
        # Assuming cam_coords[:,0] is X_cam, cam_coords[:,1] is Y_cam, cam_coords[:,2] is Z_cam (depth)
        # If your camera is front_camera_link_optical (X right, Y down, Z forward):
        # X_cam_proj = cam_coords[:,0]
        # Y_cam_proj = cam_coords[:,1]
        # Z_cam_depth = cam_coords[:,2]
        # The original code had: X_cam_proj, Y_cam_proj, Z_cam_depth = -cam_coords[:,1], -cam_coords[:,2], cam_coords[:,0]
        # This implies a coordinate system mismatch or a non-standard optical frame definition after transformation.
        # Let's stick to the original projection logic as it was working for visualization.
        # If front_lidar_link_optical is X right, Y down, Z forward (like a camera)
        # and front_camera_link_optical is also X right, Y down, Z forward
        # then cam_coords should be directly usable.
        # The negation and swapping suggests cam_coords are not yet in the desired X-right, Y-down, Z-forward.
        # E.g. if after transform, cam_coords are (X_lidar_in_cam_frame, Y_lidar_in_cam_frame, Z_lidar_in_cam_frame)
        # and these correspond to (X_fwd, Y_left, Z_up) from camera's perspective, then:
        # Z_cam_depth (true depth) = X_lidar_in_cam_frame  (cam_coords[:,0])
        # X_cam_proj (image u-coord direction) = -Y_lidar_in_cam_frame (-cam_coords[:,1])
        # Y_cam_proj (image v-coord direction) = -Z_lidar_in_cam_frame (-cam_coords[:,2])
        X_cam_proj, Y_cam_proj, Z_cam_depth = -cam_coords[:,1], -cam_coords[:,2], cam_coords[:,0]

        valid_depth_mask = Z_cam_depth > 0.01 # Points in front of camera
        
        X_cam_proj_vd = X_cam_proj[valid_depth_mask]
        Y_cam_proj_vd = Y_cam_proj[valid_depth_mask]
        Z_cam_depth_vd = Z_cam_depth[valid_depth_mask]
        original_pts_after_depth_filter = pts[valid_depth_mask] # Original LiDAR coordinates for valid depth points

        if X_cam_proj_vd.size == 0: return base_img, selected_points_for_publishing

        distances = np.linalg.norm(original_pts_after_depth_filter, axis=1) # Distance from LiDAR origin
        norm_dist = np.clip((distances - self.min_dist) / (self.max_dist - self.min_dist), 0, 1)
        intensity = ((1.0 - norm_dist) * 255).astype(np.uint8)
        colors_map = cv2.applyColorMap(intensity.reshape(-1, 1), self.colormap).reshape(-1, 3)

        u_px = (self.fx * X_cam_proj_vd / Z_cam_depth_vd + self.cx).astype(int)
        v_px = (self.fy * Y_cam_proj_vd / Z_cam_depth_vd + self.cy).astype(int)

        # Filter points that project within image bounds
        valid_proj_mask = (u_px >= 0) & (u_px < self.img_w_param) & \
                          (v_px >= 0) & (v_px < self.img_h_param)
        
        u_px_valid = u_px[valid_proj_mask]
        v_px_valid = v_px[valid_proj_mask]
        colors_valid = colors_map[valid_proj_mask]
        # These are the original LiDAR coordinates of points that project validly onto the image
        original_pts_for_valid_pixels = original_pts_after_depth_filter[valid_proj_mask]

        if u_px_valid.size == 0: return base_img, selected_points_for_publishing

        fused_img = base_img.copy()
        roi_x_abs_offset, roi_y_abs_offset = roi_offset

        for idx in range(len(u_px_valid)):
            img_u, img_v = u_px_valid[idx], v_px_valid[idx]
            # Convert pixel coordinates to be relative to the ROI
            roi_u, roi_v = img_u - roi_x_abs_offset, img_v - roi_y_abs_offset
            
            current_original_3d_point = list(original_pts_for_valid_pixels[idx])


            is_in_segmented_region = False
            if self.enable_seg and segmentation_mask_roi is not None and \
               0 <= roi_v < segmentation_mask_roi.shape[0] and \
               0 <= roi_u < segmentation_mask_roi.shape[1] and \
               segmentation_mask_roi[roi_v, roi_u] > 0:
                is_in_segmented_region = True
                # This point is in the segmented region, collect its original 3D coordinates
                selected_points_for_publishing.append(current_original_3d_point)


            # Drawing logic for visualization
            draw_this_point = False
            point_color_to_draw = tuple(map(int, colors_valid[idx]))

            if self.point_display_mode == 2: # Display all projected points
                draw_this_point = True
                if is_in_segmented_region: # If also in segmented region, color it red
                    point_color_to_draw = (0, 0, 255) 
            elif self.point_display_mode == 1: # Display only selected (segmented) points
                if is_in_segmented_region:
                    draw_this_point = True
                    point_color_to_draw = (0, 0, 255) # Color selected points red

            if draw_this_point:
                for dx_p in range(-self.point_radius, self.point_radius + 1):
                    for dy_p in range(-self.point_radius, self.point_radius + 1):
                        draw_x = np.clip(img_u + dx_p, 0, self.img_w_param - 1)
                        draw_y = np.clip(img_v + dy_p, 0, self.img_h_param - 1)
                        fused_img[draw_y, draw_x] = point_color_to_draw
                        
        return fused_img, selected_points_for_publishing

    def timer_cb(self):
        if self.latest_image is None: return

        if self.static_transform_matrix is None:
            self._get_static_transform()
            if self.static_transform_matrix is None:
                self.get_logger().info("Waiting for static transform...")
                display_img_resized = cv2.resize(self.latest_image, (800,600))
                cv2.putText(display_img_resized, "Waiting for TF transform...", (50,50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow(self.window, display_img_resized)
                cv2.waitKey(1)
                return

        # --- Image Processing ---
        # roi_image_content is the cropped part of the image, segmentation_mask_for_roi is for that crop
        roi_image_content, segmentation_mask_for_roi, roi_absolute_offset = self.crop_and_segment(self.latest_image)

        segmentation_found_something = False
        if self.enable_seg and segmentation_mask_for_roi is not None and cv2.countNonZero(segmentation_mask_for_roi) > 0:
            segmentation_found_something = True

        # --- Point Cloud Processing and Fusion ---
        output_display_image = self.latest_image.copy()
        
        if self.latest_pc is not None:
            # project_and_color will draw points on output_display_image based on point_display_mode
            # and will return points_to_publish (original 3D coords in LiDAR frame) if they fall in segmented area
            processed_image_with_points, points_to_publish = self.project_and_color(
                output_display_image, # Pass the current image to draw on
                segmentation_mask_for_roi,
                roi_absolute_offset
            )
            output_display_image = processed_image_with_points # Update image with drawn points

            # Publish the selected points if segmentation is enabled and points were found
            if self.enable_seg and points_to_publish:
                # Create a new PointCloud2 message.
                # The point_cloud2.create_cloud_xyz32 function will populate most of it.
                # We need to provide it with a header.
                # This header should use the frame_id and stamp of the LATEST incoming PC,
                # because the data we are publishing is derived from that specific PC message.

                header_for_selected_points = self.latest_pc.header # This copies the stamp and frame_id

                # The points_to_publish are in the self.latest_pc.header.frame_id
                # (which is front_lidar_link_optical)
                final_selected_cloud_msg = point_cloud2.create_cloud_xyz32(
                    header_for_selected_points,
                    points_to_publish
                )
                self.selected_points_publisher.publish(final_selected_cloud_msg)
        else:
            # self.get_logger().debug("No point cloud data yet to process for projection/publishing.")
            pass # output_display_image remains self.latest_image.copy()


        # --- Drawing ROI and Display ---
        if self.enable_seg : # Only draw ROI if segmentation is enabled, or always? Let's say always for visualization
            roi_color = (0, 255, 0) if segmentation_found_something else (255, 191, 0) # Green if found, light blue otherwise
            
            cv2.line(output_display_image, 
                     (self.roi_x_start, self.roi_y_start),
                     (self.roi_x_end -1 , self.roi_y_start),
                     roi_color, 1)
            
            y_bottom_line = self.roi_y_end -1 
            if y_bottom_line >= self.roi_y_start:
                cv2.line(output_display_image,
                         (self.roi_x_start, y_bottom_line),
                         (self.roi_x_end -1, y_bottom_line),
                         roi_color, 1)
            # Draw vertical lines for ROI too
            cv2.line(output_display_image,
                     (self.roi_x_start, self.roi_y_start),
                     (self.roi_x_start, self.roi_y_end -1),
                     roi_color, 1)
            cv2.line(output_display_image,
                     (self.roi_x_end -1, self.roi_y_start),
                     (self.roi_x_end -1, self.roi_y_end -1),
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
        if rclpy.ok(): # Check if rclpy context is still valid
            if node.is_valid(): # Check if node is still valid before destroying
                 node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()