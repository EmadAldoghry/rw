import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import tf2_ros
# import tf_transformations # Not strictly used for quaternion math here
from geometry_msgs.msg import PointStamped # For tf2_geometry_msgs
import tf2_geometry_msgs # For do_transform_point

class ImageLidarFusionNode(Node):
    def __init__(self):
        super().__init__('image_lidar_fusion')

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
            self.get_logger().info("Declared 'use_sim_time' parameter with default False.")
        
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value

        if self.use_sim_time:
            self.get_logger().info("use_sim_time is TRUE, node configured to use simulation time.")
        else:
            self.get_logger().warn("use_sim_time is FALSE. If rest of system (especially TF) uses sim time, TF will likely fail or give incorrect transforms!")

        self.image_sub = self.create_subscription(Image, 'camera/image', self.image_callback, 10)
        self.pc_sub = self.create_subscription(PointCloud2, 'scan_02/points', self.pc_callback, 10)

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_pc_msg = None

        self.img_width = 1920
        self.img_height = 1200
        hfov = 1.25
        self.fx = self.img_width / (2 * math.tan(hfov / 2))
        self.fy = self.fx
        self.cx = self.img_width / 2.0
        self.cy = self.img_height / 2.0
        self.get_logger().info(f"Camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.target_frame = 'front_camera_link_optical'
        self.expected_lidar_frame = 'front_lidar_link_optical'

        # --- NEW: Parameters for distance-based coloring ---
        self.min_render_dist = 1.5  # meters, points closer than this get min_color
        self.max_render_dist = 5.0 # meters, points further than this get max_color
        # Colormap: JET makes closer points appear "hotter" (red) and far points "cooler" (blue)
        # when mapping (1.0 - normalized_distance).
        # Other options: cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_MAGMA, cv2.COLORMAP_INFERNO, cv2.COLORMAP_PLASMA, cv2.COLORMAP_HSV
        self.colormap = cv2.COLORMAP_JET
        self.get_logger().info(f"Coloring LiDAR points from {self.min_render_dist}m to {self.max_render_dist}m using colormap ID: {self.colormap}")
        # --- END NEW ---

        for wn in ['Camera Feed', 'LiDAR View', 'Fused View']:
            cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(wn, 800, 600)

    def image_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if img.shape[1] != self.img_width or img.shape[0] != self.img_height:
                self.latest_image = cv2.resize(img, (self.img_width, self.img_height))
            else:
                self.latest_image = img
        except Exception as e:
            self.get_logger().error(f'Image convert error: {e}')

    def pc_callback(self, msg: PointCloud2):
        if msg.header.frame_id != self.expected_lidar_frame:
            self.get_logger().error(
                f"PointCloud frame_id '{msg.header.frame_id}' does not match expected lidar frame '{self.expected_lidar_frame}'!"
                " TF lookups might be incorrect if not properly chained."
            )
        self.latest_pc_msg = msg

    def project_points_to_image(self):
        if self.latest_image is None:
            self.get_logger().warn('No latest image for projection.')
            return None
        
        fused_image = self.latest_image.copy()

        if self.latest_pc_msg is None:
            self.get_logger().warn('No latest_pc_msg for projection.')
            return fused_image

        points_to_project = list(point_cloud2.read_points(self.latest_pc_msg, field_names=('x', 'y', 'z'), skip_nans=True))
        
        if not points_to_project:
            self.get_logger().info('Point cloud is empty after reading points.')
            return fused_image

        drawn_count = 0
        transform_available = False
        
        source_frame_from_pc = self.latest_pc_msg.header.frame_id
        pc_timestamp = self.latest_pc_msg.header.stamp

        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame_from_pc,
                pc_timestamp,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            transform_available = True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f"TF Exception for transform from '{source_frame_from_pc}' to '{self.target_frame}' "
                f"at time {pc_timestamp.sec}.{pc_timestamp.nanosec}: {e}"
            )
            return fused_image

        total_points = len(points_to_project)
        points_behind_camera = 0
        points_outside_fov_xy = 0
        
        for idx, (x, y, z) in enumerate(points_to_project):
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue

            # --- NEW: Calculate distance for coloring ---
            # This is the distance from the LiDAR sensor origin
            dist_from_lidar = math.sqrt(x**2 + y**2 + z**2)

            # Normalize distance to 0-1 range for colormap
            if dist_from_lidar <= self.min_render_dist:
                normalized_dist = 0.0
            elif dist_from_lidar >= self.max_render_dist:
                normalized_dist = 1.0
            else:
                normalized_dist = (dist_from_lidar - self.min_render_dist) / \
                                  (self.max_render_dist - self.min_render_dist)
            
            # To make closer points "hotter" (e.g., red in JET) and far points "cooler" (e.g., blue in JET)
            # we map `(1.0 - normalized_dist)`. If you want the opposite, use `normalized_dist`.
            # The colormap input is an 8-bit value (0-255).
            value_for_colormap = int((1.0 - normalized_dist) * 255)
            
            # Apply colormap
            # cv2.applyColorMap expects a single channel 8-bit image.
            # We create a 1x1 pixel image with our value.
            color_bgr_array = cv2.applyColorMap(np.array([[value_for_colormap]], dtype=np.uint8), self.colormap)
            # Extract the BGR tuple from the 1x1x3 array
            point_color_bgr = (int(color_bgr_array[0,0,0]), int(color_bgr_array[0,0,1]), int(color_bgr_array[0,0,2]))
            # --- END NEW ---

            point_in_source_frame = PointStamped()
            point_in_source_frame.header.stamp = pc_timestamp 
            point_in_source_frame.header.frame_id = source_frame_from_pc
            point_in_source_frame.point.x = float(x)
            point_in_source_frame.point.y = float(y)
            point_in_source_frame.point.z = float(z)

            try:
                point_in_target_frame_msg = tf2_geometry_msgs.do_transform_point(point_in_source_frame, transform_stamped)
                
                pt_x_cam_body = point_in_target_frame_msg.point.x
                pt_y_cam_body = point_in_target_frame_msg.point.y
                pt_z_cam_body = point_in_target_frame_msg.point.z

                px_optical = -pt_y_cam_body
                py_optical = -pt_z_cam_body
                pz_optical =  pt_x_cam_body

                if idx < 3: # Log details for the first 3 points
                    self.get_logger().info(
                        f"Pt {idx}: LiDAR({x:.2f},{y:.2f},{z:.2f}) Dist:{dist_from_lidar:.2f}m "
                        f"-> CamBody({pt_x_cam_body:.2f},{pt_y_cam_body:.2f},{pt_z_cam_body:.2f}) "
                        f"-> ProjCoords(Xopt:{px_optical:.2f},Yopt:{py_optical:.2f},Zopt:{pz_optical:.2f}) "
                        f"-> ColorValue:{value_for_colormap} ColorBGR:{point_color_bgr}"
                    )

                if pz_optical <= 0.01:
                    points_behind_camera += 1
                    continue
                
                u = int(self.fx * px_optical / pz_optical + self.cx)
                v = int(self.fy * py_optical / pz_optical + self.cy)

                if 0 <= u < self.img_width and 0 <= v < self.img_height:
                    # --- MODIFIED: Use dynamic color ---
                    cv2.circle(fused_image, (u, v), 3, point_color_bgr, -1)
                    # --- END MODIFIED ---
                    drawn_count += 1
                else:
                    points_outside_fov_xy +=1

            except Exception as e_transform_point:
                self.get_logger().error(f"Error transforming/projecting point {idx}: {e_transform_point}")
                continue

        self.get_logger().info(
            f"Projection Summary: Total={total_points}, Drawn={drawn_count}, "
            f"BehindCam={points_behind_camera}, OutsideXY_FOV={points_outside_fov_xy}, "
            f"TF_Available={transform_available}"
        )
        return fused_image

    def display(self):
        loop_count = 0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            
            cam_display = self.latest_image if self.latest_image is not None else \
                          np.zeros((self.img_height, self.img_width, 3), np.uint8)
            cv2.imshow('Camera Feed', cv2.resize(cam_display, (800, 600)))

            lidar_display = np.zeros((600, 600, 3), np.uint8)
            if self.latest_pc_msg:
                points_for_lidar_view = list(point_cloud2.read_points(
                    self.latest_pc_msg, field_names=('x', 'y', 'z'), skip_nans=True
                ))
                for lx, ly, lz in points_for_lidar_view:
                    if not (math.isfinite(lx) and math.isfinite(ly) and math.isfinite(lz)):
                        continue
                    
                    # Color LiDAR view points by distance too (optional, but nice)
                    dist_lidar_view = math.sqrt(lx**2 + ly**2 + lz**2)
                    if dist_lidar_view <= self.min_render_dist: norm_dist_lv = 0.0
                    elif dist_lidar_view >= self.max_render_dist: norm_dist_lv = 1.0
                    else: norm_dist_lv = (dist_lidar_view - self.min_render_dist) / (self.max_render_dist - self.min_render_dist)
                    val_lv = int((1.0 - norm_dist_lv) * 255)
                    color_lv_arr = cv2.applyColorMap(np.array([[val_lv]], dtype=np.uint8), self.colormap)
                    color_lv_bgr = (int(color_lv_arr[0,0,0]), int(color_lv_arr[0,0,1]), int(color_lv_arr[0,0,2]))

                    px_lidar_view = int(300 - ly * 10) 
                    py_lidar_view = int(300 - lx * 10) 
                    if 0 <= px_lidar_view < 600 and 0 <= py_lidar_view < 600:
                        cv2.circle(lidar_display, (px_lidar_view, py_lidar_view), 1, color_lv_bgr, -1) # Use new color
            cv2.imshow('LiDAR View', lidar_display)

            if self.latest_image is not None and self.latest_pc_msg is not None:
                fused_display = self.project_points_to_image()
                if fused_display is None: # Should not happen if latest_image is not None
                    fused_display = self.latest_image.copy() if self.latest_image is not None else np.zeros((self.img_height, self.img_width, 3), np.uint8)
            else:
                fused_display = self.latest_image.copy() if self.latest_image is not None else np.zeros((self.img_height, self.img_width, 3), np.uint8)

            # Draw a simple color bar legend (optional)
            legend_height = 50
            legend_width = 200
            legend_bar = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
            for i in range(legend_width):
                val = int((i / legend_width) * 255) # In JET, 0 is blue (far), 255 is red (close)
                color = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), self.colormap)[0,0]
                legend_bar[:, i] = color
            
            # Flip legend if using (1.0 - normalized_dist) for mapping, so red is on left (close)
            legend_bar = cv2.flip(legend_bar, 1) # Flip horizontally

            # Resize fused_display to 800x600 before overlaying legend
            fused_display_resized = cv2.resize(fused_display, (800,600))
            
            # Put legend on the display
            start_x = 10
            start_y = fused_display_resized.shape[0] - legend_height - 10 # 10px from bottom
            if start_y < 0: start_y = 10 # Ensure it's on screen

            try:
                fused_display_resized[start_y:start_y+legend_height, start_x:start_x+legend_width] = legend_bar
                cv2.putText(fused_display_resized, f"{self.min_render_dist:.1f}m", (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(fused_display_resized, f"{self.max_render_dist:.1f}m", (start_x + legend_width - 50, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            except Exception as e:
                self.get_logger().warn(f"Could not draw legend: {e}")

            cv2.imshow('Fused View', fused_display_resized)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            loop_count +=1
            if loop_count % 200 == 0: 
                self.get_logger().debug(f"Display loop alive. Count: {loop_count}")

        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = ImageLidarFusionNode()
    node.get_logger().info('Fusion node started by main(). Waiting for data...')
    try:
        node.display()
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()