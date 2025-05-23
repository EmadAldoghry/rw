import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import math
import traceback

class ImageAndLidarSubscriber(Node):
    def __init__(self):
        super().__init__('image_lidar_subscriber')

        self.image_subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            1
        )
        self.bridge = CvBridge()
        self.latest_frame = None

        self.pointcloud_subscription = self.create_subscription(
            PointCloud2,
            'scan_02/points',
            self.pointcloud_callback,
            10
        )
        self.lidar_visualization_image = None
        self.current_view_mode = "perspective_front_view" # Default to new view

        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # --- Parameters for Top-Down View ---
        self.TOP_DOWN_VIS_SIZE = 600
        self.TOP_DOWN_MAX_VIEW_METERS = 10.0

        # --- Parameters for Perspective Front View ---
        self.PERSPECTIVE_VIEW_WIDTH = 800
        self.PERSPECTIVE_VIEW_HEIGHT = 600
        # Virtual "focal length" or distance to the projection plane.
        # Adjust this to change the "zoom" or field of view effect.
        # Larger f_x, f_y means narrower FOV (more zoom).
        self.PERSPECTIVE_FOCAL_X = 400.0 # pixels
        self.PERSPECTIVE_FOCAL_Y = 400.0 # pixels
        # Max depth for coloring and consideration
        self.PERSPECTIVE_MAX_DEPTH_METERS = 20.0
        self.DISTANCE_COLORMAP = cv2.COLORMAP_JET
        self.MIN_POINT_DRAW_DISTANCE = 0.1 # Don't draw points too close (X < this value)

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}\n{traceback.format_exc()}")

    def pointcloud_callback(self, pc_msg: PointCloud2):
        points_3d_lidar_frame = []
        # ... (rest of the point extraction logic remains the same as previous correct version)
        field_names = [field.name for field in pc_msg.fields]
        if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
            self.get_logger().error("PointCloud2 does not contain x, y, z fields.")
            # Create an empty image based on current mode
            if self.current_view_mode == "perspective_front_view":
                self.lidar_visualization_image = np.zeros((self.PERSPECTIVE_VIEW_HEIGHT, self.PERSPECTIVE_VIEW_WIDTH, 3), dtype=np.uint8)
            else: # top_down
                self.lidar_visualization_image = np.zeros((self.TOP_DOWN_VIS_SIZE, self.TOP_DOWN_VIS_SIZE, 3), dtype=np.uint8)
            cv2.putText(self.lidar_visualization_image, "No XYZ in PointCloud", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return

        num_points_processed = 0
        for point in point_cloud2.read_points(pc_msg, field_names=('x', 'y', 'z'), skip_nans=True):
            x_lidar, y_lidar, z_lidar = point[0], point[1], point[2]
            if math.isinf(x_lidar) or math.isinf(y_lidar) or math.isinf(z_lidar) or \
               not (math.isfinite(x_lidar) and math.isfinite(y_lidar) and math.isfinite(z_lidar)):
                continue
            points_3d_lidar_frame.append((x_lidar, y_lidar, z_lidar))
            num_points_processed += 1
        
        self.get_logger().debug(f"Processed {num_points_processed} valid points for visualization.")

        if self.current_view_mode == "perspective_front_view":
            self.lidar_visualization_image = self.process_pointcloud_data_perspective_front_view(points_3d_lidar_frame)
        elif self.current_view_mode == "top_down":
            self.lidar_visualization_image = self.process_pointcloud_data_top_down(points_3d_lidar_frame)
        # Add other view modes here if needed in the future

    def process_pointcloud_data_top_down(self, points_3d):
        # ... (this function remains the same as the previous correct version)
        vis_img = np.zeros((self.TOP_DOWN_VIS_SIZE, self.TOP_DOWN_VIS_SIZE, 3), dtype=np.uint8)
        center_x_img, center_y_img = self.TOP_DOWN_VIS_SIZE // 2, self.TOP_DOWN_VIS_SIZE // 2
        if self.TOP_DOWN_MAX_VIEW_METERS <= 1e-6: return vis_img # Should be handled in __init__
        pixels_per_meter = (self.TOP_DOWN_VIS_SIZE / 2.0) / self.TOP_DOWN_MAX_VIEW_METERS

        for x_lidar, y_lidar, z_lidar in points_3d:
            if abs(x_lidar) > self.TOP_DOWN_MAX_VIEW_METERS * 1.1 or \
               abs(y_lidar) > self.TOP_DOWN_MAX_VIEW_METERS * 1.1:
                continue
            try:
                img_x = int(center_x_img + y_lidar * pixels_per_meter) # y_lidar for horizontal
                img_y = int(center_y_img - x_lidar * pixels_per_meter) # x_lidar for vertical
            except OverflowError: continue
            if 0 <= img_x < self.TOP_DOWN_VIS_SIZE and 0 <= img_y < self.TOP_DOWN_VIS_SIZE:
                cv2.circle(vis_img, (img_x, img_y), 1, (0, 255, 0), -1)
        cv2.circle(vis_img, (center_x_img, center_y_img), 5, (0, 0, 255), -1) # LiDAR origin
        cv2.line(vis_img, (center_x_img, center_y_img),
                 (center_x_img, center_y_img - int(1.0 * pixels_per_meter)), # Forward X
                 (255, 0, 0), 2)
        return vis_img


    def process_pointcloud_data_perspective_front_view(self, points_3d):
        vis_img = np.zeros((self.PERSPECTIVE_VIEW_HEIGHT, self.PERSPECTIVE_VIEW_WIDTH, 3), dtype=np.uint8)
        
        # Optical center of the projection (center of the image)
        cx = self.PERSPECTIVE_VIEW_WIDTH / 2.0
        cy = self.PERSPECTIVE_VIEW_HEIGHT / 2.0

        # Sort points by distance (X_lidar) so closer points are drawn on top (occlusion)
        # Draw further points first by sorting X descending.
        points_to_draw = []
        for x, y, z in points_3d:
            # Only consider points in front of the LiDAR and within max depth
            if x > self.MIN_POINT_DRAW_DISTANCE and x <= self.PERSPECTIVE_MAX_DEPTH_METERS:
                points_to_draw.append((x, y, z))
        
        points_to_draw.sort(key=lambda p: p[0], reverse=True)

        for x_lidar, y_lidar, z_lidar in points_to_draw:
            # Perspective projection:
            # u = f_x * (Y_lidar / X_lidar) + c_x
            # v = f_y * (Z_lidar / X_lidar) + c_y
            # Note: LiDAR Y is often left, Z is up.
            # Image coordinates: u is horizontal (X), v is vertical (Y)
            # If we want positive Y_lidar (left) to map to the left side of the image (smaller u):
            #   u = c_x - f_x * (y_lidar / x_lidar)
            # If we want positive Z_lidar (up) to map to the upper side of the image (smaller v):
            #   v = c_y - f_y * (z_lidar / x_lidar)
            
            # Using standard camera model where X is right, Y is down from optical center:
            # img_x_from_center = self.PERSPECTIVE_FOCAL_X * (y_lidar / x_lidar)
            # img_y_from_center = self.PERSPECTIVE_FOCAL_Y * (z_lidar / x_lidar)
            # img_u = int(cx + img_x_from_center)
            # img_v = int(cy + img_y_from_center)

            # Let's adjust for LiDAR: X forward, Y left, Z up
            # Map LiDAR Y (left) to image horizontal: Positive Y (left) -> image left (smaller u)
            # Map LiDAR Z (up) to image vertical: Positive Z (up) -> image up (smaller v)
            img_u = int(cx - (self.PERSPECTIVE_FOCAL_X * y_lidar / x_lidar))
            img_v = int(cy - (self.PERSPECTIVE_FOCAL_Y * z_lidar / x_lidar))


            # --- Color by Distance (X_lidar) ---
            normalized_distance = np.clip(x_lidar / self.PERSPECTIVE_MAX_DEPTH_METERS, 0.0, 1.0)
            color_val_gray = int(normalized_distance * 255)
            gray_pixel = np.array([[color_val_gray]], dtype=np.uint8)
            colored_pixel = cv2.applyColorMap(gray_pixel, self.DISTANCE_COLORMAP)
            color = tuple(map(int, colored_pixel[0,0]))

            if 0 <= img_u < self.PERSPECTIVE_VIEW_WIDTH and 0 <= img_v < self.PERSPECTIVE_VIEW_HEIGHT:
                # Optionally, vary point size by distance (closer points larger)
                point_size = max(1, int(3 * (1 - normalized_distance * 0.8))) # Example scaling
                cv2.circle(vis_img, (img_u, img_v), point_size, color, -1)

        return vis_img

    def display_data(self):
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Feed", 800, 600)

        lidar_window_name = "LiDAR View"
        cv2.namedWindow(lidar_window_name, cv2.WINDOW_NORMAL)
        if self.current_view_mode == "perspective_front_view":
            cv2.resizeWindow(lidar_window_name, self.PERSPECTIVE_VIEW_WIDTH, self.PERSPECTIVE_VIEW_HEIGHT)
        else: # top_down
            cv2.resizeWindow(lidar_window_name, self.TOP_DOWN_VIS_SIZE, self.TOP_DOWN_VIS_SIZE)

        running = True
        while rclpy.ok() and running:
            if self.latest_frame is not None:
                cv2.imshow("Camera Feed", self.latest_frame)
                self.latest_frame = None

            if self.lidar_visualization_image is not None:
                cv2.imshow(lidar_window_name, self.lidar_visualization_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False; break
            elif key == ord('v'):
                if self.current_view_mode == "top_down":
                    self.current_view_mode = "perspective_front_view"
                    cv2.resizeWindow(lidar_window_name, self.PERSPECTIVE_VIEW_WIDTH, self.PERSPECTIVE_VIEW_HEIGHT)
                    self.get_logger().info("Switched to LiDAR Perspective Front View")
                else: # Was perspective_front_view
                    self.current_view_mode = "top_down"
                    cv2.resizeWindow(lidar_window_name, self.TOP_DOWN_VIS_SIZE, self.TOP_DOWN_VIS_SIZE)
                    self.get_logger().info("Switched to LiDAR Top-Down View")
                self.lidar_visualization_image = None
            
            rclpy.spin_once(self, timeout_sec=0.01)
        cv2.destroyAllWindows()

def main(args=None):
    print("OpenCV version: %s" % cv2.__version__)
    rclpy.init(args=args)
    node = ImageAndLidarSubscriber()
    node.current_view_mode = "perspective_front_view" # Start in this view
    node.get_logger().info(f"Starting in LiDAR {node.current_view_mode} mode. Press 'v' to toggle.")

    try:
        node.display_data()
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.')
    except Exception as e:
        node.get_logger().error(f"An unhandled error occurred in main loop: {e}\n{traceback.format_exc()}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        elif hasattr(node, 'get_logger'):
            node.get_logger().info("rclpy context was not OK during shutdown.")

if __name__ == '__main__':
    main()