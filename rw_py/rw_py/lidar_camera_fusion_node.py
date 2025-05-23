#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration as rclpyDuration

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import message_filters
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException, Duration
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
import math
import threading
import time # For small sleeps if needed
import traceback # For detailed error messages

class LidarCameraFuser(Node):
    def __init__(self):
        super().__init__('lidar_camera_fuser')
        self.get_logger().info("Node __init__ started.")

        self.bridge = CvBridge()

        # --- OpenCV Window Names ---
        self.cv_raw_camera_window_name = "Raw Camera Image"
        self.cv_lidar_bev_window_name = "LiDAR BEV"
        self.cv_fused_window_name = "Lidar-Camera Fused Projection"

        # Data to be displayed
        self.latest_raw_cv_image = None
        self.latest_bev_cv_image = None
        self.latest_fused_cv_image = None
        self.data_lock = threading.Lock()

        # --- Parameters ---
        # ... (parameter declaration and fetching as before) ...
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("lidar_topic", "/scan_02/points")
        self.declare_parameter("output_image_topic", "/fused_projection/image")
        self.declare_parameter("camera_optical_frame", "front_camera_link_optical")
        self.declare_parameter("lidar_optical_frame", "lidar_link_optical")

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        self.output_image_topic = self.get_parameter("output_image_topic").get_parameter_value().string_value
        self.camera_optical_frame = self.get_parameter("camera_optical_frame").get_parameter_value().string_value
        self.lidar_optical_frame = self.get_parameter("lidar_optical_frame").get_parameter_value().string_value


        # --- TF2 ---
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- QoS ---
        qos_sensor_data = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # Changed to BEST_EFFORT for image/lidar
            history=HistoryPolicy.KEEP_LAST,
            depth=1 # Keep only the latest for high-rate sensors
        )
        qos_camera_info = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE # Usually good for Gazebo's CameraInfo
        )
        self.get_logger().info(f"QoS for Image ({image_topic}): R={qos_sensor_data.reliability}, H={qos_sensor_data.history}, D={qos_sensor_data.depth}")
        self.get_logger().info(f"QoS for CameraInfo ({camera_info_topic}): R={qos_camera_info.reliability}, D={qos_camera_info.durability}, H={qos_camera_info.history}, Depth={qos_camera_info.depth}")
        self.get_logger().info(f"QoS for Lidar ({lidar_topic}): R={qos_sensor_data.reliability}, H={qos_sensor_data.history}, D={qos_sensor_data.depth}")


        # --- Subscribers ---
        # Adding individual callbacks for debugging
        self.image_sub_debug = self.create_subscription(Image, image_topic, self.image_debug_callback, qos_sensor_data)
        self.cam_info_sub_debug = self.create_subscription(CameraInfo, camera_info_topic, self.cam_info_debug_callback, qos_camera_info)
        self.lidar_sub_debug = self.create_subscription(PointCloud2, lidar_topic, self.lidar_debug_callback, qos_sensor_data)
        self.get_logger().info("Individual debug subscribers created.")


        image_sub = message_filters.Subscriber(self, Image, image_topic, qos_profile=qos_sensor_data)
        camera_info_sub = message_filters.Subscriber(self, CameraInfo, camera_info_topic, qos_profile=qos_camera_info)
        lidar_sub = message_filters.Subscriber(self, PointCloud2, lidar_topic, qos_profile=qos_sensor_data)
        self.get_logger().info("Message_filters subscribers created.")

        # --- Synchronizer ---
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, camera_info_sub, lidar_sub],
            queue_size=30, # Increased queue
            slop=0.5       # Increased slop significantly for debugging
        )
        self.ts.registerCallback(self.synchronized_callback)
        self.get_logger().info(f"ApproximateTimeSynchronizer registered with queue_size=30, slop=0.5.")


        # --- Publisher ---
        self.fused_image_pub = self.create_publisher(Image, self.output_image_topic, 10)

        # --- Camera Intrinsics ---
        self.K = None
        self.D = None
        self.image_width = 0
        self.image_height = 0

        # --- BEV Parameters ---
        # ... (BEV params as before) ...
        self.bev_resolution = 0.05
        self.bev_display_range_m = 20.0
        self.bev_img_width_px = int(self.bev_display_range_m / self.bev_resolution)
        self.bev_img_height_px = int(self.bev_display_range_m / self.bev_resolution)
        self.bev_max_z_color = 2.0
        self.bev_min_z_color = -0.5

        # --- Threading Control ---
        self.running = True
        self.spin_thread = threading.Thread(target=self.ros_spin_thread_func)
        # self.spin_thread.daemon = True # Optional: thread exits when main thread exits
        self.spin_thread.start()

        self.get_logger().info(f"LidarCameraFuser node initialized. OpenCV version: {cv2.__version__}")
        self.get_logger().info(f"Topics: Image='{image_topic}', CamInfo='{camera_info_topic}', Lidar='{lidar_topic}'")
        self.get_logger().info(f"Frames: CameraOptical='{self.camera_optical_frame}', LidarOptical='{self.lidar_optical_frame}'")

    # --- DEBUGGING CALLBACKS ---
    def image_debug_callback(self, msg):
        self.get_logger().info(f"DEBUG: Received Image message on '{msg.header.frame_id}' (stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec})", throttle_duration_sec=2.0)

    def cam_info_debug_callback(self, msg):
        self.get_logger().info(f"DEBUG: Received CameraInfo message on '{msg.header.frame_id}' (stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}) K[0]={msg.k[0]}", throttle_duration_sec=2.0)

    def lidar_debug_callback(self, msg):
        self.get_logger().info(f"DEBUG: Received PointCloud2 message on '{msg.header.frame_id}' (stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}, NumPoints approx: {msg.width * msg.height})", throttle_duration_sec=2.0)
    # --- END DEBUGGING CALLBACKS ---

    def ros_spin_thread_func(self):
        self.get_logger().info("ROS Spin thread started.")
        while rclpy.ok() and self.running:
            try:
                rclpy.spin_once(self, timeout_sec=0.1)
            except Exception as e:
                self.get_logger().error(f"Exception in ros_spin_thread_func: {e}\n{traceback.format_exc()}")
        self.get_logger().info("ROS Spin thread finished.")


    def create_lidar_bev_image(self, lidar_msg):
        # ... (same as before) ...
        bev_image = np.zeros((self.bev_img_height_px, self.bev_img_width_px, 3), dtype=np.uint8)
        points_drawn = 0
        for point in pc2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True):
            lx, ly, lz = point[0], point[1], point[2]
            if abs(lx) > self.bev_display_range_m / 2 or abs(ly) > self.bev_display_range_m / 2:
                continue
            u_bev = int(self.bev_img_width_px / 2 - ly / self.bev_resolution)
            v_bev = int(self.bev_img_height_px / 2 - lx / self.bev_resolution)
            if 0 <= u_bev < self.bev_img_width_px and 0 <= v_bev < self.bev_img_height_px:
                norm_z = np.clip((lz - self.bev_min_z_color) / (self.bev_max_z_color - self.bev_min_z_color), 0, 1)
                blue = int(255 * (1 - norm_z))
                red = int(255 * norm_z)
                green = int(100 * norm_z * (1-norm_z) * 4)
                cv2.circle(bev_image, (u_bev, v_bev), radius=1, color=(blue, green, red), thickness=-1)
                points_drawn +=1
        cv2.line(bev_image, (self.bev_img_width_px // 2, self.bev_img_height_px // 2), (self.bev_img_width_px // 2, self.bev_img_height_px // 2 - int(1.0/self.bev_resolution)), (0,255,0), 1)
        cv2.line(bev_image, (self.bev_img_width_px // 2, self.bev_img_height_px // 2), (self.bev_img_width_px // 2 - int(1.0/self.bev_resolution), self.bev_img_height_px // 2), (0,0,255), 1)
        return bev_image

    def synchronized_callback(self, image_msg, camera_info_msg, lidar_msg):
        self.get_logger().info(f"SYNC CALLBACK: Image ts={image_msg.header.stamp.sec}.{image_msg.header.stamp.nanosec}, CamInfo ts={camera_info_msg.header.stamp.sec}.{camera_info_msg.header.stamp.nanosec}, Lidar ts={lidar_msg.header.stamp.sec}.{lidar_msg.header.stamp.nanosec}")
        try:
            # 1. Store Camera Intrinsics
            if self.K is None or self.image_width != camera_info_msg.width or self.K[0] != camera_info_msg.k[0]: # Check if K changed
                self.K = np.array(camera_info_msg.k).reshape((3, 3))
                self.D = np.array(camera_info_msg.d)
                self.image_width = camera_info_msg.width
                self.image_height = camera_info_msg.height
                if self.K[0,0] == 0.0:
                    self.get_logger().warn(f"Cam intrinsics (fx) is zero. K={self.K}.", throttle_duration_sec=10)
                    return # Critical error
                self.get_logger().info(f"Cam intrinsics updated: W={self.image_width}, H={self.image_height}, K[0,0]={self.K[0,0]:.2f}")

            # 2. Convert ROS Image to OpenCV image
            cv_raw_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            self.get_logger().debug("Raw image converted to CV.")

            # 3. Create LiDAR BEV Image
            bev_image = self.create_lidar_bev_image(lidar_msg)
            self.get_logger().debug("BEV image created.")

            # 4. Fusion
            cv_fused_image = cv_raw_image.copy()
            source_frame = lidar_msg.header.frame_id
            target_frame = self.camera_optical_frame
            transform_stamped = self.tf_buffer.lookup_transform(
                target_frame, source_frame, lidar_msg.header.stamp, timeout=Duration(seconds=0.1)
            )
            self.get_logger().debug(f"TF lookup successful from {source_frame} to {target_frame}.")
            
            # ... (Projection logic as before, with some debug logs) ...
            projected_points_for_drawing = []
            points_processed = 0
            for point_idx, point_data in enumerate(pc2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)):
                points_processed += 1
                lidar_point_stamped = PointStamped()
                lidar_point_stamped.header.frame_id = source_frame
                lidar_point_stamped.header.stamp = lidar_msg.header.stamp
                lidar_point_stamped.point.x = float(point_data[0])
                lidar_point_stamped.point.y = float(point_data[1])
                lidar_point_stamped.point.z = float(point_data[2])
                
                camera_point_stamped = tf2_geometry_msgs.do_transform_point(lidar_point_stamped, transform_stamped)
                
                Xc, Yc, Zc = camera_point_stamped.point.x, camera_point_stamped.point.y, camera_point_stamped.point.z
                if Zc <= 0.1: continue
                if self.K is not None and self.K[0,0] != 0.0:
                    u_raw = (self.K[0,0] * Xc / Zc) + self.K[0,2]
                    v_raw = (self.K[1,1] * Yc / Zc) + self.K[1,2]
                    if 0 <= u_raw < self.image_width and 0 <= v_raw < self.image_height:
                        projected_points_for_drawing.append(((int(u_raw), int(v_raw)), Zc))
            self.get_logger().debug(f"Processed {points_processed} lidar points, {len(projected_points_for_drawing)} projected.")
            
            min_depth_viz, max_depth_viz = 0.3, 15.0
            for (u, v), depth in projected_points_for_drawing:
                norm_depth = np.clip((depth - min_depth_viz) / (max_depth_viz - min_depth_viz), 0, 1)
                blue = int(255 * (1 - norm_depth)); red = int(255 * norm_depth); green = int(120 * (1-abs(2*norm_depth - 1)))
                cv2.circle(cv_fused_image, (u, v), radius=2, color=(blue, green, red), thickness=-1)
            self.get_logger().debug("Drawing projected points done.")


            # 5. Update shared data for GUI thread
            with self.data_lock:
                self.latest_raw_cv_image = cv_raw_image
                self.latest_bev_cv_image = bev_image
                self.latest_fused_cv_image = cv_fused_image
            self.get_logger().info("SYNC CALLBACK: Updated shared images for GUI.") # More prominent log

            # 6. Publish Fused Image Topic
            fused_image_msg = self.bridge.cv2_to_imgmsg(cv_fused_image, "bgr8")
            fused_image_msg.header = image_msg.header
            self.fused_image_pub.publish(fused_image_msg)
            self.get_logger().debug("Published fused image to topic.")

        except CvBridgeError as e_cv:
            self.get_logger().error(f"SYNC CALLBACK ERROR (CvBridge): {e_cv}\n{traceback.format_exc()}")
        except (LookupException, ConnectivityException, ExtrapolationException) as e_tf:
            self.get_logger().warn(f"SYNC CALLBACK ERROR (TF): {e_tf}", throttle_duration_sec=2.0)
            # If TF fails, still update raw and bev for display
            with self.data_lock:
                if 'cv_raw_image' in locals(): self.latest_raw_cv_image = cv_raw_image
                if 'bev_image' in locals(): self.latest_bev_cv_image = bev_image
                self.latest_fused_cv_image = None # Indicate fusion failed
            self.get_logger().info("SYNC CALLBACK: Updated raw/bev images after TF error.")
        except Exception as e:
            self.get_logger().error(f"SYNC CALLBACK ERROR (General): {e}\n{traceback.format_exc()}")


    def run_gui_loop(self):
        self.get_logger().info("GUI display loop started in main thread.")
        
        cv2.namedWindow(self.cv_raw_camera_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.cv_lidar_bev_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.cv_fused_window_name, cv2.WINDOW_AUTOSIZE)
        self.get_logger().info("OpenCV windows created by GUI thread.")

        # Create dummy black images for initial display if nothing received yet
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "Waiting for data...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        local_raw_img, local_bev_img, local_fused_img = dummy_img.copy(), dummy_img.copy(), dummy_img.copy()

        last_gui_update_time = time.time()

        while rclpy.ok() and self.running:
            data_updated_in_gui = False
            with self.data_lock:
                if self.latest_raw_cv_image is not None:
                    local_raw_img = self.latest_raw_cv_image.copy()
                    self.latest_raw_cv_image = None # Consume it
                    data_updated_in_gui = True
                if self.latest_bev_cv_image is not None:
                    local_bev_img = self.latest_bev_cv_image.copy()
                    self.latest_bev_cv_image = None # Consume it
                    data_updated_in_gui = True
                if self.latest_fused_cv_image is not None:
                    local_fused_img = self.latest_fused_cv_image.copy()
                    self.latest_fused_cv_image = None # Consume it
                    data_updated_in_gui = True
            
            if data_updated_in_gui:
                self.get_logger().info("GUI Loop: Copied new images from shared data.")
                last_gui_update_time = time.time()

            # Always show something, even if it's the old image or dummy
            cv2.imshow(self.cv_raw_camera_window_name, local_raw_img)
            cv2.imshow(self.cv_lidar_bev_window_name, local_bev_img)
            cv2.imshow(self.cv_fused_window_name, local_fused_img)

            if time.time() - last_gui_update_time > 5.0: # If no update for 5s
                self.get_logger().warn("GUI Loop: No new image data received for 5 seconds.", throttle_duration_sec=5.0)


            key = cv2.waitKey(30)
            if key != -1 and (key & 0xFF == ord('q')):
                self.get_logger().info("Quit key 'q' pressed.")
                self.running = False
                break
        
        self.get_logger().info("GUI display loop ended.")
        self.stop()

    def stop(self):
        self.get_logger().info("Stop method called.")
        if not self.running: # Already stopping/stopped
            self.get_logger().info("Stop method: Already in stopping state.")
            # Ensure spin_thread is handled if it wasn't joined yet
            if hasattr(self, 'spin_thread') and self.spin_thread.is_alive():
                self.get_logger().info("Stop method: Spin thread still alive, attempting join.")
                self.spin_thread.join(timeout=0.5) # Short timeout as it should have exited
                if self.spin_thread.is_alive():
                    self.get_logger().warn("Stop method: Spin thread did not join after additional attempt.")
            cv2.destroyAllWindows()
            self.get_logger().info("Stop method: OpenCV windows destroyed (again, to be sure).")
            return

        self.running = False
        self.get_logger().info("Stop method: self.running set to False.")

        if hasattr(self, 'spin_thread') and self.spin_thread.is_alive():
            self.get_logger().info("Stop method: Joining spin thread...")
            self.spin_thread.join(timeout=1.0)
            if self.spin_thread.is_alive():
                self.get_logger().warn("Stop method: Spin thread did not join in time!")
            else:
                self.get_logger().info("Stop method: Spin thread joined successfully.")
        else:
            self.get_logger().info("Stop method: Spin thread not active or doesn't exist.")
            
        cv2.destroyAllWindows()
        self.get_logger().info("Stop method: OpenCV windows destroyed.")


def main(args=None):
    # print(f"OpenCV version: {cv2.__version__}") # Already in init
    rclpy.init(args=args)
    fuser_node = None
    try:
        fuser_node = LidarCameraFuser()
        fuser_node.run_gui_loop()
    except KeyboardInterrupt:
        print('Keyboard interrupt received by main.')
        if fuser_node: fuser_node.get_logger().info('Keyboard interrupt in main.')
    except Exception as e:
        print(f"Unhandled exception in main: {e}\n{traceback.format_exc()}")
        if fuser_node: fuser_node.get_logger().error(f"Unhandled exception in main: {e}\n{traceback.format_exc()}")
    finally:
        print("Main finally block entered.")
        if fuser_node:
            print("Calling fuser_node.stop() from main finally.")
            fuser_node.stop()
            print("Calling fuser_node.destroy_node() from main finally.")
            fuser_node.destroy_node()
            print("fuser_node destroyed.")
        if rclpy.ok():
            print("Calling rclpy.shutdown() from main finally.")
            rclpy.shutdown()
            print("rclpy shutdown.")
        print("Main finally block finished.")

if __name__ == '__main__':
    main()