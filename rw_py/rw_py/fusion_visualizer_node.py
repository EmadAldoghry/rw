#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from rclpy.duration import Duration as RclpyDuration
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import tf2_ros
import tf_transformations
from sensor_msgs_py import point_cloud2 as pc2

class FusionVisualizerNode(Node):
    def __init__(self):
        super().__init__('fusion_visualizer_node')

        # Parameters for visualization
        self.declare_parameter('raw_image_topic', 'camera/image')
        self.declare_parameter('mask_topic', '/target_mask')
        self.declare_parameter('raw_pc_topic', 'scan_02/points')
        self.declare_parameter('target_pc_topic', '/target_points')
        self.declare_parameter('output_window', 'Fused View')
        self.declare_parameter('img_w', 1920)
        self.declare_parameter('img_h', 1200)
        self.declare_parameter('hfov', 1.25)
        self.declare_parameter('camera_optical_frame', 'front_camera_link_optical')
        self.declare_parameter('lidar_optical_frame', 'front_lidar_link_optical')
        self.declare_parameter('point_radius_viz', 2)
        
        self.img_w, self.img_h = self.get_parameter('img_w').value, self.get_parameter('img_h').value
        self.hfov = self.get_parameter('hfov').value
        self.camera_frame = self.get_parameter('camera_optical_frame').value
        self.lidar_frame = self.get_parameter('lidar_optical_frame').value
        self.output_window = self.get_parameter('output_window').value

        self.fx = self.img_w / (2 * math.tan(self.hfov / 2.0)); self.fy = self.fx
        self.cx = self.img_w / 2.0; self.cy = self.img_h / 2.0
        
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(); self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.lidar_to_cam_transform = None

        self.latest_image, self.latest_mask, self.latest_raw_pc, self.latest_target_pc = None, None, None, None

        if self.output_window: cv2.namedWindow(self.output_window, cv2.WINDOW_NORMAL); cv2.resizeWindow(self.output_window, 800, 600)
        
        qos = rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(Image, self.get_parameter('raw_image_topic').value, lambda msg: setattr(self, 'latest_image', msg), qos)
        self.create_subscription(Image, self.get_parameter('mask_topic').value, lambda msg: setattr(self, 'latest_mask', msg), qos)
        self.create_subscription(PointCloud2, self.get_parameter('raw_pc_topic').value, lambda msg: setattr(self, 'latest_raw_pc', msg), qos)
        self.create_subscription(PointCloud2, self.get_parameter('target_pc_topic').value, lambda msg: setattr(self, 'latest_target_pc', msg), qos)
        
        self.timer = self.create_timer(1.0 / 15.0, self.visualization_callback)
        self.get_logger().info('Fusion Visualizer Node started.')

    def get_transform(self):
        try:
            t = self.tf_buffer.lookup_transform(self.camera_frame, self.lidar_frame, RclpyTime(), timeout=RclpyDuration(seconds=1.0))
            self.lidar_to_cam_transform = t
            return True
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"Could not get transform from {self.lidar_frame} to {self.camera_frame}: {e}", throttle_duration_sec=5)
            return False

    def project_pc_to_image(self, pc_msg):
        if self.lidar_to_cam_transform is None:
            if not self.get_transform(): return [], []

        pts_raw = pc2.read_points_numpy(pc_msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if pts_raw.size == 0: return [], []
        
        # --- FIX IS HERE ---
        # The line below was causing the crash. It has been replaced.
        # OLD: pts_lidar_frame = np.vstack((pts_raw['x'], pts_raw['y'], pts_raw['z'])).T
        # NEW:
        pts_lidar_frame = pts_raw.reshape(-1, 3)

        pts_h = np.hstack([pts_lidar_frame, np.ones((len(pts_lidar_frame), 1))])
        
        q = self.lidar_to_cam_transform.transform.rotation
        trans = self.lidar_to_cam_transform.transform.translation
        T_mat = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T_mat[:3, 3] = [trans.x, trans.y, trans.z]
        
        pts_cam = (T_mat @ pts_h.T).T[:, :3]

        Z, X, Y = pts_cam[:, 0], -pts_cam[:, 1], -pts_cam[:, 2]
        valid_idx = Z > 0.01
        
        u = (self.fx * X[valid_idx] / Z[valid_idx] + self.cx).astype(int)
        v = (self.fy * Y[valid_idx] / Z[valid_idx] + self.cy).astype(int)
        
        on_img_idx = (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)
        return u[on_img_idx], v[on_img_idx]

    def visualization_callback(self):
        if self.latest_image is None or not self.output_window: return
        
        try:
            display_img = self.bridge.imgmsg_to_cv2(self.latest_image, 'bgr8').copy()

            if self.latest_mask:
                mask = self.bridge.imgmsg_to_cv2(self.latest_mask, 'mono8')
                colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                colored_mask[np.where((colored_mask == [255,255,255]).all(axis=2))] = [255,0,0] # Blue for mask
                display_img = cv2.addWeighted(display_img, 1.0, colored_mask, 0.4, 0)
                
            if self.latest_raw_pc:
                u_raw, v_raw = self.project_pc_to_image(self.latest_raw_pc)
                for i in range(len(u_raw)):
                    cv2.circle(display_img, (u_raw[i], v_raw[i]), self.get_parameter('point_radius_viz').value, (0, 255, 0), -1) # Green for all points
            
            if self.latest_target_pc:
                u_target, v_target = self.project_pc_to_image(self.latest_target_pc)
                for i in range(len(u_target)):
                    cv2.circle(display_img, (u_target[i], v_target[i]), self.get_parameter('point_radius_viz').value, (0, 0, 255), -1) # Red for target points
            
            cv2.imshow(self.output_window, cv2.resize(display_img, (800, 600)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.destroy_node(); rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f"Error in visualizer callback: {e}", throttle_duration_sec=5.0)

    def destroy_node(self):
        if self.output_window: cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FusionVisualizerNode()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Unhandled exception in FusionVisualizerNode: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()