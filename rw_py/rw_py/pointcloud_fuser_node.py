#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from rclpy.duration import Duration as RclpyDuration
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge
import numpy as np
import math
import tf2_ros
import tf_transformations

class PointCloudFuserNode(Node):
    def __init__(self):
        super().__init__('pointcloud_fuser_node')
        
        # Parameters
        self.declare_parameter('input_pc_topic', 'scan_02/points')
        self.declare_parameter('input_mask_topic', '/target_mask')
        self.declare_parameter('output_pc_topic', '/target_points')
        self.declare_parameter('img_w', 1920)
        self.declare_parameter('img_h', 1200)
        self.declare_parameter('hfov', 1.25)
        self.declare_parameter('camera_optical_frame', 'front_camera_link_optical')
        self.declare_parameter('lidar_optical_frame', 'front_lidar_link_optical')
        self.declare_parameter('static_transform_lookup_timeout_sec', 5.0)

        self.img_w_param = self.get_parameter('img_w').value
        self.img_h_param = self.get_parameter('img_h').value
        self.hfov_ = self.get_parameter('hfov').value
        self.camera_optical_frame_ = self.get_parameter('camera_optical_frame').value
        self.lidar_optical_frame_ = self.get_parameter('lidar_optical_frame').value
        self.tf_timeout_sec_ = self.get_parameter('static_transform_lookup_timeout_sec').value

        # Calculate camera intrinsics from HFOV
        self.fx_ = self.img_w_param / (2 * math.tan(self.hfov_ / 2.0))
        self.fy_ = self.fx_
        self.cx_ = self.img_w_param / 2.0
        self.cy_ = self.img_h_param / 2.0
        
        self.bridge_ = CvBridge()
        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)
        self.static_transform_lidar_to_cam_ = None
        self._get_static_transform_lidar_to_cam() # Initial attempt

        # Synchronized Subscribers
        qos_sensor = rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        pc_sub = message_filters.Subscriber(self, PointCloud2, self.get_parameter('input_pc_topic').value, qos_profile=qos_sensor)
        mask_sub = message_filters.Subscriber(self, Image, self.get_parameter('input_mask_topic').value, qos_profile=qos_sensor)
        self.ts_ = message_filters.ApproximateTimeSynchronizer([pc_sub, mask_sub], 10, 0.1)
        self.ts_.registerCallback(self.fusing_callback)
        
        # Publisher
        self.target_points_pub_ = self.create_publisher(PointCloud2, self.get_parameter('output_pc_topic').value, 10)
        
        self.get_logger().info('PointCloud Fuser Node started.')

    def _get_static_transform_lidar_to_cam(self):
        try:
            transform_stamped = self.tf_buffer_.lookup_transform(
                self.camera_optical_frame_, self.lidar_optical_frame_, RclpyTime(),
                timeout=RclpyDuration(seconds=self.tf_timeout_sec_))
            q = transform_stamped.transform.rotation
            t = transform_stamped.transform.translation
            T_mat = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            T_mat[:3, 3] = [t.x, t.y, t.z]
            self.static_transform_lidar_to_cam_ = T_mat
            self.get_logger().info('Successfully acquired static transform from LiDAR to Camera.')
        except Exception as e:
            self.get_logger().warn(f"TF Fail (Lidar->Cam): {e}. Will retry.", throttle_duration_sec=5.0)
            self.static_transform_lidar_to_cam_ = None

    def fusing_callback(self, pc_msg: PointCloud2, mask_msg: Image):
        if self.static_transform_lidar_to_cam_ is None:
            self._get_static_transform_lidar_to_cam()
            if self.static_transform_lidar_to_cam_ is None:
                return # Can't process without the transform

        cv_mask = self.bridge_.imgmsg_to_cv2(mask_msg, 'mono8')
        pts_raw = pc2.read_points_numpy(pc_msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if pts_raw.size == 0:
            return
        
        # --- FIX IS HERE ---
        # The line below was causing the crash. It has been replaced.
        # OLD: pts_lidar_frame = np.vstack((pts_raw['x'], pts_raw['y'], pts_raw['z'])).T
        # NEW:
        pts_lidar_frame = pts_raw.reshape(-1, 3)

        pts_h_lidar = np.hstack((pts_lidar_frame, np.ones((pts_lidar_frame.shape[0], 1))))
        pts_in_cam_optical_frame = (self.static_transform_lidar_to_cam_ @ pts_h_lidar.T).T[:, :3]

        Z_depth = pts_in_cam_optical_frame[:, 0]
        X_px_val = -pts_in_cam_optical_frame[:, 1]
        Y_px_val = -pts_in_cam_optical_frame[:, 2]

        valid_depth_mask = Z_depth > 0.01
        u_img_coords = (self.fx_ * X_px_val[valid_depth_mask] / Z_depth[valid_depth_mask] + self.cx_).astype(int)
        v_img_coords = (self.fy_ * Y_px_val[valid_depth_mask] / Z_depth[valid_depth_mask] + self.cy_).astype(int)
        
        pts_lidar_with_valid_depth = pts_lidar_frame[valid_depth_mask]

        valid_proj_mask = (u_img_coords >= 0) & (u_img_coords < self.img_w_param) & \
                          (v_img_coords >= 0) & (v_img_coords < self.img_h_param)

        u_on_img, v_on_img = u_img_coords[valid_proj_mask], v_img_coords[valid_proj_mask]
        pts_lidar_on_img = pts_lidar_with_valid_depth[valid_proj_mask]

        if u_on_img.size == 0:
            return
            
        mask_values = cv_mask[v_on_img, u_on_img]
        target_points_mask = mask_values > 0
        
        final_target_points = pts_lidar_on_img[target_points_mask]

        if final_target_points.shape[0] > 0:
            header = pc_msg.header
            target_pc_msg = pc2.create_cloud_xyz32(header, final_target_points)
            self.target_points_pub_.publish(target_pc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudFuserNode()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Unhandled exception in PointCloudFuserNode: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()