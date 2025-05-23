import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import tf2_ros
import tf_transformations

class ImageLidarFusionNode(Node):
    def __init__(self):
        super().__init__('image_lidar_fusion_optimized')
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        self.use_sim_time = self.get_parameter('use_sim_time').value
        if self.use_sim_time:
            self.get_logger().info("Node using simulation time.")
        else:
            self.get_logger().warn("use_sim_time is FALSE. Ensure TF uses correct time source.")

        # Subscriptions
        self.image_sub = self.create_subscription(Image, 'camera/image', self.image_callback, 1)
        self.pc_sub = self.create_subscription(PointCloud2, 'scan_02/points', self.pc_callback, 1)

        # State
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_pc_msg = None

        # Camera intrinsics
        self.img_w, self.img_h = 1920, 1200
        hfov = 1.25
        self.fx = self.img_w / (2 * math.tan(hfov / 2))
        self.fy = self.fx
        self.cx, self.cy = self.img_w / 2.0, self.img_h / 2.0

        # Coloring params
        self.min_dist, self.max_dist = 1.5, 5.0
        self.colormap = cv2.COLORMAP_JET

        # Point drawing size
        self.point_radius = 2

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.target_frame = 'front_camera_link_optical'
        self.source_frame = 'front_lidar_link_optical'

        # Only fused view window
        cv2.namedWindow('Fused View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Fused View', 800, 600)

        # Timer for display
        self.create_timer(1.0/30.0, self.timer_callback)

    def image_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            img = cv2.resize(img, (self.img_w, self.img_h))
            self.latest_image = img
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def pc_callback(self, msg: PointCloud2):
        self.latest_pc_msg = msg

    def project_points(self):
        img, pc_msg = self.latest_image, self.latest_pc_msg
        if img is None or pc_msg is None:
            return img.copy() if img is not None else np.zeros((self.img_h, self.img_w, 3), np.uint8)

        raw = point_cloud2.read_points_numpy(pc_msg, field_names=('x','y','z'), skip_nans=True)
        if raw.dtype.names:
            pts = np.vstack((raw['x'], raw['y'], raw['z'])).T
        else:
            pts = raw.reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] == 0:
            return img.copy()

        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.target_frame, self.source_frame,
                pc_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return img.copy()

        q = tf_stamped.transform.rotation
        t = tf_stamped.transform.translation
        T = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[:3,3] = [t.x, t.y, t.z]

        pts_h = np.hstack((pts, np.ones((pts.shape[0],1))))
        cam = (T @ pts_h.T).T[:, :3]

        Xopt = -cam[:,1]
        Yopt = -cam[:,2]
        Zopt = cam[:,0]

        mask = Zopt > 0.01
        Xo, Yo, Zo, src = Xopt[mask], Yopt[mask], Zopt[mask], pts[mask]
        if Xo.size == 0:
            return img.copy()

        d = np.linalg.norm(src, axis=1)
        norm = np.clip((d - self.min_dist)/(self.max_dist - self.min_dist), 0, 1)
        vals = ((1.0 - norm) * 255).astype(np.uint8)
        colors = cv2.applyColorMap(vals.reshape(-1,1), self.colormap).reshape(-1,3)

        u = (self.fx * Xo / Zo + self.cx).astype(int)
        v = (self.fy * Yo / Zo + self.cy).astype(int)
        valid = (u>=0)&(u<self.img_w)&(v>=0)&(v<self.img_h)
        u, v, colors = u[valid], v[valid], colors[valid]

        fused = img.copy()
        r = self.point_radius
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                uu = np.clip(u + dx, 0, self.img_w-1)
                vv = np.clip(v + dy, 0, self.img_h-1)
                fused[vv, uu] = colors
        return fused

    def timer_callback(self):
        fused = self.project_points()
        cv2.imshow('Fused View', cv2.resize(fused, (800,600)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Quit requested, shutting down.')
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ImageLidarFusionNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()