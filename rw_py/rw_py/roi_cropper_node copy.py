#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ROICropperNode(Node):
    def __init__(self):
        super().__init__('roi_cropper_node')

        # Declare parameters for ROI percentages and topic names
        self.declare_parameter('input_image_topic', 'camera/image')
        self.declare_parameter('output_image_topic', '/image_roi_cropped')
        self.declare_parameter('x_min_percent', 0)      # Percentage from left
        self.declare_parameter('x_max_percent', 100)    # Percentage from left
        self.declare_parameter('y_min_percent', 19)      # Percentage from top
        self.declare_parameter('y_max_percent', 76)    # Percentage from top

        # Get parameter values
        input_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value
        self.x_min_percent = self.get_parameter('x_min_percent').get_parameter_value().integer_value
        self.x_max_percent = self.get_parameter('x_max_percent').get_parameter_value().integer_value
        self.y_min_percent = self.get_parameter('y_min_percent').get_parameter_value().integer_value
        self.y_max_percent = self.get_parameter('y_max_percent').get_parameter_value().integer_value

        # Validate percentages (basic validation)
        if not (0 <= self.x_min_percent <= 100 and \
                0 <= self.x_max_percent <= 100 and \
                0 <= self.y_min_percent <= 100 and \
                0 <= self.y_max_percent <= 100 and \
                self.x_min_percent < self.x_max_percent and \
                self.y_min_percent < self.y_max_percent):
            self.get_logger().error("Invalid ROI percentage parameters. Ensure 0 <= min < max <= 100.")
            # You might want to raise an exception or use default full image if parameters are bad
            # For now, we'll proceed, but cropping might fail or produce unexpected results.
            # Better: use default values if invalid
            self.get_logger().warn("Using default full image due to invalid ROI parameters.")
            self.x_min_percent = 0
            self.x_max_percent = 100
            self.y_min_percent = 0
            self.y_max_percent = 100


        self.bridge = CvBridge()

        # Subscriber to the input image topic
        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value  # Appropriate QoS for sensor data
        )

        # Publisher for the cropped image
        self.cropped_image_pub = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info(f"ROI Cropper Node started. Subscribing to '{input_topic}'.")
        self.get_logger().info(f"Publishing cropped ROI to '{output_topic}'.")
        self.get_logger().info(f"ROI Parameters: x_min={self.x_min_percent}%, x_max={self.x_max_percent}%, "
                               f"y_min={self.y_min_percent}%, y_max={self.y_max_percent}%")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        height, width, _ = cv_image.shape

        # Convert percentage ROI to pixel coordinates
        # Ensure pixel values are integers and within image bounds
        x_min_px = max(0, int(width * (self.x_min_percent / 100.0)))
        x_max_px = min(width, int(width * (self.x_max_percent / 100.0)))
        y_min_px = max(0, int(height * (self.y_min_percent / 100.0)))
        y_max_px = min(height, int(height * (self.y_max_percent / 100.0)))
        
        # Ensure min is less than max after conversion and clamping
        # If x_min_px >= x_max_px or y_min_px >= y_max_px, the crop will be empty or invalid
        if x_min_px >= x_max_px or y_min_px >= y_max_px:
            self.get_logger().warn_once( # Log only once to avoid spamming
                f"Calculated pixel ROI is invalid or empty: "
                f"x_px=[{x_min_px},{x_max_px}], y_px=[{y_min_px},{y_max_px}]. "
                f"Original image dimensions: {width}x{height}. "
                "Publishing original image or an empty image if crop is zero-size."
            )
            # Option 1: Publish original if ROI is bad (safer for downstream nodes)
            # cropped_image = cv_image
            # Option 2: Proceed, which might result in an empty image if, e.g., x_min_px == x_max_px
            if x_min_px == x_max_px or y_min_px == y_max_px:
                # Create an empty image of the same type if the crop area is zero
                # This might be preferable to crashing a downstream node expecting an image
                # However, an empty image might also cause issues.
                # For now, let's publish a very small (1x1) black image to signify an issue.
                self.get_logger().debug("Crop dimensions are zero, publishing minimal image.")
                cropped_image = np.zeros((1, 1, 3), dtype=cv_image.dtype)
            else:
                 cropped_image = cv_image[y_min_px:y_max_px, x_min_px:x_max_px]

        else:
            # Crop the image using NumPy slicing
            cropped_image = cv_image[y_min_px:y_max_px, x_min_px:x_max_px]

        try:
            # Convert cropped OpenCV image back to ROS Image message
            cropped_msg = self.bridge.cv2_to_imgmsg(cropped_image, "bgr8")
            # Preserve the header from the original message (timestamp, frame_id)
            cropped_msg.header = msg.header
            self.cropped_image_pub.publish(cropped_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error on publishing: {e}")
        except Exception as e:
            self.get_logger().error(f"Error during cropping or publishing: {e}")
            self.get_logger().error(f"ROI px: x=[{x_min_px},{x_max_px}], y=[{y_min_px},{y_max_px}]")
            self.get_logger().error(f"Original shape: {cv_image.shape}, Cropped shape attempt: {cropped_image.shape if 'cropped_image' in locals() else 'N/A'}")


def main(args=None):
    rclpy.init(args=args)
    roi_cropper_node = ROICropperNode()
    try:
        rclpy.spin(roi_cropper_node)
    except KeyboardInterrupt:
        pass
    finally:
        roi_cropper_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()