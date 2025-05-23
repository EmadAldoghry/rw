#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ROISegmentationNode(Node):
    def __init__(self):
        super().__init__('roi_segmentation_node')

        # --- ROI Cropping Parameters ---
        self.declare_parameter('input_image_topic', 'camera/image')
        self.declare_parameter('output_roi_image_topic', '/image_roi_cropped') # Original ROI output
        self.declare_parameter('x_min_percent', 0)
        self.declare_parameter('x_max_percent', 100)
        self.declare_parameter('y_min_percent', 19)
        self.declare_parameter('y_max_percent', 76)

        # --- Black Item Segmentation Parameters ---
        self.declare_parameter('enable_black_segmentation', True)
        self.declare_parameter('output_contour_image_topic', '/image_roi_contours') # New output with contours
        # HSV Thresholds for Black (these will likely need tuning)
        self.declare_parameter('black_h_min', 0)
        self.declare_parameter('black_s_min', 0)
        self.declare_parameter('black_v_min', 0)    # Lower bound for Value (brightness)
        self.declare_parameter('black_h_max', 180)  # Hue can be wide for black
        self.declare_parameter('black_s_max', 255)  # Saturation can also be wide
        self.declare_parameter('black_v_max', 50)   # Upper bound for Value (brightness) - crucial for black

        # --- Get Cropping Parameter Values ---
        input_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value
        output_roi_topic = self.get_parameter('output_roi_image_topic').get_parameter_value().string_value
        self.x_min_percent = self.get_parameter('x_min_percent').get_parameter_value().integer_value
        self.x_max_percent = self.get_parameter('x_max_percent').get_parameter_value().integer_value
        self.y_min_percent = self.get_parameter('y_min_percent').get_parameter_value().integer_value
        self.y_max_percent = self.get_parameter('y_max_percent').get_parameter_value().integer_value

        # --- Get Segmentation Parameter Values ---
        self.enable_segmentation = self.get_parameter('enable_black_segmentation').get_parameter_value().bool_value
        output_contour_topic = self.get_parameter('output_contour_image_topic').get_parameter_value().string_value
        self.h_min = self.get_parameter('black_h_min').get_parameter_value().integer_value
        self.s_min = self.get_parameter('black_s_min').get_parameter_value().integer_value
        self.v_min = self.get_parameter('black_v_min').get_parameter_value().integer_value
        self.h_max = self.get_parameter('black_h_max').get_parameter_value().integer_value
        self.s_max = self.get_parameter('black_s_max').get_parameter_value().integer_value
        self.v_max = self.get_parameter('black_v_max').get_parameter_value().integer_value

        # Validate ROI percentages
        if not (0 <= self.x_min_percent <= 100 and \
                0 <= self.x_max_percent <= 100 and \
                0 <= self.y_min_percent <= 100 and \
                0 <= self.y_max_percent <= 100 and \
                self.x_min_percent < self.x_max_percent and \
                self.y_min_percent < self.y_max_percent):
            self.get_logger().error("Invalid ROI percentage parameters. Ensure 0 <= min < max <= 100.")
            self.get_logger().warn("Using default full image due to invalid ROI parameters.")
            self.x_min_percent, self.x_max_percent = 0, 100
            self.y_min_percent, self.y_max_percent = 0, 100

        self.bridge = CvBridge()

        # Subscriber
        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        )

        # Publisher for the cropped ROI (without contours)
        self.roi_image_pub = self.create_publisher(Image, output_roi_topic, 10)

        # Publisher for the image with contours (if segmentation enabled)
        if self.enable_segmentation:
            self.contour_image_pub = self.create_publisher(Image, output_contour_topic, 10)
            self.get_logger().info(f"Black item segmentation enabled. Contours will be published to '{output_contour_topic}'.")
            self.get_logger().info(f"Black HSV range: H({self.h_min}-{self.h_max}), S({self.s_min}-{self.s_max}), V({self.v_min}-{self.v_max})")

        self.get_logger().info(f"ROI Segmentation Node started. Subscribing to '{input_topic}'.")
        self.get_logger().info(f"Publishing cropped ROI to '{output_roi_topic}'.")
        self.get_logger().info(f"ROI Params: x_min={self.x_min_percent}%, x_max={self.x_max_percent}%, "
                               f"y_min={self.y_min_percent}%, y_max={self.y_max_percent}%")

    def image_callback(self, msg):
        try:
            cv_image_original = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        height, width, _ = cv_image_original.shape

        # --- 1. Crop to ROI ---
        x_min_px = max(0, int(width * (self.x_min_percent / 100.0)))
        x_max_px = min(width, int(width * (self.x_max_percent / 100.0)))
        y_min_px = max(0, int(height * (self.y_min_percent / 100.0)))
        y_max_px = min(height, int(height * (self.y_max_percent / 100.0)))

        if x_min_px >= x_max_px or y_min_px >= y_max_px:
            self.get_logger().warn_once(
                f"Calculated pixel ROI is invalid or empty. Original: {width}x{height}. "
                f"ROI px: x=[{x_min_px},{x_max_px}], y=[{y_min_px},{y_max_px}]."
            )
            # If ROI is bad, we might publish the original or an empty image
            # For segmentation, an empty ROI won't work well. Let's use a minimal valid image.
            if x_min_px == x_max_px or y_min_px == y_max_px:
                 # If crop is zero-size, create a tiny black image
                cropped_image = np.zeros((1, 1, 3), dtype=cv_image_original.dtype)
            else: # Should not happen if x_min_px >= x_max_px check above is robust
                cropped_image = cv_image_original[y_min_px:y_max_px, x_min_px:x_max_px]
        else:
            cropped_image = cv_image_original[y_min_px:y_max_px, x_min_px:x_max_px]

        # Publish the (potentially just) cropped ROI image
        try:
            cropped_msg = self.bridge.cv2_to_imgmsg(cropped_image, "bgr8")
            cropped_msg.header = msg.header
            self.roi_image_pub.publish(cropped_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error publishing cropped ROI: {e}")
        except Exception as e:
            self.get_logger().error(f"Error publishing cropped ROI: {e}")


        # --- 2. Segment Black Items and Draw Contours (if enabled) ---
        if self.enable_segmentation and cropped_image.size > 0: # Ensure cropped_image is not empty
            try:
                # Convert cropped image to HSV
                hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

                # Define lower and upper bounds for black color in HSV
                lower_black = np.array([self.h_min, self.s_min, self.v_min])
                upper_black = np.array([self.h_max, self.s_max, self.v_max])

                # Create a mask for black regions
                black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

                # Optional: Morphological operations to clean up the mask
                kernel = np.ones((5,5), np.uint8)
                black_mask_morphed = cv2.erode(black_mask, kernel, iterations=1)
                black_mask_morphed = cv2.dilate(black_mask_morphed, kernel, iterations=1)
                # You might want to publish black_mask_morphed for debugging

                # Find contours
                # For cv2.findContours, ensure the input is a binary image (black_mask_morphed)
                # The mode cv2.RETR_EXTERNAL retrieves only the extreme outer contours.
                # cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments.
                contours, hierarchy = cv2.findContours(black_mask_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Create an image to draw contours on (make a copy of the cropped BGR image)
                image_with_contours = cropped_image.copy()

                # Draw contours
                # -1 for contourIdx draws all contours
                # (0, 255, 0) is green color for contours
                # 2 is the thickness
                cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

                # Publish the image with contours
                contour_msg = self.bridge.cv2_to_imgmsg(image_with_contours, "bgr8")
                contour_msg.header = msg.header # Use original message header
                self.contour_image_pub.publish(contour_msg)

            except CvBridgeError as e:
                self.get_logger().error(f"CvBridge Error during segmentation/publishing: {e}")
            except Exception as e:
                self.get_logger().error(f"Error during black item segmentation: {e}")
                self.get_logger().error(f"Cropped image shape: {cropped_image.shape}")


def main(args=None):
    rclpy.init(args=args)
    roi_segmentation_node = ROISegmentationNode()
    try:
        rclpy.spin(roi_segmentation_node)
    except KeyboardInterrupt:
        pass
    finally:
        roi_segmentation_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()