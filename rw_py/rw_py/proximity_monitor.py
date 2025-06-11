#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from rclpy.duration import Duration as RclpyDuration
import tf2_ros
import math

from geometry_msgs.msg import PoseStamped
# We need to import the Enum to check the state correctly and robustly
from rw_interfaces.srv import GetWaypoints
from rw_interfaces.msg import NavigationStatus, ProximityStatus

# It's best practice to define the Enum where it's used or import it.
# For simplicity in this standalone node, we can redefine it or import if it were in a shared library.
# Let's import the NavState from the orchestrator to keep it robust.
# NOTE: For a real large project, this Enum would go into a shared python library. For now, we'll just be careful.
# Hardcoding the value is also an option if we are certain. Let's do that for simplicity here.
NAV_STATE_FOLLOWING_GLOBAL = 2 

class ProximityMonitor(Node):
    def __init__(self):
        super().__init__('proximity_monitor_node')

        self.declare_parameter('robot_base_frame', 'base_footprint')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('correction_activation_distance', 7.0)

        self.robot_base_frame_ = self.get_parameter('robot_base_frame').value
        self.global_frame_ = self.get_parameter('global_frame').value
        self.activation_dist_sq_ = self.get_parameter('correction_activation_distance').value ** 2

        self.all_waypoints_ = []
        self.current_nav_status_ = None

        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)

        self.proximity_publisher_ = self.create_publisher(ProximityStatus, 'proximity_status', 10)
        self.nav_status_sub_ = self.create_subscription(
            NavigationStatus, 'navigation_status', self.nav_status_callback, 10
        )
        self.get_waypoints_client = self.create_client(GetWaypoints, 'get_waypoints')
        
        self.timer = self.create_timer(1.0, self.check_proximity)

        self.get_logger().info('Proximity Monitor initialized. Waiting for waypoints...')
        self.fetch_waypoints()

    def fetch_waypoints(self):
        while not self.get_waypoints_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waypoint service not available, waiting...')
        
        request = GetWaypoints.Request()
        future = self.get_waypoints_client.call_async(request)
        future.add_done_callback(self.waypoints_response_callback)

    def waypoints_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.all_waypoints_ = response.waypoints
                self.get_logger().info(f"Successfully fetched {len(self.all_waypoints_)} waypoints.")
            else:
                self.get_logger().error("Failed to fetch waypoints from server.")
        except Exception as e:
            self.get_logger().error(f"Service call for waypoints failed: {e}")

    def nav_status_callback(self, msg: NavigationStatus):
        # Add logging to see that we are receiving status updates
        if self.current_nav_status_ is None or self.current_nav_status_.status_code != msg.status_code:
            self.get_logger().info(f"Received new NavStatus. Current state code: {msg.status_code} ({msg.status_message})")
        self.current_nav_status_ = msg

    def get_robot_pose(self) -> PoseStamped | None:
        try:
            transform = self.tf_buffer_.lookup_transform(
                self.global_frame_, self.robot_base_frame_, RclpyTime()
            )
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = self.global_frame_
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            return pose
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"Could not get robot pose: {e}", throttle_duration_sec=2)
            return None

    def check_proximity(self):
        if not self.all_waypoints_ or self.current_nav_status_ is None:
            self.get_logger().debug("Waiting for waypoints or nav status...", throttle_duration_sec=5)
            return

        # *** THE FIX IS HERE ***
        # Check if the orchestrator is in the global navigation state (code 2)
        if self.current_nav_status_.status_code != NAV_STATE_FOLLOWING_GLOBAL:
            self.get_logger().debug(f"Not in global nav state (current: {self.current_nav_status_.status_code}). Skipping proximity check.", throttle_duration_sec=5)
            return

        target_idx = self.current_nav_status_.current_waypoint_index
        if not (0 <= target_idx < len(self.all_waypoints_)):
            self.get_logger().warn(f"Received invalid target index {target_idx}. Skipping check.")
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        target_waypoint = self.all_waypoints_[target_idx]
        dist_sq = (robot_pose.pose.position.x - target_waypoint.pose.position.x)**2 + \
                  (robot_pose.pose.position.y - target_waypoint.pose.position.y)**2
        distance = math.sqrt(dist_sq)

        # Log the distance check
        self.get_logger().info(f"Checking proximity to waypoint {target_idx}. Distance: {distance:.2f}m. Activation at: {math.sqrt(self.activation_dist_sq_):.2f}m")
        
        prox_status_msg = ProximityStatus()
        prox_status_msg.header.stamp = self.get_clock().now().to_msg()
        prox_status_msg.header.frame_id = self.global_frame_
        prox_status_msg.waypoint_index = target_idx
        prox_status_msg.distance_to_target = distance
        
        if dist_sq < self.activation_dist_sq_:
            prox_status_msg.is_within_activation_distance = True
            self.get_logger().info(f"ALERT: Robot is WITHIN activation distance of waypoint {target_idx}!")
        else:
            prox_status_msg.is_within_activation_distance = False
        
        self.proximity_publisher_.publish(prox_status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ProximityMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()