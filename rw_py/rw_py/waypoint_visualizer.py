#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point
from rw_interfaces.srv import GetWaypoints
from rw_interfaces.msg import NavigationStatus

class WaypointVisualizer(Node):
    def __init__(self):
        super().__init__('waypoint_visualizer_node')
        
        self.all_waypoints_ = []
        self.current_nav_status_ = None
        self.latest_corrected_goal_ = None

        self.marker_publisher_ = self.create_publisher(MarkerArray, 'waypoint_markers', 10)

        # Subscribers
        self.nav_status_sub_ = self.create_subscription(
            NavigationStatus, 'navigation_status', self.nav_status_callback, 10
        )
        # *** THIS IS THE MISSING PIECE FOR THE VISUALIZER ***
        _qos_reliable_transient = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.corrected_goal_sub = self.create_subscription(
            PoseStamped, '/corrected_local_goal', self.corrected_goal_callback, _qos_reliable_transient
        )

        self.get_waypoints_client = self.create_client(GetWaypoints, 'get_waypoints')

        self.timer = self.create_timer(0.5, self.publish_markers)

        self.get_logger().info('Waypoint Visualizer initialized. Waiting for waypoints...')
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
            else:
                self.get_logger().error("Failed to fetch waypoints from server.")
        except Exception as e:
            self.get_logger().error(f"Service call for waypoints failed: {e}")

    def nav_status_callback(self, msg: NavigationStatus):
        self.current_nav_status_ = msg
        
    def corrected_goal_callback(self, msg: PoseStamped):
        if msg.header.frame_id:
            self.latest_corrected_goal_ = msg
        else:
            self.latest_corrected_goal_ = None

    def publish_markers(self):
        if not self.all_waypoints_: return

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        
        for i, pose_stamped in enumerate(self.all_waypoints_):
            marker = Marker()
            marker.header.frame_id = pose_stamped.header.frame_id
            marker.header.stamp = now
            marker.ns = "global_waypoints"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = pose_stamped.pose
            marker.scale.x = 0.8; marker.scale.y = 0.1; marker.scale.z = 0.1
            marker.lifetime = RclpyDuration(seconds=1.1).to_msg()
            if self.current_nav_status_:
                if i <= self.current_nav_status_.last_completed_waypoint_index:
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 0.7
                elif i == self.current_nav_status_.current_waypoint_index:
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 1.0, 1.0
                else:
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 0.0, 1.0, 0.5
            else:
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 0.0, 1.0, 0.5
            marker_array.markers.append(marker)

        if self.latest_corrected_goal_:
            local_marker = Marker()
            local_marker.header = self.latest_corrected_goal_.header
            local_marker.header.stamp = now
            local_marker.ns = "local_corrected_target"
            local_marker.id = 1000
            local_marker.type = Marker.SPHERE
            local_marker.action = Marker.ADD
            local_marker.pose = self.latest_corrected_goal_.pose
            local_marker.scale.x, local_marker.scale.y, local_marker.scale.z = 0.4, 0.4, 0.4
            local_marker.color.r, local_marker.color.g, local_marker.color.b, local_marker.color.a = 1.0, 0.0, 0.0, 1.0
            local_marker.lifetime = RclpyDuration(seconds=1.1).to_msg()
            marker_array.markers.append(local_marker)
            if self.current_nav_status_ and 0 <= self.current_nav_status_.current_waypoint_index < len(self.all_waypoints_):
                og_pose = self.all_waypoints_[self.current_nav_status_.current_waypoint_index].pose
                line_marker = Marker()
                line_marker.header = self.latest_corrected_goal_.header
                line_marker.header.stamp = now
                line_marker.ns = "correction_line"
                line_marker.id = 1001
                line_marker.type = Marker.LINE_STRIP
                line_marker.action = Marker.ADD
                line_marker.scale.x = 0.05
                line_marker.color.r, line_marker.color.g, line_marker.color.b, line_marker.color.a = 1.0, 1.0, 0.0, 0.8
                line_marker.points.extend([og_pose.position, self.latest_corrected_goal_.pose.position])
                line_marker.lifetime = RclpyDuration(seconds=1.1).to_msg()
                marker_array.markers.append(line_marker)
        if marker_array.markers:
            self.marker_publisher_.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = WaypointVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()