import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav2_msgs.action import FollowWaypoints # Action definition
import yaml # For loading waypoints from YAML file
from pathlib import Path # For path manipulation
import sys # For exiting gracefully
import os # For os.path.join
from ament_index_python.packages import get_package_share_directory # To find package paths

class WaypointLoaderFollower(Node):
    """
    A ROS 2 node that loads waypoints from a YAML file and sends them
    to the Nav2 FollowWaypoints action server.
    It attempts to use a default YAML path from the 'rw' package
    if no path is provided via parameters.
    """
    def __init__(self):
        super().__init__('waypoint_loader_follower_node')

        # --- Determine Default YAML Path ---
        default_yaml_file_path = '' # Initialize with an empty string
        # The package where the default config/nav_waypoints.yaml is expected to be
        default_config_package_name = 'rw'
        try:
            package_share_directory = get_package_share_directory(default_config_package_name)
            default_yaml_file_path = os.path.join(
                package_share_directory,
                'config',
                'nav_waypoints.yaml'
            )
            self.get_logger().info(
                f"Constructed default waypoints YAML path: '{default_yaml_file_path}' "
                f"(from package '{default_config_package_name}')"
            )
        except Exception as e: # More general exception to catch ModuleNotFoundError etc.
            self.get_logger().warn(
                f"Could not automatically determine default YAML path from package "
                f"'{default_config_package_name}'. Error: {e}. "
                f"The 'waypoints_yaml_path' parameter will need to be set explicitly if the default is not found."
            )
            default_yaml_file_path = '' # Ensure it's empty if lookup fails

        # Declare the parameter for the waypoints YAML file path
        param_descriptor = ParameterDescriptor(
            description=(
                'Full path to the YAML file containing the waypoints. '
                f'If not set, defaults to trying: {default_yaml_file_path} (if resolvable).'
            )
        )
        # Use the determined default_yaml_file_path as the default value for the parameter
        self.declare_parameter('waypoints_yaml_path', default_yaml_file_path, param_descriptor)

        # Get the YAML path from the parameter server (will use default if not overridden)
        self.yaml_path_param_ = self.get_parameter('waypoints_yaml_path').get_parameter_value().string_value

        self.waypoints_ = [] # To store the loaded PoseStamped messages

        # Create an action client for the FollowWaypoints action
        self._action_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')

        self.get_logger().info("WaypointLoaderFollower node initialized.")

        # Validate the (potentially defaulted or overridden) YAML path
        if not self.yaml_path_param_:
            self.get_logger().error("Effective 'waypoints_yaml_path' is empty after considering defaults and parameters.")
            self.get_logger().error("Please ensure the default path is valid or set the parameter explicitly.")
            sys.exit(1)

        resolved_path = Path(self.yaml_path_param_)
        if not resolved_path.is_file():
            self.get_logger().error(f"Waypoint YAML file not found at the effective path: '{resolved_path}'")
            self.get_logger().error("This path might be from a parameter or the default package lookup.")
            sys.exit(1)

        self.get_logger().info(f"Attempting to load waypoints from: '{resolved_path}'")

        # Load waypoints and, if successful, send them
        if self.load_waypoints_from_yaml(resolved_path): # Pass the Path object
            if self.waypoints_:
                self.send_waypoints_goal()
            else:
                self.get_logger().info("No waypoints were loaded. Node will not send any goal.")
                rclpy.shutdown()
        else:
            self.get_logger().error("Failed to load waypoints. Exiting.")
            rclpy.shutdown()


    def load_waypoints_from_yaml(self, yaml_file_path: Path) -> bool: # Accept Path object
        """
        Loads waypoints from the given YAML file path.
        Populates self.waypoints_ with a list of PoseStamped messages.
        Returns True on success, False on failure.
        """
        try:
            with open(yaml_file_path, 'r') as file:
                yaml_data = yaml.safe_load(file)

            if not yaml_data or 'poses' not in yaml_data:
                self.get_logger().error(
                    f"YAML file '{yaml_file_path}' is empty or does not contain a 'poses' key."
                )
                return False

            loaded_waypoints = []
            for i, pose_data_entry in enumerate(yaml_data['poses']):
                try:
                    ps_msg = PoseStamped()
                    header_data = pose_data_entry.get('header', {})
                    ps_msg.header.frame_id = header_data.get('frame_id', 'map')
                    stamp_data = header_data.get('stamp', {})
                    ps_msg.header.stamp.sec = int(stamp_data.get('sec', 0))
                    ps_msg.header.stamp.nanosec = int(stamp_data.get('nanosec', 0))

                    pose_block = pose_data_entry.get('pose', {})
                    position_data = pose_block.get('position', {})
                    ps_msg.pose.position.x = float(position_data.get('x', 0.0))
                    ps_msg.pose.position.y = float(position_data.get('y', 0.0))
                    ps_msg.pose.position.z = float(position_data.get('z', 0.0))

                    orientation_data = pose_block.get('orientation', {})
                    ps_msg.pose.orientation.x = float(orientation_data.get('x', 0.0))
                    ps_msg.pose.orientation.y = float(orientation_data.get('y', 0.0))
                    ps_msg.pose.orientation.z = float(orientation_data.get('z', 0.0))
                    ps_msg.pose.orientation.w = float(orientation_data.get('w', 1.0))

                    loaded_waypoints.append(ps_msg)
                    self.get_logger().debug(
                        f"Loaded waypoint {i+1}: "
                        f"p({ps_msg.pose.position.x:.2f}, {ps_msg.pose.position.y:.2f}, {ps_msg.pose.position.z:.2f}) "
                        f"q({ps_msg.pose.orientation.x:.2f}, {ps_msg.pose.orientation.y:.2f}, {ps_msg.pose.orientation.z:.2f}, {ps_msg.pose.orientation.w:.2f}) "
                        f"in frame '{ps_msg.header.frame_id}'"
                    )
                except (TypeError, ValueError, KeyError) as e:
                    self.get_logger().error(
                        f"Error parsing waypoint entry {i+1} from YAML: {e}. Data: {pose_data_entry}"
                    )
                    return False

            self.waypoints_ = loaded_waypoints
            self.get_logger().info(
                f"Successfully loaded {len(self.waypoints_)} waypoints from '{yaml_file_path}'."
            )
            return True
        # FileNotFoundError should be caught by the check in __init__ now
        except yaml.YAMLError as e:
            self.get_logger().error(f"Error parsing YAML file '{yaml_file_path}': {e}")
            return False
        except Exception as e:
            self.get_logger().error(f"An unexpected error occurred while loading waypoints from '{yaml_file_path}': {e}")
            return False

    def send_waypoints_goal(self):
        """
        Sends the loaded waypoints to the FollowWaypoints action server.
        """
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = self.waypoints_

        self.get_logger().info('Waiting for "follow_waypoints" action server...')
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('"follow_waypoints" action server not available after 10 seconds. Exiting.')
            rclpy.shutdown()
            return

        self.get_logger().info(
            f'Sending {len(self.waypoints_)} waypoints as a goal to "follow_waypoints" action server.'
        )
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Callback for when the action server accepts or rejects the goal.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Waypoint goal was rejected by the action server.')
            rclpy.shutdown()
            return

        self.get_logger().info('Waypoint goal accepted by the action server. Waiting for result...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """
        Callback for receiving feedback during waypoint navigation.
        """
        feedback = feedback_msg.feedback
        current_waypoint_index = feedback.current_waypoint
        self.get_logger().info(
            f'Navigating to waypoint {current_waypoint_index + 1} of {len(self.waypoints_)}...'
        )

    def get_result_callback(self, future):
        """
        Callback for when the waypoint navigation action completes.
        """
        result = future.result().result
        if result and result.missed_waypoints:
            self.get_logger().warn(
                f'Waypoint following completed with {len(result.missed_waypoints)} missed waypoints.'
            )
            for i, missed_wp_index in enumerate(result.missed_waypoints):
                 self.get_logger().warn(f"  - Missed waypoint with original index (0-based): {missed_wp_index}")
        else:
            self.get_logger().info('Waypoint following completed successfully.')

        self.get_logger().info("Shutting down WaypointLoaderFollower node.")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    waypoint_loader_follower_node = None
    try:
        waypoint_loader_follower_node = WaypointLoaderFollower()
        rclpy.spin(waypoint_loader_follower_node)
    except KeyboardInterrupt:
        if waypoint_loader_follower_node:
            waypoint_loader_follower_node.get_logger().info('Node interrupted by user (KeyboardInterrupt).')
    except SystemExit as e:
        if waypoint_loader_follower_node:
            if e.code == 0:
                 waypoint_loader_follower_node.get_logger().info('Node is shutting down as planned.')
            else:
                 waypoint_loader_follower_node.get_logger().error(f'Node exited with error code: {e.code}.')
        else:
            print("Node initialization failed before SystemExit.")
    except Exception as e:
        if waypoint_loader_follower_node:
            waypoint_loader_follower_node.get_logger().error(f"An unhandled exception occurred: {e}")
            import traceback
            waypoint_loader_follower_node.get_logger().error(traceback.format_exc())
        else:
            print(f"An unhandled exception occurred during node initialization: {e}")
            import traceback
            print(traceback.format_exc())
    finally:
        if waypoint_loader_follower_node and rclpy.ok():
            waypoint_loader_follower_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()