#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Point, Quaternion, PointStamped
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import yaml
from pathlib import Path
import sys
import os
from ament_index_python.packages import get_package_share_directory
import tf2_ros
from tf2_geometry_msgs import do_transform_point # For transforming PointStamped
import numpy as np
import math
from enum import Enum, auto

class NavState(Enum):
    IDLE = auto()
    LOADING_WAYPOINTS = auto()
    NAVIGATING_TO_PREDEFINED = auto()
    WAITING_FOR_LIDAR_CORRECTION = auto()
    NAVIGATING_TO_CORRECTED = auto()
    WAYPOINT_SEQUENCE_COMPLETE = auto()
    PROCESSING_NEXT_WAYPOINT = auto()
    FAILED = auto()

class WaypointCorrectorNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_corrector_navigator_node')

        # --- Parameters ---
        default_yaml_file_path = self._get_default_yaml_path()
        self.declare_parameter('waypoints_yaml_path', default_yaml_file_path,
                               ParameterDescriptor(description='Full path to the waypoints YAML file.'))
        self.declare_parameter('correction_proximity_threshold', 2.0, # meters
                               ParameterDescriptor(description='Distance to predefined waypoint to trigger Lidar correction phase.'))
        self.declare_parameter('lidar_data_timeout', 5.0, # seconds
                               ParameterDescriptor(description='Time to wait for Lidar data for correction.'))
        self.declare_parameter('selected_lidar_topic', 'selected_lidar_points',
                               ParameterDescriptor(description='Topic for selected Lidar points for correction.'))
        self.declare_parameter('navigation_frame', 'map',
                               ParameterDescriptor(description='The frame_id for navigation goals.'))
        self.declare_parameter('use_original_orientation_for_corrected', True,
                               ParameterDescriptor(description='If true, corrected waypoint uses orientation of original predefined waypoint.'))


        self.yaml_path_ = self.get_parameter('waypoints_yaml_path').get_parameter_value().string_value
        self.correction_proximity_threshold_ = self.get_parameter('correction_proximity_threshold').get_parameter_value().double_value
        self.lidar_data_timeout_sec_ = self.get_parameter('lidar_data_timeout').get_parameter_value().double_value
        self.selected_lidar_topic_ = self.get_parameter('selected_lidar_topic').get_parameter_value().string_value
        self.navigation_frame_ = self.get_parameter('navigation_frame').get_parameter_value().string_value
        self.use_original_orientation_ = self.get_parameter('use_original_orientation_for_corrected').get_parameter_value().bool_value

        self.predefined_waypoints_ = []
        self.current_waypoint_index_ = -1
        self.current_goal_handle_ = None
        self.last_feedback_pose_ = None # Store the last feedback pose from NavigateToPose

        self.state_ = NavState.IDLE

        # Action client for NavigateToPose
        self._nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # TF Buffer and Listener
        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)

        # Subscriber for Lidar data (created when needed)
        self.lidar_sub_ = None
        self.lidar_correction_timer_ = None # Timer for Lidar data timeout
        self.received_lidar_data_ = None # Store received lidar data

        qos_profile_reliable_transient_local = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        # Create publisher for current target pose for visualization
        self.target_pose_pub_ = self.create_publisher(PoseStamped, 'waypoint_corrector/current_target', qos_profile_reliable_transient_local)


        self.get_logger().info("WaypointCorrectorNavigator node initialized.")
        self.change_state(NavState.LOADING_WAYPOINTS)
        self._process_state() # Start processing

    def _get_default_yaml_path(self):
        default_yaml_file_path = ''
        default_config_package_name = 'rw' # Change if your package name is different
        try:
            package_share_directory = get_package_share_directory(default_config_package_name)
            default_yaml_file_path = os.path.join(
                package_share_directory, 'config', 'nav_waypoints.yaml'
            )
            self.get_logger().info(f"Constructed default waypoints YAML path: '{default_yaml_file_path}'")
        except Exception as e:
            self.get_logger().warn(f"Could not find package '{default_config_package_name}' for default YAML path: {e}")
        return default_yaml_file_path

    def change_state(self, new_state: NavState):
        if self.state_ != new_state:
            self.get_logger().info(f"State transition: {self.state_.name} -> {new_state.name}")
            self.state_ = new_state

    def _process_state(self):
        """Main state machine processing loop, called after state changes."""
        if self.state_ == NavState.LOADING_WAYPOINTS:
            if not self.yaml_path_:
                self.get_logger().error("Effective 'waypoints_yaml_path' is empty. Please set the parameter.")
                self.change_state(NavState.FAILED)
                self._process_state()
                return

            resolved_path = Path(self.yaml_path_)
            if not resolved_path.is_file():
                self.get_logger().error(f"Waypoint YAML file not found: '{resolved_path}'")
                self.change_state(NavState.FAILED)
                self._process_state()
                return

            if self.load_waypoints_from_yaml(resolved_path):
                if self.predefined_waypoints_:
                    self.get_logger().info(f"Successfully loaded {len(self.predefined_waypoints_)} waypoints.")
                    self.change_state(NavState.PROCESSING_NEXT_WAYPOINT)
                else:
                    self.get_logger().info("No waypoints loaded. Sequence complete.")
                    self.change_state(NavState.WAYPOINT_SEQUENCE_COMPLETE)
            else:
                self.get_logger().error("Failed to load waypoints.")
                self.change_state(NavState.FAILED)
            self._process_state() # Re-call to process the new state

        elif self.state_ == NavState.PROCESSING_NEXT_WAYPOINT:
            self.current_waypoint_index_ += 1
            if self.current_waypoint_index_ < len(self.predefined_waypoints_):
                predefined_goal = self.predefined_waypoints_[self.current_waypoint_index_]
                self.get_logger().info(f"Processing predefined waypoint {self.current_waypoint_index_ + 1}/{len(self.predefined_waypoints_)}")
                self.send_navigation_goal(predefined_goal, is_corrected=False)
                self.change_state(NavState.NAVIGATING_TO_PREDEFINED)
                # Do not call _process_state() here, wait for action server response/feedback
            else:
                self.get_logger().info("All predefined waypoints processed.")
                self.change_state(NavState.WAYPOINT_SEQUENCE_COMPLETE)
                self._process_state()

        elif self.state_ == NavState.NAVIGATING_TO_PREDEFINED:
            # Primarily driven by feedback and result callbacks
            # If feedback indicates close proximity, it will change state
            pass # Waiting for action server

        elif self.state_ == NavState.WAITING_FOR_LIDAR_CORRECTION:
            self.get_logger().info("Waiting for Lidar data for correction...")
            self.received_lidar_data_ = None # Clear previous data
            if self.lidar_sub_ is None:
                 qos_profile_sensor_data = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT, # Lidar data is often best effort
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1 # Only care about the latest
                )
                 self.lidar_sub_ = self.create_subscription(
                    PointCloud2,
                    self.selected_lidar_topic_,
                    self.lidar_data_callback,
                    qos_profile_sensor_data) # Use appropriate QoS
            if self.lidar_correction_timer_ is not None and self.lidar_correction_timer_.is_ready():
                 self.destroy_timer(self.lidar_correction_timer_) # Ensure old timer is gone
            self.lidar_correction_timer_ = self.create_timer(self.lidar_data_timeout_sec_, self.lidar_timeout_callback)
            # Do not call _process_state() here, wait for lidar_data_callback or lidar_timeout_callback

        elif self.state_ == NavState.NAVIGATING_TO_CORRECTED:
            # Driven by action server callbacks
            pass # Waiting for action server

        elif self.state_ == NavState.WAYPOINT_SEQUENCE_COMPLETE:
            self.get_logger().info("Waypoint sequence finished successfully.")
            rclpy.shutdown()

        elif self.state_ == NavState.FAILED:
            self.get_logger().error("Navigation process failed. Shutting down.")
            if self.current_goal_handle_ and self.current_goal_handle_.is_active:
                self.get_logger().info("Cancelling active navigation goal due to failure.")
                self.current_goal_handle_.cancel_goal_async()
            rclpy.shutdown()

    def load_waypoints_from_yaml(self, yaml_file_path: Path) -> bool:
        try:
            with open(yaml_file_path, 'r') as file:
                yaml_data = yaml.safe_load(file)
            if not yaml_data or 'poses' not in yaml_data:
                self.get_logger().error(f"YAML '{yaml_file_path}' empty or no 'poses' key.")
                return False

            loaded_waypoints = []
            for i, pose_entry in enumerate(yaml_data['poses']):
                try:
                    ps_msg = PoseStamped()
                    # Ensure poses are in the navigation frame
                    ps_msg.header.frame_id = self.navigation_frame_ # Override frame_id from YAML
                    ps_msg.header.stamp = self.get_clock().now().to_msg() # Use current time

                    pose_block = pose_entry.get('pose', {})
                    pos_data = pose_block.get('position', {})
                    orient_data = pose_block.get('orientation', {})

                    ps_msg.pose.position.x = float(pos_data.get('x', 0.0))
                    ps_msg.pose.position.y = float(pos_data.get('y', 0.0))
                    ps_msg.pose.position.z = float(pos_data.get('z', 0.0)) # Usually 0 for 2D nav
                    ps_msg.pose.orientation.x = float(orient_data.get('x', 0.0))
                    ps_msg.pose.orientation.y = float(orient_data.get('y', 0.0))
                    ps_msg.pose.orientation.z = float(orient_data.get('z', 0.0))
                    ps_msg.pose.orientation.w = float(orient_data.get('w', 1.0))
                    loaded_waypoints.append(ps_msg)
                except (TypeError, ValueError, KeyError) as e:
                    self.get_logger().error(f"Error parsing waypoint {i+1}: {e}. Data: {pose_entry}")
                    return False
            self.predefined_waypoints_ = loaded_waypoints
            return True
        except Exception as e:
            self.get_logger().error(f"Error loading/parsing YAML '{yaml_file_path}': {e}")
            return False

    def send_navigation_goal(self, pose_stamped_goal: PoseStamped, is_corrected: bool):
        if not self._nav_to_pose_client.server_is_ready():
            self.get_logger().error("'navigate_to_pose' action server not available. Waiting...")
            # Could implement a retry or a wait with timeout here
            if not self._nav_to_pose_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("Server still not available after timeout.")
                self.change_state(NavState.FAILED)
                self._process_state()
                return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_stamped_goal
        # goal_msg.behavior_tree = "" # Optionally specify a behavior tree

        # Publish the target for visualization
        self.target_pose_pub_.publish(pose_stamped_goal)

        goal_type_str = "corrected" if is_corrected else "predefined"
        self.get_logger().info(f"Sending {goal_type_str} goal: P(x={pose_stamped_goal.pose.position.x:.2f}, "
                               f"y={pose_stamped_goal.pose.position.y:.2f}) "
                               f"Q(z={pose_stamped_goal.pose.orientation.z:.2f}, "
                               f"w={pose_stamped_goal.pose.orientation.w:.2f})")

        send_goal_future = self._nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=lambda feedback_msg: self.navigation_feedback_callback(feedback_msg, is_corrected)
        )
        send_goal_future.add_done_callback(
            lambda future: self.navigation_goal_response_callback(future, is_corrected)
        )

    def navigation_goal_response_callback(self, future, is_corrected_goal: bool):
        self.current_goal_handle_ = future.result()
        if not self.current_goal_handle_.accepted:
            self.get_logger().error(f"Goal (is_corrected: {is_corrected_goal}) was rejected by server.")
            self.change_state(NavState.FAILED) # Or retry logic
            self._process_state()
            return

        goal_type_str = "Corrected" if is_corrected_goal else "Predefined"
        self.get_logger().info(f"{goal_type_str} goal accepted. Waiting for result...")
        self._get_result_future = self.current_goal_handle_.get_result_async()
        self._get_result_future.add_done_callback(
            lambda res_future: self.navigation_result_callback(res_future, is_corrected_goal)
        )

    def navigation_feedback_callback(self, feedback_msg, is_predefined_goal_feedback: bool):
        feedback = feedback_msg.feedback
        self.last_feedback_pose_ = feedback.current_pose # Store for potential use
        # self.get_logger().debug(f"Nav feedback: Dist remaining: {feedback.distance_remaining:.2f}, "
        #                       f"Time: {feedback.navigation_time.sec}s")

        # Only check for proximity if navigating to a PREDEFINED waypoint
        if self.state_ == NavState.NAVIGATING_TO_PREDEFINED and not is_predefined_goal_feedback: # is_predefined_goal_feedback is actually inverted here.
                                                                                                   # This means it's feedback for a PREDEFINED goal
            current_pos = feedback.current_pose.pose.position
            target_pos = self.predefined_waypoints_[self.current_waypoint_index_].pose.position
            distance = math.sqrt(
                (current_pos.x - target_pos.x)**2 +
                (current_pos.y - target_pos.y)**2 +
                (current_pos.z - target_pos.z)**2 # Include Z, though often 0
            )
            # self.get_logger().debug(f"Distance to predefined waypoint {self.current_waypoint_index_+1}: {distance:.2f}m")

            if distance < self.correction_proximity_threshold_:
                self.get_logger().info(f"Close to predefined waypoint {self.current_waypoint_index_+1} (dist: {distance:.2f}m). "
                                       "Attempting Lidar correction.")
                # Cancel current navigation to predefined waypoint
                if self.current_goal_handle_ and self.current_goal_handle_.is_active:
                    self.get_logger().info("Cancelling current navigation to predefined waypoint.")
                    cancel_future = self.current_goal_handle_.cancel_goal_async()
                    # We don't strictly need to wait for cancel_future here, state change handles it.
                    # cancel_future.add_done_callback(self.cancel_done_callback)
                self.change_state(NavState.WAITING_FOR_LIDAR_CORRECTION)
                self._process_state() # Trigger lidar subscription and timeout

    def navigation_result_callback(self, future, was_corrected_goal: bool):
        result = future.result().result
        status = future.result().status
        goal_type_str = "Corrected" if was_corrected_goal else "Predefined"

        if status == rclpy.action.GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"{goal_type_str} waypoint navigation succeeded!")
            if self.state_ == NavState.WAITING_FOR_LIDAR_CORRECTION and not was_corrected_goal:
                # This means the original predefined goal succeeded *while* we were waiting for lidar,
                # or lidar timeout occurred and we decided to let it finish.
                self.get_logger().info("Original predefined goal completed during/after Lidar phase.")
                # Clean up lidar stuff if it was active
                self._cleanup_lidar_watch()

            self.change_state(NavState.PROCESSING_NEXT_WAYPOINT)
        elif status == rclpy.action.GoalStatus.STATUS_CANCELED:
            if self.state_ == NavState.WAITING_FOR_LIDAR_CORRECTION:
                 self.get_logger().info(f"{goal_type_str} waypoint navigation was canceled, likely for Lidar correction.")
                 # This is expected if we are moving to WAITING_FOR_LIDAR_CORRECTION
            else:
                 self.get_logger().warn(f"{goal_type_str} waypoint navigation was canceled unexpectedly.")
                 self.change_state(NavState.FAILED) # Or retry
        else:
            self.get_logger().error(f"{goal_type_str} waypoint navigation failed with status: {status}. Result: {result}")
            self.change_state(NavState.FAILED) # Or retry logic
        
        self.current_goal_handle_ = None # Clear the handle
        if self.state_ != NavState.WAITING_FOR_LIDAR_CORRECTION: # Don't auto-process if waiting for lidar
            self._process_state()

    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Goal successfully canceled for Lidar correction.')
        else:
            self.get_logger().warn('Goal cancellation failed or no goal to cancel.')


    def lidar_data_callback(self, msg: PointCloud2):
        if self.state_ != NavState.WAITING_FOR_LIDAR_CORRECTION:
            return # Not expecting data right now

        self.get_logger().info(f"Received Lidar data on '{self.selected_lidar_topic_}'.")
        self.received_lidar_data_ = msg # Store it
        self._cleanup_lidar_watch() # Stop timer and unsubscribe

        self.process_lidar_correction()

    def _cleanup_lidar_watch(self):
        if self.lidar_correction_timer_ is not None:
            self.destroy_timer(self.lidar_correction_timer_)
            self.lidar_correction_timer_ = None
        if self.lidar_sub_ is not None:
            self.destroy_subscription(self.lidar_sub_)
            self.lidar_sub_ = None
        self.get_logger().debug("Cleaned up Lidar watch (timer and subscriber).")


    def lidar_timeout_callback(self):
        if self.state_ != NavState.WAITING_FOR_LIDAR_CORRECTION:
            return

        self.get_logger().warn("Timeout waiting for Lidar data for correction.")
        self._cleanup_lidar_watch() # Stop timer and unsubscribe

        if self.received_lidar_data_:
            self.get_logger().info("Processing Lidar data that arrived just before timeout.")
            self.process_lidar_correction()
        else:
            self.get_logger().info("No Lidar data received. Proceeding to original predefined waypoint.")
            # The original goal might have been cancelled. Or it might still be running if cancel failed.
            # Simplest is to resend the original goal.
            original_goal = self.predefined_waypoints_[self.current_waypoint_index_]
            self.send_navigation_goal(original_goal, is_corrected=False) # Treat as a new "predefined" attempt
            self.change_state(NavState.NAVIGATING_TO_PREDEFINED) # Back to navigating to it
            # No _process_state() here, wait for action server.

    def process_lidar_correction(self):
        if not self.received_lidar_data_:
            self.get_logger().error("Process Lidar Correction called but no data available.")
            # Fallback: go to original predefined waypoint
            original_goal = self.predefined_waypoints_[self.current_waypoint_index_]
            self.send_navigation_goal(original_goal, is_corrected=False)
            self.change_state(NavState.NAVIGATING_TO_PREDEFINED)
            return

        try:
            points_list = []
            # Assuming points are (x,y,z)
            for point in pc2.read_points(self.received_lidar_data_, field_names=("x", "y", "z"), skip_nans=True):
                points_list.append([point[0], point[1], point[2]])

            if not points_list:
                self.get_logger().warn("Lidar data received, but no valid points found after parsing.")
                original_goal = self.predefined_waypoints_[self.current_waypoint_index_]
                self.send_navigation_goal(original_goal, is_corrected=False)
                self.change_state(NavState.NAVIGATING_TO_PREDEFINED)
                return

            points_np = np.array(points_list)
            centroid_lidar_frame = np.mean(points_np, axis=0)
            self.get_logger().info(f"Lidar points centroid (in {self.received_lidar_data_.header.frame_id}): "
                                   f"x={centroid_lidar_frame[0]:.2f}, y={centroid_lidar_frame[1]:.2f}, z={centroid_lidar_frame[2]:.2f}")

            # Transform centroid to navigation frame (e.g., 'map')
            point_stamped_lidar = PointStamped()
            point_stamped_lidar.header = self.received_lidar_data_.header # Use stamp and frame_id from cloud
            point_stamped_lidar.point.x = float(centroid_lidar_frame[0])
            point_stamped_lidar.point.y = float(centroid_lidar_frame[1])
            point_stamped_lidar.point.z = float(centroid_lidar_frame[2])

            try:
                # Wait for transform to be available
                self.tf_buffer_.can_transform(
                    self.navigation_frame_, point_stamped_lidar.header.frame_id,
                    point_stamped_lidar.header.stamp, timeout=Duration(seconds=1.0)
                )
                point_stamped_map = self.tf_buffer_.transform(
                    point_stamped_lidar,
                    self.navigation_frame_,
                    timeout=Duration(seconds=1.0) # Give some time for transform
                )
            except tf2_ros.TransformException as ex:
                self.get_logger().error(f"Could not transform Lidar point from '{point_stamped_lidar.header.frame_id}' "
                                       f"to '{self.navigation_frame_}': {ex}")
                self.get_logger().info("Falling back to original predefined waypoint.")
                original_goal = self.predefined_waypoints_[self.current_waypoint_index_]
                self.send_navigation_goal(original_goal, is_corrected=False)
                self.change_state(NavState.NAVIGATING_TO_PREDEFINED)
                return

            # Create new PoseStamped goal
            corrected_goal = PoseStamped()
            corrected_goal.header.stamp = self.get_clock().now().to_msg()
            corrected_goal.header.frame_id = self.navigation_frame_
            corrected_goal.pose.position = point_stamped_map.point # Use transformed point

            original_predefined_pose = self.predefined_waypoints_[self.current_waypoint_index_].pose
            if self.use_original_orientation_:
                corrected_goal.pose.orientation = original_predefined_pose.orientation
            else:
                # Option: Use robot's current orientation if available from last feedback
                if self.last_feedback_pose_:
                    corrected_goal.pose.orientation = self.last_feedback_pose_.pose.orientation
                    self.get_logger().info("Using robot's last known orientation for corrected goal.")
                else: # Default to identity or original
                    self.get_logger().warn("No last feedback pose for orientation, using original predefined orientation.")
                    corrected_goal.pose.orientation = original_predefined_pose.orientation


            self.get_logger().info("Sending corrected Lidar-based goal.")
            self.send_navigation_goal(corrected_goal, is_corrected=True)
            self.change_state(NavState.NAVIGATING_TO_CORRECTED)
            # No _process_state() here, wait for action server.

        except Exception as e:
            self.get_logger().error(f"Error processing Lidar data: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.get_logger().info("Falling back to original predefined waypoint due to Lidar processing error.")
            original_goal = self.predefined_waypoints_[self.current_waypoint_index_]
            self.send_navigation_goal(original_goal, is_corrected=False)
            self.change_state(NavState.NAVIGATING_TO_PREDEFINED)
        finally:
            self.received_lidar_data_ = None # Clear processed data


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = WaypointCorrectorNavigator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info('Node interrupted by user.')
    except SystemExit:
        if node: node.get_logger().info('Node is shutting down.')
    except Exception as e:
        if node:
            node.get_logger().error(f"Unhandled exception: {e}")
            import traceback
            node.get_logger().error(traceback.format_exc())
        else:
            print(f"Unhandled exception during node init: {e}")
            import traceback
            print(traceback.format_exc())
    finally:
        if node and rclpy.ok() and node.is_valid():
             # If state indicates a running goal, try to cancel it
            if node.current_goal_handle_ and \
               (node.state_ == NavState.NAVIGATING_TO_PREDEFINED or \
                node.state_ == NavState.NAVIGATING_TO_CORRECTED):
                node.get_logger().info("Attempting to cancel active goal on shutdown...")
                if node.current_goal_handle_.is_active: # Humble might not have is_active
                    try:
                        # Check if cancel_goal_async exists and is callable
                        if hasattr(node.current_goal_handle_, 'cancel_goal_async') and callable(node.current_goal_handle_.cancel_goal_async):
                             cancel_future = node.current_goal_handle_.cancel_goal_async()
                             # Give it a moment, but don't block shutdown indefinitely
                             rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=1.0)
                             node.get_logger().info("Cancel request sent.")
                        else:
                            node.get_logger().warn("Goal handle does not support cancel_goal_async or is not active.")
                    except Exception as cancel_ex:
                        node.get_logger().error(f"Exception during goal cancellation: {cancel_ex}")

            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("WaypointCorrectorNavigator shut down.")

if __name__ == '__main__':
    main()