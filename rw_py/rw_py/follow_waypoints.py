#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time as RclpyTime

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints, NavigateToPose
from std_srvs.srv import SetBool
from action_msgs.msg import GoalStatus
from enum import Enum, auto
import traceback
import tf2_ros

from rw_interfaces.srv import GetWaypoints
from rw_interfaces.msg import ProximityStatus, NavigationStatus

class NavState(Enum):
    IDLE = 0; FETCHING_WAYPOINTS = 1; FOLLOWING_GLOBAL_WAYPOINTS = 2
    AWAITING_LOCAL_CORRECTION = 3; FOLLOWING_LOCAL_TARGET = 4
    WAYPOINT_SEQUENCE_COMPLETE = 5; MISSION_FAILED = 6

class WaypointFollowerCorrected(Node):
    def __init__(self):
        super().__init__('waypoint_follower_corrected_node')
        
        # Parameters
        self.declare_parameter('local_target_arrival_threshold', 0.35)
        self.declare_parameter('local_goal_update_threshold', 0.25)
        self.declare_parameter('correction_wait_timeout', 10.0)
        self.declare_parameter('robot_base_frame', 'base_footprint')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('segmentation_node_name', 'roi_lidar_fusion_node')
        self.declare_parameter('corrected_local_goal_topic', '/corrected_local_goal')

        # Get Parameters
        self.local_arrival_thresh_sq_ = self.get_parameter('local_target_arrival_threshold').value ** 2
        self.local_goal_update_threshold_sq_ = self.get_parameter('local_goal_update_threshold').value ** 2
        self.correction_wait_timeout_ = self.get_parameter('correction_wait_timeout').value
        self.robot_base_frame_ = self.get_parameter('robot_base_frame').value
        self.global_frame_ = self.get_parameter('global_frame').value
        self.activation_srv_name_ = f"/{self.get_parameter('segmentation_node_name').value}/activate_segmentation"
        self.corrected_goal_topic_ = self.get_parameter('corrected_local_goal_topic').value
        
        # State variables
        self.state_ = NavState.IDLE
        self.all_waypoints_ = []
        self.last_truly_completed_global_idx_ = -1
        self.correction_active_for_wp_idx_ = -1
        self.latest_corrected_goal_ = None
        self.last_sent_local_goal_pose_ = None
        self.correction_wait_timer_ = None
        self._current_nav2_wp_index = 0

        # ROS Communications
        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)
        self._follow_waypoints_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self._navigate_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.segmentation_activation_client_ = self.create_client(SetBool, self.activation_srv_name_)
        self.get_waypoints_client_ = self.create_client(GetWaypoints, 'get_waypoints')
        self.proximity_sub_ = self.create_subscription(ProximityStatus, 'proximity_status', self.proximity_callback, 10)
        self.corrected_goal_sub_ = None
        self.nav_status_publisher_ = self.create_publisher(NavigationStatus, 'navigation_status', 10)
        self._current_follow_wp_goal_handle = None
        self._current_nav_to_pose_goal_handle = None

        self.get_logger().info('Orchestrator Initialized.')
        self._change_state_and_process(NavState.FETCHING_WAYPOINTS)

    def _change_state_and_process(self, new_state: NavState):
        if self.state_ == new_state: return
        self.get_logger().info(f"STATE: {self.state_.name} -> {new_state.name}")
        old_state = self.state_
        self.state_ = new_state
        if old_state == NavState.AWAITING_LOCAL_CORRECTION:
            self._destroy_correction_wait_timer()
            if new_state != NavState.FOLLOWING_LOCAL_TARGET:
                self._destroy_corrected_goal_subscriber()
        self._process_state_actions()
        self.publish_nav_status()

    def _process_state_actions(self):
        if self.state_ == NavState.FETCHING_WAYPOINTS: self.fetch_waypoints_from_server()
        elif self.state_ == NavState.AWAITING_LOCAL_CORRECTION:
            self._create_corrected_goal_subscriber(); self._start_correction_wait_timer()
        elif self.state_ == NavState.MISSION_FAILED:
            self._cancel_all_navigation_actions(); self._activate_segmentation_node(False)

    def publish_nav_status(self):
        status_msg = NavigationStatus()
        status_msg.status_code = self.state_.value; status_msg.status_message = self.state_.name
        status_msg.last_completed_waypoint_index = self.last_truly_completed_global_idx_
        status_msg.is_using_corrected_goal = (self.state_ == NavState.FOLLOWING_LOCAL_TARGET)
        current_goal_for_status = None
        if self.state_ == NavState.FOLLOWING_LOCAL_TARGET and self.latest_corrected_goal_:
            status_msg.current_waypoint_index = self.correction_active_for_wp_idx_
            current_goal_for_status = self.latest_corrected_goal_
        elif self.state_ in [NavState.FOLLOWING_GLOBAL_WAYPOINTS, NavState.AWAITING_LOCAL_CORRECTION]:
            global_idx = self.last_truly_completed_global_idx_ + 1 + self._current_nav2_wp_index
            if 0 <= global_idx < len(self.all_waypoints_):
                status_msg.current_waypoint_index = global_idx
                current_goal_for_status = self.all_waypoints_[global_idx]
        if current_goal_for_status: status_msg.current_goal_pose = current_goal_for_status
        self.nav_status_publisher_.publish(status_msg)

    def fetch_waypoints_from_server(self):
        while not self.get_waypoints_client_.wait_for_service(timeout_sec=2.0): self.get_logger().info('Waypoint service not available...');
        self.get_waypoints_client_.call_async(GetWaypoints.Request()).add_done_callback(self.waypoints_response_callback)

    def waypoints_response_callback(self, future):
        try:
            response = future.result()
            if response.success and response.waypoints: self.all_waypoints_ = response.waypoints; self._send_follow_waypoints_goal_from_index(0)
            else: self._change_state_and_process(NavState.MISSION_FAILED)
        except Exception: self._change_state_and_process(NavState.MISSION_FAILED)

    def proximity_callback(self, msg: ProximityStatus):
        if self.state_ != NavState.FOLLOWING_GLOBAL_WAYPOINTS or not msg.is_within_activation_distance: return
        if self._activate_segmentation_node(True):
            self.correction_active_for_wp_idx_ = msg.waypoint_index
            self._change_state_and_process(NavState.AWAITING_LOCAL_CORRECTION)

    def corrected_goal_callback(self, msg: PoseStamped):
        if not msg.header.frame_id: return
        if self.state_ not in [NavState.AWAITING_LOCAL_CORRECTION, NavState.FOLLOWING_LOCAL_TARGET]: return
        self.latest_corrected_goal_ = msg
        self.publish_nav_status()
        
        if self.state_ == NavState.AWAITING_LOCAL_CORRECTION:
            self.get_logger().info("First valid corrected goal received. Switching to local navigation.")
            self._change_state_and_process(NavState.FOLLOWING_LOCAL_TARGET)
            self._send_navigate_to_pose_goal(self.latest_corrected_goal_)
        elif self.state_ == NavState.FOLLOWING_LOCAL_TARGET:
            is_new_goal_different = True
            if self.last_sent_local_goal_pose_:
                dist_sq = (msg.pose.position.x - self.last_sent_local_goal_pose_.pose.position.x)**2 + \
                          (msg.pose.position.y - self.last_sent_local_goal_pose_.pose.position.y)**2
                if dist_sq < self.local_goal_update_threshold_sq_: is_new_goal_different = False
            if is_new_goal_different: self._send_navigate_to_pose_goal(self.latest_corrected_goal_)

    def _create_corrected_goal_subscriber(self):
        if self.corrected_goal_sub_ is None:
            _qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.corrected_goal_sub_ = self.create_subscription(PoseStamped, self.corrected_goal_topic_, self.corrected_goal_callback, _qos)
    
    def _destroy_corrected_goal_subscriber(self):
        if self.corrected_goal_sub_ is not None: self.destroy_subscription(self.corrected_goal_sub_); self.corrected_goal_sub_ = None

    def _start_correction_wait_timer(self):
        self._destroy_correction_wait_timer()
        self.correction_wait_timer_ = self.create_timer(self.correction_wait_timeout_, self.correction_wait_timeout_cb)

    def _destroy_correction_wait_timer(self):
        if self.correction_wait_timer_: self.correction_wait_timer_.cancel(); self.correction_wait_timer_ = None
    
    def correction_wait_timeout_cb(self):
        if self.state_ == NavState.AWAITING_LOCAL_CORRECTION:
            self.get_logger().warn("Timed out waiting for a corrected goal. Resuming global navigation.")
            self._destroy_correction_wait_timer(); self._resume_global_waypoints_after_failed_correction()

    def _activate_segmentation_node(self, activate: bool):
        if not self.segmentation_activation_client_.service_is_ready(): return False
        self.segmentation_activation_client_.call_async(SetBool.Request(data=activate))
        return True

    def _cancel_all_navigation_actions(self):
        if self._current_follow_wp_goal_handle and self._current_follow_wp_goal_handle.status < GoalStatus.STATUS_SUCCEEDED:
            self._current_follow_wp_goal_handle.cancel_goal_async()
        if self._current_nav_to_pose_goal_handle and self._current_nav_to_pose_goal_handle.status < GoalStatus.STATUS_SUCCEEDED:
            self._current_nav_to_pose_goal_handle.cancel_goal_async()

    def _send_follow_waypoints_goal_from_index(self, start_index: int):
        self._cancel_all_navigation_actions()
        if start_index >= len(self.all_waypoints_): self._change_state_and_process(NavState.WAYPOINT_SEQUENCE_COMPLETE); return
        if not self._follow_waypoints_client.wait_for_server(timeout_sec=5.0): self._change_state_and_process(NavState.MISSION_FAILED); return
        send_goal_future = self._follow_waypoints_client.send_goal_async(FollowWaypoints.Goal(poses=self.all_waypoints_[start_index:]), self.follow_waypoints_feedback_cb)
        send_goal_future.add_done_callback(self.follow_waypoints_goal_response_cb)
        self._change_state_and_process(NavState.FOLLOWING_GLOBAL_WAYPOINTS)

    def follow_waypoints_goal_response_cb(self, future):
        goal_handle = future.result();
        if not goal_handle.accepted: self._change_state_and_process(NavState.MISSION_FAILED); return
        self._current_follow_wp_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.follow_waypoints_result_cb)

    def follow_waypoints_feedback_cb(self, feedback_msg: FollowWaypoints.Feedback):
        self._current_nav2_wp_index = feedback_msg.feedback.current_waypoint; self.publish_nav_status()

    def follow_waypoints_result_cb(self, future):
        result = future.result(); self._current_follow_wp_goal_handle = None
        if self.state_ != NavState.FOLLOWING_GLOBAL_WAYPOINTS: return
        if result.status == GoalStatus.STATUS_SUCCEEDED: self.last_truly_completed_global_idx_ = len(self.all_waypoints_) - 1; self._change_state_and_process(NavState.WAYPOINT_SEQUENCE_COMPLETE)
        else: self._change_state_and_process(NavState.MISSION_FAILED)

    def _send_navigate_to_pose_goal(self, target_pose: PoseStamped):
        if self._current_nav_to_pose_goal_handle and self._current_nav_to_pose_goal_handle.status < GoalStatus.STATUS_SUCCEEDED:
             self.get_logger().info("Preempting active NavigateToPose goal.")
             self._current_nav_to_pose_goal_handle.cancel_goal_async(); return
        if not self._navigate_to_pose_client.wait_for_server(timeout_sec=3.0): self._resume_global_waypoints_after_failed_correction(); return
        self.last_sent_local_goal_pose_ = target_pose
        send_future = self._navigate_to_pose_client.send_goal_async(NavigateToPose.Goal(pose=target_pose), feedback_callback=self.navigate_to_pose_feedback_cb)
        send_future.add_done_callback(self.navigate_to_pose_goal_response_cb)

    def navigate_to_pose_feedback_cb(self, feedback_msg: NavigateToPose.Feedback):
        if not self.latest_corrected_goal_: return
        robot_pos = feedback_msg.feedback.current_pose.pose.position
        target_pos = self.latest_corrected_goal_.pose.position
        dist_sq = (robot_pos.x - target_pos.x)**2 + (robot_pos.y - target_pos.y)**2
        if dist_sq < self.local_arrival_thresh_sq_ and self._current_nav_to_pose_goal_handle and self._current_nav_to_pose_goal_handle.status < GoalStatus.STATUS_SUCCEEDED:
            self._current_nav_to_pose_goal_handle.cancel_goal_async()

    def navigate_to_pose_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted: self._resume_global_waypoints_after_failed_correction(); return
        self._current_nav_to_pose_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.navigate_to_pose_result_cb)

    def navigate_to_pose_result_cb(self, future):
        result = future.result(); self._current_nav_to_pose_goal_handle = None
        if result.status == GoalStatus.STATUS_SUCCEEDED: self._handle_local_target_completion(); return
        if result.status == GoalStatus.STATUS_CANCELED:
            robot_pose = self.get_robot_pose()
            if robot_pose and self.latest_corrected_goal_:
                target_pos = self.latest_corrected_goal_.pose.position
                dist_sq = (robot_pose.pose.position.x - target_pos.x)**2 + (robot_pose.pose.position.y - target_pos.y)**2
                if dist_sq < self.local_arrival_thresh_sq_: self._handle_local_target_completion(); return
            if self.latest_corrected_goal_ and self.last_sent_local_goal_pose_ != self.latest_corrected_goal_:
                self._send_navigate_to_pose_goal(self.latest_corrected_goal_); return
        self._resume_global_waypoints_after_failed_correction()
    
    def get_robot_pose(self) -> PoseStamped | None:
        try:
            transform = self.tf_buffer_.lookup_transform(self.global_frame_, self.robot_base_frame_, RclpyTime())
            pose = PoseStamped(); pose.header.frame_id = self.global_frame_
            pose.pose.position = transform.transform.translation
            return pose
        except tf2_ros.TransformException: return None

    def _handle_local_target_completion(self):
        self.get_logger().info(f"Local target for global_idx {self.correction_active_for_wp_idx_} is complete.")
        self.last_truly_completed_global_idx_ = self.correction_active_for_wp_idx_
        self._activate_segmentation_node(False); self._destroy_corrected_goal_subscriber()
        self._send_follow_waypoints_goal_from_index(self.last_truly_completed_global_idx_ + 1)

    def _resume_global_waypoints_after_failed_correction(self):
        self.get_logger().warn("Resuming global waypoints after a local navigation issue.")
        self._activate_segmentation_node(False); self._destroy_corrected_goal_subscriber()
        if self.correction_active_for_wp_idx_ > self.last_truly_completed_global_idx_:
            self.last_truly_completed_global_idx_ = self.correction_active_for_wp_idx_
        self._send_follow_waypoints_goal_from_index(self.last_truly_completed_global_idx_ + 1)

    def destroy_node(self):
        self._activate_segmentation_node(False); self._cancel_all_navigation_actions(); super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try: node = WaypointFollowerCorrected(); rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if node and rclpy.ok(): node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__': main()