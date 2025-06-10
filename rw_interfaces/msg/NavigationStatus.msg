      
int32 current_waypoint_index # -1 if no mission active or before start
int32 total_waypoints
uint8 status_code       # e.g., 0:IDLE, 1:NAV_PREDEFINED, 2:AWAITING_CORRECTION, 3:NAV_CORRECTED, 4:WAYPOINT_DONE, 5:MISSION_COMPLETE, 6:FAILED
string status_message
bool is_using_corrected_goal
geometry_msgs/PoseStamped current_goal_pose # The pose it's currently trying to reach

    