# e.g., 0:IDLE, 1:NAV_GLOBAL, 2:AWAITING_CORRECTION, 3:NAV_LOCAL, 4:WAYPOINT_DONE, 5:MISSION_COMPLETE, 6:FAILED
uint8 status_code
string status_message

# The index of the global waypoint Nav2 is currently heading towards, or the one being corrected
int32 current_waypoint_index

int32 last_completed_waypoint_index # -1 if none are complete

bool is_using_corrected_goal
geometry_msgs/PoseStamped current_goal_pose # The goal pose the robot is trying to reach (can be global or local)