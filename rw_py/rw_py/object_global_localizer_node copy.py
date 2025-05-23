#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time as RclpyTime # Explicit import for clarity
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point as RosPoint, PoseStamped
from std_msgs.msg import ColorRGBA, Header
from builtin_interfaces.msg import Time as TimeMsg # For storing and loading stamps

import numpy as np
import os
import json
import yaml
import threading
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs # For automatic registration of PoseStamped with TF Buffer
import traceback # For logging exception details

# Helper function to calculate 2D Euclidean distance (ignoring Z for waypoint association)
def euclidean_distance_2d(p1, p2):
    # p1, p2 are lists or tuples [x, y, (z)]
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

class WaypointRefinementNode(Node):
    def __init__(self):
        super().__init__('waypoint_refinement_node')

        # --- Parameters ---
        self.declare_parameter('input_pc_topic', 'selected_lidar_points')
        self.declare_parameter('output_marker_topic', 'refined_waypoint_markers')
        self.declare_parameter('global_waypoints_path', '/home/aldoghry/test_ws/src/rw/config/nav_waypoints.yaml')
        self.declare_parameter('refined_waypoints_save_path', '/home/aldoghry/test_ws/src/rw_py/founds/refined_waypoints.json')
        self.declare_parameter('voxel_size', 0.3)
        self.declare_parameter('association_radius_2d', 5.0)
        self.declare_parameter('min_points_for_refinement', 1) # Defaulting to 1 as seen in previous logs
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')

        self.input_pc_topic = self.get_parameter('input_pc_topic').get_parameter_value().string_value
        self.output_marker_topic = self.get_parameter('output_marker_topic').get_parameter_value().string_value
        self.global_waypoints_path = self.get_parameter('global_waypoints_path').get_parameter_value().string_value
        self.refined_waypoints_save_path = self.get_parameter('refined_waypoints_save_path').get_parameter_value().string_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self.association_radius_2d = self.get_parameter('association_radius_2d').get_parameter_value().double_value
        self.min_points_for_refinement = self.get_parameter('min_points_for_refinement').get_parameter_value().integer_value
        self.robot_base_frame = self.get_parameter('robot_base_frame').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.global_waypoints = [] # List of [x, y, z] in map_frame
        # Key: global_wp_index (int)
        # Value: {'pose_map': [x,y,z], 'source_points_count': N, 'stamp': builtin_interfaces.msg.Time}
        self.refined_waypoints = {}
        self.data_lock = threading.Lock()
        self.last_pc_header = None

        self.load_global_waypoints()
        self.load_refined_waypoints()

        self.pc_subscriber = self.create_subscription(
            PointCloud2,
            self.input_pc_topic,
            self.point_cloud_callback,
            rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        )
        self.marker_publisher = self.create_publisher(MarkerArray, self.output_marker_topic, 10)
        self.marker_publish_timer = self.create_timer(1.0, self.publish_all_markers_callback)

        self.get_logger().info(
            f"Waypoint Refinement Node initialized.\n"
            f"\tGlobal Waypoints: {len(self.global_waypoints)} loaded (after duplicates removed) from '{self.global_waypoints_path}'\n"
            f"\tRefined Waypoints Loaded: {len(self.refined_waypoints)} from '{self.refined_waypoints_save_path}'\n"
            f"\tLiDAR Input: '{self.input_pc_topic}'\n"
            f"\tAssociation Radius (2D): {self.association_radius_2d}m\n"
            f"\tMin Points for Refinement: {self.min_points_for_refinement}"
        )

    def load_global_waypoints(self):
        raw_loaded_waypoints = []
        try:
            with open(self.global_waypoints_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'poses' in data and isinstance(data['poses'], list):
                    for i, pose_stamped_dict in enumerate(data['poses']):
                        try:
                            if pose_stamped_dict['header']['frame_id'] != self.map_frame:
                                self.get_logger().warn(
                                    f"Global waypoint {i} frame_id '{pose_stamped_dict['header']['frame_id']}' "
                                    f"does not match expected map_frame '{self.map_frame}'. Skipping."
                                )
                                continue
                            pos = pose_stamped_dict['pose']['position']
                            raw_loaded_waypoints.append([pos['x'], pos['y'], pos.get('z', 0.0)])
                        except KeyError as e:
                            self.get_logger().warn(f"Malformed global waypoint entry {i} (missing {e}). Skipping.")
                else:
                    self.get_logger().error(f"Global waypoints file '{self.global_waypoints_path}' does not contain a 'poses' list.")
                    return

            self.get_logger().info(f"Successfully loaded {len(raw_loaded_waypoints)} raw global waypoints.")
            
            unique_global_waypoints = []
            seen_coords_str = set()
            for wp in raw_loaded_waypoints:
                coord_str = f"{wp[0]:.4f}_{wp[1]:.4f}_{wp[2]:.4f}"
                if coord_str not in seen_coords_str:
                    unique_global_waypoints.append(wp)
                    seen_coords_str.add(coord_str)
            
            if len(unique_global_waypoints) < len(raw_loaded_waypoints):
                self.get_logger().info(f"Removed {len(raw_loaded_waypoints) - len(unique_global_waypoints)} duplicate global waypoints.")
            self.global_waypoints = unique_global_waypoints

        except FileNotFoundError:
            self.get_logger().error(f"Global waypoints file not found: {self.global_waypoints_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load or parse global waypoints: {e}\n{traceback.format_exc()}")

    def point_cloud_callback(self, msg: PointCloud2):
        self.get_logger().info("!!!!!!!! POINT CLOUD CALLBACK TRIGGERED !!!!!!!!");
        self.get_logger().debug(f"PC received: {msg.width * msg.height} pts, frame '{msg.header.frame_id}', stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        self.last_pc_header = msg.header

        if not self.global_waypoints:
            self.get_logger().warn("No global waypoints loaded, cannot refine.", throttle_duration_sec=10)
            return

        if msg.width * msg.height == 0:
            self.get_logger().debug("Received empty PointCloud. Skipping.")
            return

        try:
            points_structured_list = list(point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            ))
            self.get_logger().debug(f"PC: Converted to list of structured points. Count: {len(points_structured_list)}")
            if not points_structured_list or len(points_structured_list) < self.min_points_for_refinement:
                self.get_logger().info(f"PC: Not enough valid points ({len(points_structured_list)}) for refinement (min: {self.min_points_for_refinement}). Skipping.")
                return
            points_in_lidar_frame = np.array([[p['x'], p['y'], p['z']] for p in points_structured_list], dtype=np.float32)
            self.get_logger().debug(f"PC: Converted to Nx3 NumPy array in '{msg.header.frame_id}'. Shape: {points_in_lidar_frame.shape}")
        except Exception as e:
            self.get_logger().error(f"PC: Failed to read/convert points: {e}\n{traceback.format_exc()}")
            return

        voxel_indices_lidar = np.floor(points_in_lidar_frame / self.voxel_size).astype(int)
        lidar_voxels = {}
        for i in range(points_in_lidar_frame.shape[0]):
            idx_tuple = tuple(voxel_indices_lidar[i])
            if idx_tuple not in lidar_voxels: lidar_voxels[idx_tuple] = []
            lidar_voxels[idx_tuple].append(points_in_lidar_frame[i])

        if not lidar_voxels:
            self.get_logger().debug("PC: No voxels formed from points. Skipping.")
            return

        best_voxel_points_lidar = []
        max_points_in_voxel = 0
        for points_list in lidar_voxels.values():
            if len(points_list) > max_points_in_voxel:
                max_points_in_voxel = len(points_list)
                best_voxel_points_lidar = points_list
        
        if max_points_in_voxel < self.min_points_for_refinement:
            self.get_logger().info(f"PC: Dominant cluster has {max_points_in_voxel} points, less than min {self.min_points_for_refinement}. Skipping.")
            return
        self.get_logger().info(f"PC: Dominant cluster in lidar frame found with {max_points_in_voxel} points.")

        centroid_in_lidar_frame_np = np.mean(np.array(best_voxel_points_lidar), axis=0)
        
        centroid_lidar_ps = PoseStamped()
        centroid_lidar_ps.header = msg.header
        centroid_lidar_ps.pose.position.x = float(centroid_in_lidar_frame_np[0])
        centroid_lidar_ps.pose.position.y = float(centroid_in_lidar_frame_np[1])
        centroid_lidar_ps.pose.position.z = float(centroid_in_lidar_frame_np[2])
        centroid_lidar_ps.pose.orientation.w = 1.0

        try:
            # Ensure msg.header.stamp is valid for TF
            # If stamp is (0,0), TF will use the oldest available transform, which might be wrong.
            # Using self.get_clock().now() if stamp is (0,0) is a pragmatic choice if use_sim_time=true
            # and the data is meant to be "current".
            tf_lookup_time = msg.header.stamp
            if tf_lookup_time.sec == 0 and tf_lookup_time.nanosec == 0:
                self.get_logger().warn(
                    f"Pointcloud header stamp is (0,0) for frame '{msg.header.frame_id}'. "
                    f"Using current ROS time for TF lookup. Ensure upstream node sets a valid timestamp."
                )
                tf_lookup_time = RclpyTime().to_msg() # Get current ROS time from the node's clock by default


            centroid_in_map_ps = self.tf_buffer.transform(
                centroid_lidar_ps, self.map_frame, timeout=Duration(seconds=0.5)
            ) # Removed explicit tf_lookup_time from transform() as PoseStamped carries its own header.stamp
              # The Buffer will use the stamp from centroid_lidar_ps.header

            perceived_target_in_map = [
                centroid_in_map_ps.pose.position.x,
                centroid_in_map_ps.pose.position.y,
                centroid_in_map_ps.pose.position.z
            ]
            self.get_logger().info(f"TF: Perceived target centroid in lidar: {centroid_in_lidar_frame_np.tolist()}. Transformed to '{self.map_frame}': {perceived_target_in_map}.")
        except Exception as e:
            self.get_logger().error(
                f"TF: Failed to transform perceived centroid from '{msg.header.frame_id}' (stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}) to '{self.map_frame}': {e}\n"
                f"{traceback.format_exc()}"
            )
            return

        with self.data_lock:
            closest_global_wp_idx = -1
            min_dist_to_global_wp_2d = float('inf')

            for i, gw_pos_map in enumerate(self.global_waypoints):
                dist_2d = euclidean_distance_2d(perceived_target_in_map, gw_pos_map)
                if dist_2d < self.association_radius_2d and dist_2d < min_dist_to_global_wp_2d:
                    min_dist_to_global_wp_2d = dist_2d
                    closest_global_wp_idx = i
            
            if closest_global_wp_idx != -1:
                self.get_logger().info(f"ASSOCIATION: Perceived target associated with global_wp_idx {closest_global_wp_idx} (2D dist: {min_dist_to_global_wp_2d:.2f}m).")
                
                should_update = True
                if closest_global_wp_idx in self.refined_waypoints:
                    if max_points_in_voxel <= self.refined_waypoints[closest_global_wp_idx].get('source_points_count', 0):
                        should_update = False
                        self.get_logger().info(f"UPDATE: Existing refinement for GW_idx {closest_global_wp_idx} (pts: {self.refined_waypoints[closest_global_wp_idx].get('source_points_count',0)}) is better/equal to current (pts: {max_points_in_voxel}). Not updating.")
                
                if should_update:
                    self.refined_waypoints[closest_global_wp_idx] = {
                        'pose_map': perceived_target_in_map,
                        'source_points_count': max_points_in_voxel,
                        'stamp': msg.header.stamp # Store the PointCloud's original stamp
                    }
                    self.get_logger().info(f"UPDATE: Refined waypoint for global_wp_idx {closest_global_wp_idx} UPDATED/ADDED with {max_points_in_voxel} points, stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
                    self.save_refined_waypoints()
                
            else:
                self.get_logger().info(f"ASSOCIATION: Perceived target (at map_coords {perceived_target_in_map[0]:.2f},{perceived_target_in_map[1]:.2f}) not close enough to any global waypoint (radius {self.association_radius_2d}m).")

    def publish_all_markers_callback(self):
        self.get_logger().info("!!!!!!!! PUBLISH ALL MARKERS CALLBACK TRIGGERED !!!!!!!!"); # <--- ADD THIS
        self.get_logger().debug(f"Publish_all_markers_callback. Global waypoints: {len(self.global_waypoints)}, Refined waypoints: {len(self.refined_waypoints)}")
        marker_array = MarkerArray()
        marker_id_counter = 0
        
        # Use current ROS time for static global waypoints as their "display time"
        current_display_time = self.get_clock().now().to_msg()

        for i, gw_pos in enumerate(self.global_waypoints):
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = current_display_time
            marker.ns = "global_waypoints"
            marker.id = marker_id_counter; marker_id_counter += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = gw_pos[0]
            marker.pose.position.y = gw_pos[1]
            marker.pose.position.z = gw_pos[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.6; marker.scale.y = 0.6; marker.scale.z = 0.6
            marker.color.r = 0.2; marker.color.g = 0.2; marker.color.b = 1.0; marker.color.a = 0.7
            marker.lifetime = Duration(seconds=0).to_msg()
            marker_array.markers.append(marker)

        with self.data_lock:
            for gw_idx_str, refined_data in self.refined_waypoints.items():
                gw_idx = int(gw_idx_str) 
                refined_pos_map = refined_data['pose_map']
                # Use the stored stamp from when the refinement was made, or current time as fallback
                refined_marker_stamp = refined_data.get('stamp', current_display_time)

                marker = Marker()
                marker.header.frame_id = self.map_frame
                marker.header.stamp = refined_marker_stamp # Use specific stamp for this refinement
                marker.ns = "refined_waypoints"
                marker.id = gw_idx
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = refined_pos_map[0]
                marker.pose.position.y = refined_pos_map[1]
                marker.pose.position.z = refined_pos_map[2]
                marker.pose.orientation.w = 1.0
                marker.scale.x = self.voxel_size; marker.scale.y = self.voxel_size; marker.scale.z = self.voxel_size
                marker.color.r = 0.1; marker.color.g = 0.8; marker.color.b = 0.1; marker.color.a = 0.9
                marker.lifetime = Duration(seconds=0).to_msg()
                marker_array.markers.append(marker)

                if gw_idx < len(self.global_waypoints):
                    global_pos_map = self.global_waypoints[gw_idx]
                    line_marker = Marker()
                    line_marker.header.frame_id = self.map_frame
                    line_marker.header.stamp = refined_marker_stamp # Match refined marker's stamp
                    line_marker.ns = "refinement_lines"
                    line_marker.id = gw_idx
                    line_marker.type = Marker.LINE_STRIP
                    line_marker.action = Marker.ADD
                    line_marker.scale.x = 0.07
                    line_marker.color.r = 0.9; line_marker.color.g = 0.9; line_marker.color.b = 0.1; line_marker.color.a = 0.6
                    
                    p1 = RosPoint(); p1.x = global_pos_map[0]; p1.y = global_pos_map[1]; p1.z = global_pos_map[2]
                    p2 = RosPoint(); p2.x = refined_pos_map[0]; p2.y = refined_pos_map[1]; p2.z = refined_pos_map[2]
                    line_marker.points.append(p1)
                    line_marker.points.append(p2)
                    line_marker.lifetime = Duration(seconds=0).to_msg()
                    marker_array.markers.append(line_marker)
        
        # After all markers are added to marker_array:
        num_global_markers = sum(1 for m in marker_array.markers if m.ns == "global_waypoints")
        num_refined_markers = sum(1 for m in marker_array.markers if m.ns == "refined_waypoints")
        num_line_markers = sum(1 for m in marker_array.markers if m.ns == "refinement_lines")
        self.get_logger().info(
            f"MARKER_PUB: Total markers in array: {len(marker_array.markers)}. "
            f"Global: {num_global_markers}, Refined: {num_refined_markers}, Lines: {num_line_markers}."
        )

        if marker_array.markers:
            self.marker_publisher.publish(marker_array)
            self.get_logger().debug(f"Published {len(marker_array.markers)} total markers (global, refined, lines).")

    def load_refined_waypoints(self):
        with self.data_lock:
            if os.path.exists(self.refined_waypoints_save_path):
                try:
                    with open(self.refined_waypoints_save_path, 'r') as f:
                        content = f.read()
                        if not content.strip():
                            self.get_logger().warn(f"Refined waypoints file '{self.refined_waypoints_save_path}' is empty. Starting fresh.")
                            self.refined_waypoints = {}
                            return

                        loaded_data = json.loads(content)
                        temp_refined_waypoints = {}
                        for k, v_dict in loaded_data.items():
                            # Reconstruct Time object if stamp components are present
                            if 'stamp' in v_dict and isinstance(v_dict['stamp'], dict) and \
                               'sec' in v_dict['stamp'] and 'nanosec' in v_dict['stamp']:
                                v_dict['stamp'] = TimeMsg(sec=v_dict['stamp']['sec'], nanosec=v_dict['stamp']['nanosec'])
                            else:
                                # Fallback for older files or if stamp is missing/malformed
                                self.get_logger().warn(f"Waypoint {k} in save file missing valid stamp. Defaulting to current time on load.")
                                v_dict['stamp'] = self.get_clock().now().to_msg() # Or handle as error / skip
                            temp_refined_waypoints[int(k)] = v_dict
                        self.refined_waypoints = temp_refined_waypoints
                    self.get_logger().info(f"Loaded {len(self.refined_waypoints)} refined waypoints from '{self.refined_waypoints_save_path}'")
                except json.JSONDecodeError:
                     self.get_logger().error(f"Error decoding JSON from '{self.refined_waypoints_save_path}'. File might be corrupt. Starting fresh.")
                     self.refined_waypoints = {}
                except Exception as e:
                    self.get_logger().error(f"Failed to load refined waypoints from '{self.refined_waypoints_save_path}': {e}\n{traceback.format_exc()}")
                    self.refined_waypoints = {}
            else:
                self.get_logger().warn(f"Refined waypoints file not found: '{self.refined_waypoints_save_path}'. Starting with empty set.")
                self.refined_waypoints = {}

    def save_refined_waypoints(self):
        self.get_logger().info(f"!!!!!! SAVE REFINED WAYPOINTS CALLED with {len(self.refined_waypoints)} entries !!!!!!"); # Outer call log
        with self.data_lock:
            try:
                self.get_logger().info("SAVE_DEBUG: Entered try block.") # Log S1
                dir_name = os.path.dirname(self.refined_waypoints_save_path)
                self.get_logger().info(f"SAVE_DEBUG: dir_name='{dir_name}'") # Log S2

                if dir_name and not os.path.exists(dir_name):
                    self.get_logger().info(f"SAVE_DEBUG: Directory '{dir_name}' does not exist. Attempting to create.") # Log S3
                    os.makedirs(dir_name, exist_ok=True)
                    self.get_logger().info(f"SAVE_DEBUG: Directory creation/check complete for: '{dir_name}'") # Log S4
                elif not dir_name:
                    self.get_logger().info(f"SAVE_DEBUG: No directory path in save_path ('{self.refined_waypoints_save_path}'), saving to current dir.") # Log S3.1
                else:
                    self.get_logger().info(f"SAVE_DEBUG: Directory '{dir_name}' already exists.") # Log S3.2

                
                self.get_logger().info("SAVE_DEBUG: Preparing data for JSON serialization...") # Log S5
                serializable_refined_waypoints = {}
                if not self.refined_waypoints: # Handle case where dict is empty
                    self.get_logger().info("SAVE_DEBUG: refined_waypoints dictionary is empty. Nothing to serialize for saving.")
                
                for k, v_dict in self.refined_waypoints.items(): # This loop won't run if dict is empty
                    self.get_logger().debug(f"SAVE_DEBUG: Processing waypoint key {k}, data: {v_dict}") # Log S6
                    data_to_save = v_dict.copy()
                    if 'stamp' in data_to_save:
                        if isinstance(data_to_save['stamp'], TimeMsg):
                            self.get_logger().debug(f"SAVE_DEBUG: Converting TimeMsg stamp for key {k}") # Log S7
                            data_to_save['stamp'] = {'sec': data_to_save['stamp'].sec, 'nanosec': data_to_save['stamp'].nanosec}
                        else:
                            self.get_logger().warn(f"SAVE_DEBUG: Waypoint key {k} has a non-TimeMsg stamp: {type(data_to_save['stamp'])}. Value: {data_to_save['stamp']}. Saving as is.")
                    else:
                        self.get_logger().warn(f"SAVE_DEBUG: Waypoint key {k} is missing 'stamp' field. This is unexpected.")
                    serializable_refined_waypoints[k] = data_to_save
                    self.get_logger().debug(f"SAVE_DEBUG: Prepared serializable data for key {k}: {data_to_save}") # Log S8
                
                self.get_logger().info(f"SAVE_DEBUG: Data preparation complete. Serialized dict has {len(serializable_refined_waypoints)} items. About to open file '{self.refined_waypoints_save_path}' for writing.") # Log S9
                
                with open(self.refined_waypoints_save_path, 'w') as f:
                    self.get_logger().info("SAVE_DEBUG: File opened. Attempting json.dump...") # Log S10
                    json.dump(serializable_refined_waypoints, f, indent=4)
                    self.get_logger().info("SAVE_DEBUG: json.dump complete.") # Log S11
                
                self.get_logger().info(f"SAVE_DEBUG: SAVE ATTEMPT (after file close): JSON dump complete to '{self.refined_waypoints_save_path}'. File should now exist/be updated.") # Log S12

                self.get_logger().info(f"Successfully saved {len(self.refined_waypoints)} refined waypoints to '{self.refined_waypoints_save_path}'") # Log S13 (Original success log)
            except Exception as e:
                self.get_logger().error(f"SAVE_DEBUG: FAILED to save refined waypoints to '{self.refined_waypoints_save_path}': {e}\n{traceback.format_exc()}")

    def on_shutdown(self):
        self.get_logger().info("Node shutting down. Attempting final save of refined waypoints.")
        self.save_refined_waypoints()

def main(args=None):
    node = None
    rclpy.init(args=args)
    try:
        node = WaypointRefinementNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Log is attempted, might fail if context is already invalid
        if node and node.context.ok(): node.get_logger().info('Keyboard interrupt received by WaypointRefinementNode.')
        else: print('Keyboard interrupt received (node context potentially invalid or node not fully up).')
    except Exception as e:
        if node and node.context.ok(): node.get_logger().fatal(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        else: print(f"Unhandled exception (node context potentially invalid or node not fully up): {e}\n{traceback.format_exc()}")
    finally:
        if node:
            # Attempt to log, understanding it might fail if context is gone
            if node.context.ok(): node.get_logger().info('Executing final shutdown sequence for WaypointRefinementNode...')
            else: print('Executing final shutdown sequence for WaypointRefinementNode (logger context likely invalid)...')
            
            node.on_shutdown() # This will attempt its own logging for saving

            if node.context.ok():
                 node.destroy_node()
                 if node.context.ok(): node.get_logger().info('WaypointRefinementNode destroyed.') # Check again before logging
                 else: print('WaypointRefinementNode destroyed (logger context likely invalid).')
            else:
                if node.context.ok(): node.get_logger().warn("RCLPY context was not OK during node destruction for WaypointRefinementNode.")
                else: print("RCLPY context was not OK during node destruction for WaypointRefinementNode (logger context likely invalid).")
        else:
            print("WaypointRefinementNode object was not created successfully or an early error occurred.")

        if rclpy.ok():
            rclpy.shutdown()
        print("WaypointRefinementNode rclpy shutdown complete.")

if __name__ == '__main__':
    main()