#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time as RclpyTime
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point as RosPoint, PoseStamped
from std_msgs.msg import ColorRGBA, Header
from builtin_interfaces.msg import Time as TimeMsg

import numpy as np
import os
import csv # Import CSV module
import yaml
import threading
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs
import traceback

def euclidean_distance_2d(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

class WaypointRefinementNode(Node):
    def __init__(self):
        super().__init__('waypoint_refinement_node')

        self.declare_parameter('input_pc_topic', 'selected_lidar_points')
        self.declare_parameter('output_marker_topic', 'refined_waypoint_markers')
        self.declare_parameter('global_waypoints_path', '/home/aldoghry/test_ws/src/rw/config/nav_waypoints.yaml')
        # Changed default to .csv
        self.declare_parameter('refined_waypoints_save_path', '/home/aldoghry/test_ws/src/rw_py/founds/refined_waypoints.csv')
        self.declare_parameter('voxel_size', 0.3)
        self.declare_parameter('association_radius_2d', 5.0)
        self.declare_parameter('min_points_for_refinement', 1)
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

        self.global_waypoints = []
        self.refined_waypoints = {} # Key: global_wp_idx (int), Value: {'pose_map': [x,y,z], 'source_points_count': N, 'stamp': TimeMsg}
        self.data_lock = threading.Lock()
        self.last_pc_header = None

        self.load_global_waypoints()
        self.load_refined_waypoints() # Will now use CSV load logic

        self.pc_subscriber = self.create_subscription(
            PointCloud2,
            self.input_pc_topic,
            self.point_cloud_callback,
            rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        )
        self.marker_publisher = self.create_publisher(MarkerArray, self.output_marker_topic, 10)
        self.marker_publish_timer = self.create_timer(1.0, self.publish_all_markers_callback)

        self.get_logger().info(
            f"Waypoint Refinement Node initialized (CSV persistence).\n" # Indicate CSV
            f"\tGlobal Waypoints: {len(self.global_waypoints)} loaded (after duplicates removed) from '{self.global_waypoints_path}'\n"
            f"\tRefined Waypoints Loaded: {len(self.refined_waypoints)} from '{self.refined_waypoints_save_path}'\n"
            f"\tLiDAR Input: '{self.input_pc_topic}'\n"
            f"\tAssociation Radius (2D): {self.association_radius_2d}m\n"
            f"\tMin Points for Refinement: {self.min_points_for_refinement}"
        )

    # load_global_waypoints remains the same (uses YAML)
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

    # point_cloud_callback remains largely the same, calls new save_refined_waypoints
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
            tf_lookup_time = msg.header.stamp # Use the stamp from the PCL message
            if tf_lookup_time.sec == 0 and tf_lookup_time.nanosec == 0:
                self.get_logger().warn(
                    f"Pointcloud header stamp is (0,0) for frame '{msg.header.frame_id}'. "
                    f"Using current ROS time for TF lookup. Ensure upstream node sets a valid timestamp."
                )
                # If using current time, it should be from the node's clock
                # centroid_lidar_ps.header.stamp = self.get_clock().now().to_msg() # Option: overwrite header for transform

            centroid_in_map_ps = self.tf_buffer.transform(
                centroid_lidar_ps, self.map_frame, timeout=Duration(seconds=0.5)
            )
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
                        'stamp': msg.header.stamp
                    }
                    self.get_logger().info(f"UPDATE: Refined waypoint for global_wp_idx {closest_global_wp_idx} UPDATED/ADDED with {max_points_in_voxel} points, stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec % 1000000000:09d}")
                    self.save_refined_waypoints()
                
            else:
                self.get_logger().info(f"ASSOCIATION: Perceived target (at map_coords {perceived_target_in_map[0]:.2f},{perceived_target_in_map[1]:.2f}) not close enough to any global waypoint (radius {self.association_radius_2d}m).")

    # publish_all_markers_callback remains the same
    def publish_all_markers_callback(self):
        self.get_logger().info("!!!!!!!! PUBLISH ALL MARKERS CALLBACK TRIGGERED !!!!!!!!");
        self.get_logger().debug(f"Publish_all_markers_callback. Global waypoints: {len(self.global_waypoints)}, Refined waypoints: {len(self.refined_waypoints)}")
        marker_array = MarkerArray()
        marker_id_counter = 0
        
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
                refined_marker_stamp = refined_data.get('stamp', current_display_time)

                marker = Marker()
                marker.header.frame_id = self.map_frame
                marker.header.stamp = refined_marker_stamp
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
                    line_marker.header.stamp = refined_marker_stamp
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
        # CSV: global_wp_idx, x, y, z, points_count, stamp_sec, stamp_nanosec
        csv_header = ['global_wp_idx', 'x', 'y', 'z', 'points_count', 'stamp_sec', 'stamp_nanosec']
        with self.data_lock:
            self.refined_waypoints = {} # Clear existing before load
            if os.path.exists(self.refined_waypoints_save_path):
                try:
                    with open(self.refined_waypoints_save_path, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        # Check if header matches expected (optional but good for robustness)
                        if not reader.fieldnames or any(h not in reader.fieldnames for h in csv_header):
                            self.get_logger().warn(f"CSV file '{self.refined_waypoints_save_path}' has unexpected header or is empty. Will attempt to read anyway or start fresh.")
                            # If you want to be strict, you could return here.

                        for i, row in enumerate(reader):
                            try:
                                gw_idx = int(row['global_wp_idx'])
                                pose_map = [float(row['x']), float(row['y']), float(row['z'])]
                                points_count = int(row['points_count'])
                                stamp_sec = int(row['stamp_sec'])
                                stamp_nanosec = int(row['stamp_nanosec'])
                                
                                self.refined_waypoints[gw_idx] = {
                                    'pose_map': pose_map,
                                    'source_points_count': points_count,
                                    'stamp': TimeMsg(sec=stamp_sec, nanosec=stamp_nanosec)
                                }
                            except (KeyError, ValueError) as ve:
                                self.get_logger().error(f"Error parsing row {i+1} in CSV '{self.refined_waypoints_save_path}': {ve}. Skipping row.")
                                continue
                    self.get_logger().info(f"Loaded {len(self.refined_waypoints)} refined waypoints from CSV '{self.refined_waypoints_save_path}'")
                except FileNotFoundError: # Should be caught by os.path.exists, but good practice
                     self.get_logger().warn(f"Refined waypoints file (CSV) not found: '{self.refined_waypoints_save_path}'. Starting with empty set.")
                except Exception as e:
                    self.get_logger().error(f"Failed to load refined waypoints from CSV '{self.refined_waypoints_save_path}': {e}\n{traceback.format_exc()}")
                    self.refined_waypoints = {} # Reset on error
            else:
                self.get_logger().warn(f"Refined waypoints file (CSV) not found: '{self.refined_waypoints_save_path}'. Starting with empty set.")


    def save_refined_waypoints(self):
        # CSV: global_wp_idx, x, y, z, points_count, stamp_sec, stamp_nanosec
        csv_header = ['global_wp_idx', 'x', 'y', 'z', 'points_count', 'stamp_sec', 'stamp_nanosec']
        self.get_logger().info(f"!!!!!! SAVE REFINED WAYPOINTS (CSV) CALLED with {len(self.refined_waypoints)} entries !!!!!!");
        with self.data_lock:
            try:
                self.get_logger().info("SAVE_DEBUG: Entered try block for CSV save.")
                dir_name = os.path.dirname(self.refined_waypoints_save_path)
                self.get_logger().info(f"SAVE_DEBUG: dir_name='{dir_name}'")

                if dir_name and not os.path.exists(dir_name):
                    self.get_logger().info(f"SAVE_DEBUG: Directory '{dir_name}' does not exist. Attempting to create.")
                    os.makedirs(dir_name, exist_ok=True)
                    self.get_logger().info(f"SAVE_DEBUG: Directory creation/check complete for: '{dir_name}'")
                elif not dir_name:
                     self.get_logger().info(f"SAVE_DEBUG: No directory path, saving CSV to current dir.")
                else:
                    self.get_logger().info(f"SAVE_DEBUG: Directory '{dir_name}' already exists.")
                
                self.get_logger().info(f"SAVE_DEBUG: Preparing data for CSV serialization... {len(self.refined_waypoints)} items.")
                
                rows_to_write = []
                for gw_idx, data_dict in self.refined_waypoints.items():
                    self.get_logger().debug(f"SAVE_DEBUG: Processing CSV row for gw_idx {gw_idx}")
                    row = {
                        'global_wp_idx': gw_idx,
                        'x': data_dict['pose_map'][0],
                        'y': data_dict['pose_map'][1],
                        'z': data_dict['pose_map'][2],
                        'points_count': data_dict['source_points_count'],
                        'stamp_sec': data_dict['stamp'].sec,
                        'stamp_nanosec': data_dict['stamp'].nanosec
                    }
                    rows_to_write.append(row)
                    self.get_logger().debug(f"SAVE_DEBUG: Prepared CSV row: {row}")

                self.get_logger().info(f"SAVE_DEBUG: CSV Data preparation complete. About to open file '{self.refined_waypoints_save_path}' for writing.")
                
                # newline='' is important for csv writer on some platforms
                with open(self.refined_waypoints_save_path, 'w', newline='') as f:
                    self.get_logger().info("SAVE_DEBUG: CSV File opened. Attempting csv.DictWriter...")
                    writer = csv.DictWriter(f, fieldnames=csv_header)
                    writer.writeheader()
                    self.get_logger().info("SAVE_DEBUG: CSV Header written.")
                    writer.writerows(rows_to_write)
                    self.get_logger().info(f"SAVE_DEBUG: CSV {len(rows_to_write)} rows written.")
                
                self.get_logger().info(f"SAVE_DEBUG: SAVE ATTEMPT (after CSV file close): CSV write complete to '{self.refined_waypoints_save_path}'. File should now exist/be updated.")
                self.get_logger().info(f"Successfully saved {len(self.refined_waypoints)} refined waypoints to CSV '{self.refined_waypoints_save_path}'")
            except Exception as e:
                self.get_logger().error(f"SAVE_DEBUG: FAILED to save refined waypoints to CSV '{self.refined_waypoints_save_path}': {e}\n{traceback.format_exc()}")

    def on_shutdown(self):
        self.get_logger().info("Node shutting down. Attempting final save of refined waypoints (CSV).")
        self.save_refined_waypoints()

# main function remains the same
def main(args=None):
    node = None
    rclpy.init(args=args)
    try:
        node = WaypointRefinementNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node and node.context.ok(): node.get_logger().info('Keyboard interrupt received by WaypointRefinementNode.')
        else: print('Keyboard interrupt received (node context potentially invalid or node not fully up).')
    except Exception as e:
        if node and node.context.ok(): node.get_logger().fatal(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        else: print(f"Unhandled exception (node context potentially invalid or node not fully up): {e}\n{traceback.format_exc()}")
    finally:
        if node:
            if node.context.ok(): node.get_logger().info('Executing final shutdown sequence for WaypointRefinementNode...')
            else: print('Executing final shutdown sequence for WaypointRefinementNode (logger context likely invalid)...')
            
            node.on_shutdown()

            if node.context.ok():
                 node.destroy_node()
                 if node.context.ok(): node.get_logger().info('WaypointRefinementNode destroyed.')
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