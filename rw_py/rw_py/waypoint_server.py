#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import yaml
from pathlib import Path
import os
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rw_interfaces.srv import GetWaypoints

class WaypointServer(Node):
    def __init__(self):
        super().__init__('waypoint_server_node')
        
        default_yaml_path = os.path.join(
            get_package_share_directory('rw'), 'config', 'nav_waypoints.yaml'
        )
        self.declare_parameter('waypoints_yaml_path', default_yaml_path)
        
        self.all_waypoints_ = []
        self.load_waypoints()

        self.srv = self.create_service(GetWaypoints, 'get_waypoints', self.get_waypoints_callback)
        self.get_logger().info('Waypoint Server is ready and serving waypoints.')

    def load_waypoints(self):
        yaml_path = self.get_parameter('waypoints_yaml_path').get_parameter_value().string_value
        self.get_logger().info(f"Loading waypoints from: {yaml_path}")
        try:
            with open(yaml_path, 'r') as file:
                yaml_data = yaml.safe_load(file)
            if not yaml_data or 'poses' not in yaml_data:
                self.get_logger().error(f"YAML '{yaml_path}' is empty or has no 'poses' key.")
                return
            
            for i, pose_entry in enumerate(yaml_data['poses']):
                ps_msg = PoseStamped()
                ps_msg.header.frame_id = pose_entry.get('header', {}).get('frame_id', 'map')
                # Stamp will be set by consumers if needed
                pose_block = pose_entry.get('pose', {})
                pos = pose_block.get('position', {})
                orient = pose_block.get('orientation', {})
                ps_msg.pose.position.x = float(pos.get('x', 0.0))
                ps_msg.pose.position.y = float(pos.get('y', 0.0))
                ps_msg.pose.position.z = float(pos.get('z', 0.0))
                ps_msg.pose.orientation.x = float(orient.get('x', 0.0))
                ps_msg.pose.orientation.y = float(orient.get('y', 0.0))
                ps_msg.pose.orientation.z = float(orient.get('z', 0.0))
                ps_msg.pose.orientation.w = float(orient.get('w', 1.0))
                self.all_waypoints_.append(ps_msg)

            self.get_logger().info(f"Successfully loaded {len(self.all_waypoints_)} waypoints.")
        except Exception as e:
            self.get_logger().error(f"Failed to load or parse YAML file: {e}")

    def get_waypoints_callback(self, request, response):
        if self.all_waypoints_:
            response.waypoints = self.all_waypoints_
            response.success = True
            response.message = f"Successfully returned {len(self.all_waypoints_)} waypoints."
        else:
            response.success = False
            response.message = "No waypoints have been loaded into the server."
        return response

def main(args=None):
    rclpy.init(args=args)
    node = WaypointServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()