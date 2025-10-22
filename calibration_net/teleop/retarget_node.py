from __future__ import annotations

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray

# Project‑specific utilities
from calibration_net.policy.model import load_model
from calibration_net.policy.inferencer import ModelInferencer
from queue import Empty
import multiprocessing
import yaml
import tyro

def process_hand_marker_array(msg):
    joints_number = 21

    try:
        # Create mapping from marker ID to marker object
        markers_by_id = {marker.id: marker for marker in msg.markers}
        
        # Get wrist marker position and orientation
        wrist_marker = markers_by_id[0]
        wrist_pos = np.array([
            wrist_marker.pose.position.x,
            wrist_marker.pose.position.y,
            wrist_marker.pose.position.z
        ])
        
        wrist_ori = np.array([
            wrist_marker.pose.orientation.x,
            wrist_marker.pose.orientation.y,
            wrist_marker.pose.orientation.z,
            wrist_marker.pose.orientation.w
        ])
        
        # Initialize position array - for storing world coordinates of all joints
        keypoint_3d_array = np.zeros((joints_number, 3))
        keypoint_3d_array[0] = wrist_pos  # Set wrist position
        
        # Initialize orientation data array
        orientations_data = np.zeros((joints_number, 4))
        orientations_data[0] = wrist_ori  # Set wrist orientation
        
        # Process all other joint markers
        for joint_id in range(1, joints_number):
            if joint_id not in markers_by_id:
                # Use default values if marker is missing for a joint
                continue
            
            # Get joint marker position and orientation
            marker = markers_by_id[joint_id]
            keypoint_3d_array[joint_id] = [
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z
            ]
            
            orientations_data[joint_id] = [
                marker.pose.orientation.x,
                marker.pose.orientation.y,
                marker.pose.orientation.z,
                marker.pose.orientation.w
            ]
            
        # 1. Make all joint coordinates relative to wrist
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]

        
        return keypoint_3d_array
        
    except Exception as e:
        print(f"Error processing hand data: {e}")
        return None

# =============================================================================
class RetargetNode(Node):
    def __init__(self, model_dir:str, side:str, publish_topic:str, betas:np.ndarray=None, topic:str="/joints_position", control_frequency:int=60, verbose:bool=False):
        super().__init__(f"model_retarget_node_{side}")

        # ---------------------------- Parameters ---------------------------- #
        self.verbose = verbose
        self.side = side
        self.freq = control_frequency
        self.topic = topic
        self.publish_topic = publish_topic

        self.inferencer = ModelInferencer(model_dir, side, device="cpu", betas=betas, fixed_noise=True)
        self.marker_queue = multiprocessing.Queue(maxsize=2)
        self.marker_sub = self.create_subscription(
            MarkerArray, topic, self.marker_callback, 1
        )

        # ----------------------- ROS 2 Publishers --------------------------- #
        self.pub = self.create_publisher(Float32MultiArray, self.publish_topic, 1)

        # --------------------------- Main Timer ---------------------------- #
        period = 1.0 / float(self.freq)
        self.timer = self.create_timer(period, self.timer_cb)
        self.get_logger().info("Retarget node initialised")

        self.i = 0

    def marker_callback(self, msg):
        # Check if marker array is empty
        if not msg.markers:
            self.get_logger().warn("Received empty marker array")
            return

        # Filter markers based on hand type
        filtered_markers = MarkerArray()
        
        # Get the namespace from the first marker
        if msg.markers and hasattr(msg.markers[0], 'ns'):
            namespace = msg.markers[0].ns
            
            # Check if this is the correct hand type
            is_left_hand = "left" in namespace.lower()
            is_right_hand = "right" in namespace.lower()
            
            # Only process markers for the correct hand
            if (self.side.lower() == "left" and is_left_hand) or (self.side.lower() == "right" and is_right_hand):
                filtered_markers = msg
                self.get_logger().debug(f"Processing {self.side} hand markers, namespace: {namespace}")
            else:
                # Skip markers for the wrong hand
                return
        else:
            # If no namespace, process all markers
            filtered_markers = msg
            
        # Put marker data into queue
        if self.marker_queue.full():
            self.get_logger().warning("Marker queue is full, dropping old data")
            try:
                self.marker_queue.get_nowait()  # Remove oldest item
            except Empty:
                pass
        self.marker_queue.put(filtered_markers)
        self.get_logger().debug(
            f"Marker data put into queue, current queue size: {self.marker_queue.qsize()}"
        )

    # --------------------------------------------------------------------- #
    def timer_cb(self):
        now = self.get_clock().now().to_msg()
        try:
            data = self.marker_queue.get(block=False)

        except Empty:
            return

        joint_positions = process_hand_marker_array(data)
        retargeted_qpos = self.inferencer.inference(joint_positions)
        msg = Float32MultiArray()
        msg.data = retargeted_qpos.tolist()
        self.pub.publish(msg)
        
    # --------------------------------------------------------------------- #
    def destroy_node(self):
        self.get_logger().info("Shutting down tele‑op…")
        super().destroy_node()

# =============================================================================

def main(model_dir:str, side:str, publish_topic:str, beta_path:str=None, topic:str="/joints_position",  control_frequency:int=60, verbose:bool=False):
    if beta_path is not None:
        betas = yaml.load(open(beta_path, 'r'), Loader=yaml.FullLoader)
    else:
        betas = None

    rclpy.init()
    node = RetargetNode(model_dir, side, publish_topic, betas=betas, topic=topic, control_frequency=control_frequency, verbose=verbose)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
