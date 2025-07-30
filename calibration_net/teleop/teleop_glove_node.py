#!/usr/bin/env python3
"""ROS 2 node: OpenCyberGlove → InspireHand tele‑op

Features
--------
* Streams glove data at `control_frequency` Hz.
* Predicts joint angles with a loaded Torch model.
* Publishes `/left_hand/joint_states` and `/right_hand/joint_states` as
  `sensor_msgs/JointState`.
* Optionally drives Inspire Hand robots in real time.
* On‑startup *min/max* calibration available via the `calibration` parameter.

Run
---
colcon build --packages-select teleop_glove
source install/setup.bash
ros2 run teleop_glove teleop_glove_node \
        --ros-args -p left_port:=/dev/ttyACM0 -p model_dir:=/path/to/model \
        -p calibration:=true
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray

from tqdm import tqdm

from open_cyber_glove.sdk import OpenCyberGlove
from inspire_hand.hand import FingerID
from inspire_hand import InspireHand

# Project‑specific utilities
from calibration_net.policy.model import load_model
from calibration_net.utils import (
    initilize_glove,
    initialize_robot,
    apply_angles,
    FrameRateContext,
)

# =============================================================================
# Calibration helper (unchanged from original script)
# =============================================================================

def calibrate_glove(sdk: OpenCyberGlove, *, hand_type: str = "left", samples_min_max: int = 1000):
    """Collect *samples_min_max* frames and compute per‑channel min/max/mean/std."""
    tensile_data = []
    print(f"[{hand_type}] Calibration Pose 1: Make a fist and open your hand multiple times. Press Enter to continue…")
    input("")
    for _ in tqdm(range(samples_min_max), desc=f"[{hand_type}] min/max calibration"):
        data = sdk.get_data(hand_type=hand_type).tensile_data
        tensile_data.append(data)

    tensile_data = np.array(tensile_data)
    mean = np.mean(tensile_data, axis=0)
    std  = np.std(tensile_data,  axis=0)
    max_ = np.max(tensile_data,  axis=0)
    min_ = np.min(tensile_data,  axis=0)

    return {
        "x": {
            "mean": mean,
            "std":  std,
            "min":  min_,
            "max":  max_,
        }
    }

# =============================================================================
class GloveTeleopNode(Node):
    def __init__(self):
        super().__init__("glove_teleop_node")

        # ---------------------------- Parameters ---------------------------- #
        self.declare_parameters(
            namespace="",
            parameters=[
                ("left_port", ""),          # default None → empty string
                ("right_port", ""),
                ("left_robot_port", ""),
                ("right_robot_port", ""),
                ("robot_speed", 0.4),
                ("control_frequency", 30),
                ("model_dir", ""),
                ("calibration", False),
            ],
        )
        p = self.get_parameter
        self.freq = p("control_frequency").value

        # -------------------------- SDK Initialisation ---------------------- #
        self.get_logger().info("Initialising gloves…")
        self.glove_sdk = initilize_glove(
            left_port=p("left_port").value or None,
            right_port=p("right_port").value or None,
            calibrate=False,
        )

        self.left_robot: Optional[InspireHand] = None
        self.right_robot: Optional[InspireHand] = None
        if p("left_robot_port").value:
            self.left_robot = initialize_robot(
                port=p("left_robot_port").value,
                speed=p("robot_speed").value,
            )
        if p("right_robot_port").value:
            self.right_robot = initialize_robot(
                port=p("right_robot_port").value,
                speed=p("robot_speed").value,
            )

        self.model, self.norm_stats, _ = load_model(p("model_dir").value, device="cpu")

        # ---------------------------- Calibration --------------------------- #
        if p("calibration").value:
            if self.glove_sdk.left_glove is not None:
                new_stats = calibrate_glove(self.glove_sdk, hand_type="left")
                self.get_logger().info("[left] calibration done")
                self.norm_stats["x"] = new_stats["x"]
            if self.glove_sdk.right_glove is not None:
                new_stats = calibrate_glove(self.glove_sdk, hand_type="right")
                self.get_logger().info("[right] calibration done")
                self.norm_stats["x"] = new_stats["x"]

        # ----------------------- ROS 2 Publishers --------------------------- #
        self.pub_left = self.create_publisher(Float32MultiArray, "/retargeted_qpos", 1)
        self.pub_right = self.create_publisher(Float32MultiArray, "/retargeted_qpos", 1)

        # --------------------------- Main Timer ---------------------------- #
        period = 1.0 / float(self.freq)
        self.timer = self.create_timer(period, self.timer_cb)
        self.get_logger().info("Glove tele‑op node initialised")

    # --------------------------------------------------------------------- #
    def timer_cb(self):
        now = self.get_clock().now().to_msg()

        # LEFT HAND --------------------------------------------------------- #
        if self.glove_sdk.left_glove is not None:
            data = self.glove_sdk.get_data(hand_type="left").tensile_data
            angles = self.model.predict(np.array(data), self.norm_stats, norm_type="meanstd").tolist()
            
            qpos = [1 - np.clip(angle, 0, 1.0) for angle in angles]
            msg = Float32MultiArray()
            msg.data = qpos
            self.pub_left.publish(msg)

            if self.left_robot is not None:
                apply_angles(self.left_robot, angles, wait_for_response=False)

        # RIGHT HAND -------------------------------------------------------- #
        if self.glove_sdk.right_glove is not None:
            data = self.glove_sdk.get_data(hand_type="right").tensile_data
            angles = self.model.predict(np.array(data), self.norm_stats, norm_type="meanstd").tolist()
            
            qpos = [1 - np.clip(angle, 0, 1.0) for angle in angles]
            msg = Float32MultiArray()
            msg.data = qpos
            
            self.pub_right.publish(msg)

            if self.right_robot is not None:
                apply_angles(self.right_robot, angles, wait_for_response=False)

    # --------------------------------------------------------------------- #
    def destroy_node(self):
        self.get_logger().info("Shutting down tele‑op…")
        self.glove_sdk.stop()
        if self.left_robot:
            self.left_robot.close()
        if self.right_robot:
            self.right_robot.close()
        super().destroy_node()

# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = GloveTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
