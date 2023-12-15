import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point


class VisualServoNode(Node):
    def __init__(self, node_name="visual_servo_node"):
        super().__init__(node_name=node_name)

        self.paddle_z_offset = 0.19                                                         # Distance from center of paddle to EE tip is 19cm
        self.paddle_y = -1.65                                                               # Half-table length is 137cm, but we add a bit extra
                                                                                            # to avoid collision with the table

        self.prev_targets = []                                                              # The low pass filter is a moving average with
        self.moving_average_size = 5                                                        # size 5.
        
        self.bse_sub = self.create_subscription(                                            # Subscribes to the ball position estimate.
            Point, "/ball_state_estimate", self.on_bse, 10
        )
        
        self.target_pub = self.create_publisher(                                            # Publishes a target position.
            Point, "/target_prediction", 10
        )


    def on_bse(self, ball):
        target = Point()
        target.x = ball.x / 100                                                             # Convert x from cm to m.
        target.y = self.paddle_y                                                            # Keeps the y coordinate of the paddle constant.
        target.z = (ball.z / 100) - self.paddle_z_offset                                    # Convert z from cm to m, and shift by an offset.

        target.x = max(min(target.x, 0.6), -0.6)                                            # Keeps x coordinate of paddle between -0.6 and +0.6.
        target.z = max(min(target.z, 0.85), -0.1)                                           # Keeps z coordinate of paddle between 0.35 and 1.3
        self.prev_targets.append(target)                                                    # (the coordinates are in the world frame).

        if len(self.prev_targets) < self.moving_average_size:
            return
        if len(self.prev_targets) > self.moving_average_size:
            self.prev_targets.pop(0)

        final_target = Point()                                                              # Computes the moving average over the last 5 targets.
        final_target.x = sum(t.x for t in self.prev_targets) / self.moving_average_size     
        final_target.y = sum(t.y for t in self.prev_targets) / self.moving_average_size
        final_target.z = sum(t.z for t in self.prev_targets) / self.moving_average_size

        self.target_pub.publish(final_target)


def main(args=None):
    rclpy.init(args=args)
    visual_servo_node = VisualServoNode()
    rclpy.spin(visual_servo_node)
    rclpy.shutdown()
