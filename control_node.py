import numpy as np
import rclpy
import kinpy
import xacro
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point

from .ik_controller import IKController


class ControlNode(Node):
    def __init__(self, node_name="control_node"):
        super().__init__(node_name=node_name)

        urdf = xacro.process("iwa14.urdf.xacro")
        rot = np.array([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])                              # -Pi/2 rotation from world to KUKA frame
        pos = np.array([0, -1.9416, -0.44926])                                              # Coordinates of KUKA base in world frame
        world_config = kinpy.Transform(rot, pos)
        self.controller = IKController(urdf, "link_0", "link_ee", world_config)

        self.target = None

        self.counter = 0                                                                    # Samples the joint angles only once every
        self.ignore_states = 100                                                            # 100 frames.

        self.joint_state_sub = self.create_subscription(                                    # Subscribes to current joint angles.
            JointState, "/joint_states", self.on_joint_state, 1
        )

        self.target_sub = self.create_subscription(                                         # Subscribes to the target position.
            Point, "/target_prediction", self.on_target, 10
        )
        
        self.joint_trajectory_pub = self.create_publisher(                                  # Publishes a trajectory to the KUKA.
            JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 1
        )


    def on_target(self, target):                                                            # Stores the target whenever it is updated.
        self.target = np.array([target.x, target.y, target.z])


    def on_joint_state(self, joint_state):
        self.counter += 1                                                                   # Downsamples the joint angles topic to
        if self.counter % self.ignore_states != 0:                                          # avoid overwhelming the KUKA with commands.
            return

        if self.target is None:                                                             # Send no command if there is no target found.
            return

        joint_trajectory = JointTrajectory()                                                # Create a trajectory with a single point.
        joint_trajectory.joint_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
        point = JointTrajectoryPoint()

        position, required_time = self.controller.step_control(self.target, joint_state)    # Get joint angles and required time from controller.

        if position is None:                                                                # Send no command if the error is already small.
            return

        point.positions = position.tolist()
        point.time_from_start.sec = int(required_time)                                      # Convert the required time into seconds
        nanosecs = int((required_time - int(required_time)) * 1e9)                          # and nanoseconds, with nanoseconds in
        point.time_from_start.nanosec = max(min(nanosecs, 4294967295), 0)                   # the range [0, 4294967296).
        joint_trajectory.points = [point]

        self.joint_trajectory_pub.publish(joint_trajectory)


def main(args=None):
    rclpy.init(args=args)
    control_node = ControlNode()
    rclpy.spin(control_node)
    rclpy.shutdown()
