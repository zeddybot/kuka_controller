import kinpy
import numpy as np
from sensor_msgs.msg import JointState

class IKController:
    
    def __init__(self, urdf, base_link="link_0", ee_link="link_ee", world_config=kinpy.Transform()):
        self.chain = kinpy.build_serial_chain_from_urdf(                                                # Gets the kinematics information
            data=urdf,                                                                                  # for the robot from the urdf.
            root_link_name=base_link,
            end_link_name=ee_link,
        )
        self.world_config = world_config.matrix()
        velocity_scale = 0.3
        self.velocity_limits = np.array([1.4, 1.4, 1.7, 1.25, 2.2, 2.3, 2.3]) * velocity_scale          # Computes the limits on velocity
        self.position_limits = np.array([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05])                     # and position.

        self.error_epsilon = 0.01                                                                       # Reduce error to within 1cm.

    def step_control(self, target, joint_state):
        theta = np.array(joint_state.position)                                                          # Get the joint angles.
        theta[2], theta[3] = theta[3], theta[2]                                                         # Fixes an evil bug.
        target_position = (np.linalg.inv(self.world_config) @ np.append(target, 1))[:3]                 # Put the target in the KUKA base frame.
        target_config = kinpy.Transform(np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2]), target_position)  # Orients paddle facing towards table.

        current_config = self.chain.forward_kinematics(theta).matrix()                                  # Get the current configuation
        error_config = np.linalg.inv(current_config) @ target_config.matrix()                           # and the error configuration,
        if np.linalg.norm(error_config[:3, 3]) < self.error_epsilon:                                    # and stops sending commands to KUKA
            return None, None                                                                           # when the error is small.

        target_theta = self.chain.inverse_kinematics(target_config)                                     # Compute goal joint angles using IK.
        target_theta = np.minimum(target_theta, self.position_limits)                                   # Keeps joint angles in safe region.
        target_theta = np.maximum(target_theta, -self.position_limits)

        required_time = np.max(np.abs(theta - target_theta) / self.velocity_limits)                     # Computes minimum time to execute move
        return target_theta, required_time                                                              # given limits on joint velocity.
