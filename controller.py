import kinpy
import numpy as np
from lbr_fri_msgs.msg import LBRPositionCommand, LBRState
from typing import Tuple

class WorkspaceVelocityController:
    
    def __init__(
        self,
        robot_description: str,
        base_link: str = "link_0",
        end_effector_link: str = "link_ee",
        world_config: kinpy.Transform,
        Kp: np.ndarray,
    ) -> None:
        self.chain_ = kinpy.build_serial_chain_from_urdf(
            data=robot_description,
            root_link_name=base_link,
            end_link_name=end_effector_link,
        )
        self.dof_ = len(self.chain_.get_joint_parameter_names())
        self.world_config = world_config
        self.Kp = np.diag(Kp)

    def hat(v):
        """
        See https://en.wikipedia.org/wiki/Hat_operator or the MLS book

        Parameters
        ----------
        v : :obj:`numpy.ndarrray`
            vector form of shape 3x1, 3x, 6x1, or 6x

        Returns
        -------
        3x3 or 6x6 :obj:`numpy.ndarray`
            hat version of the vector v
        """
        if v.shape == (3, 1) or v.shape == (3,):
            return np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
        elif v.shape == (6, 1) or v.shape == (6,):
            return np.array([
                    [0, -v[5], v[4], v[0]],
                    [v[5], 0, -v[3], v[1]],
                    [-v[4], v[3], 0, v[2]],
                    [0, 0, 0, 0]
                ])
        else:
            raise ValueError

    def adj(g):
        """
        Adjoint of a rotation matrix.  See the MLS book

        Parameters
        ----------
        g : 4x4 :obj:`numpy.ndarray`
            Rotation matrix

        Returns
        -------
        6x6 :obj:`numpy.ndarray` 
        """
        if g.shape != (4, 4):
            raise ValueError

        R = g[0:3,0:3]
        p = g[0:3,3]
        result = np.zeros((6, 6))
        result[0:3,0:3] = R
        result[0:3,3:6] = hat(p) * R
        result[3:6,3:6] = R
        return result

    def g_matrix_log(self, g: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Given a homogenous transform g, returns a unit twist xi and
        real number theta so that g = exp(xi * theta)
        """
        R = g[:3, :3]
        p = g[:3, 3]
        w, theta = self.rot_matrix_log(R)
        if w.any():
            A = np.matmul(np.eye(3) - R, hat(w)) + np.outer(w, w) * theta
            v = np.linalg.solve(A, p)
        else: 
            w = np.zeros(3)
            theta = np.linalg.norm(p)
            v = p / theta
        xi = np.hstack((v, w))
        return xi, theta

    def rot_matrix_log(self, R: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Given a rotation matrix R, returns the axis angle representation
        of the rotation. Returns (w, theta) so that exp(w * theta) = R.
        """
        tr_R = min(3, max(-1, sum(R[i, i] for i in range(3))))
        theta = np.arccos((tr_R - 1) / 2.0)
        w = (1 / (2 * np.sin(theta))) * np.array([R[2, 1] - R[1, 2],
                                                  R[0, 2] - R[2, 0],
                                                  R[1, 0] - R[0, 1]])
        return w, theta

    def step_control(self, target: np.ndarray, lbr_state: LBRState, sample_period: float) -> np.ndarray:
        """
        Parameters
        ----------
        target: 3x' :obj:`numpy.ndarray` of the desired target position in the world frame
        lbr_state: LBRState of the current joint configuration
        sample_period: float of time between calls to controller

        Returns
        -------
        7x' :obj:`numpy.ndarray` of joint positions for arm to move towards
        """
        theta = np.array(lbr_state.measured_joint_position)
        target_position = (np.linalg.inv(self.world_config) @ np.append(target, 1))[:3]
        current_config = self.chain.forward_kinematics(theta).matrix()
        target_config = kinpy.Transform(np.array([0, 1, 0, 0]), target_position).matrix()
        error_config = np.linalg.inv(current_config) @ target_config
        xi, theta = self.g_matrix_log(error_config)
        error_twist = xi * theta
        jacobian_pinv = np.linalg.pinv(self.chain.jacobian(theta), rcond=0.1)
        theta_dot = jacobian_pinv @ self.Kp @ self.adjoint(current_config) @ error_twist
        return theta + sample_period * theta_dot



