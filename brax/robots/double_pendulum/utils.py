from brax.base import System
from etils import epath
from brax.io import mjcf
from jax import numpy as jp


class DoublePendulumUtils:

    @staticmethod
    def get_system() -> System:
        """Returns the system for the double pendulum."""

        # load in urdf file
        path = epath.resource_path('brax')
        path /= 'robots/double_pendulum/double_pendulum.xml'
        sys = mjcf.load(path)

        return sys

    @staticmethod
    def get_approx_system() -> System:
        """Returns the approximate system for the double pendulum."""

        # load in urdf file
        path = epath.resource_path('brax')
        path /= 'robots/double_pendulum/double_pendulum_approx.xml'
        sys = mjcf.load(path)

        return sys

    @staticmethod
    def end_effector_position(q: jp.ndarray) -> jp.ndarray:
        """Returns the y, z position of the end effector in the global frame.
        Assumes length of 1m for both links as defines in the urdf file"""

        joint1_angle = q[0]
        joint2_angle = q[1]

        x = 0.0
        y = -jp.sin(joint1_angle) - jp.sin(joint1_angle + joint2_angle)
        z = jp.cos(joint1_angle) + jp.cos(joint1_angle + joint2_angle)

        return jp.array([x, y, z])

    @staticmethod
    def end_effector_velocity(q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
        """Returns the y, z velocity of the end effector in the global frame.
        Assumes length of 1m for both links as defines in the urdf file"""
        return jp.matmul(DoublePendulumUtils.compute_jacobian(q)[0:3], qd)

    @staticmethod
    def end_effector_state(q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
        """Returns the end effector state in the global frame."""
        return jp.concatenate(
            [
                DoublePendulumUtils.end_effector_position(q),
                DoublePendulumUtils.end_effector_velocity(q, qd),
            ]
        )

    @staticmethod
    def compute_jacobian(q: jp.ndarray) -> jp.ndarray:
        """Computes the Jacobian for the double pendulum.
        Defined by J = dx/dq where x is the end effector position
        and q is the joint angles."""
        # Assumes link length of 1m for both links
        jacobian = jp.array(
            [
                [0.0, 0.0],
                [-jp.cos(q[0]) - jp.cos(q[0] + q[1]), -jp.cos(q[0] + q[1])],
                [-jp.sin(q[0]) - jp.sin(q[0] + q[1]), -jp.sin(q[0] + q[1])],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        return jacobian

    @staticmethod
    def compute_link_mass_matrix():
        """Computes the mass matrix for the double pendulum."""
        # Ideally would be extracted from MuJoCo XML file
        link_mass = 1.0
        link_mass_matrix = jp.array(
            [
                [link_mass, 0, 0, 0, 0, 0],
                [0, link_mass, 0, 0, 0, 0],
                [0, 0, link_mass, 0, 0, 0],
                [0, 0, 0, 0.083958, 0, 0],
                [0, 0, 0, 0, 0.083958, 0],
                [0, 0, 0, 0, 0, 0.00125],
            ]
        )
        return link_mass_matrix

    @staticmethod
    def compute_full_mass_matrix(
        link_mass_matrix: jp.ndarray, J: jp.ndarray
    ) -> jp.ndarray:
        """Computes the full mass matrix for the double pendulum."""
        return 2 * jp.matmul(J.T, jp.matmul(link_mass_matrix, J))

    @staticmethod
    def compute_os_mass_matrix(q: jp.ndarray) -> jp.ndarray:
        """compute the mass matrix in the operational space."""
        J = DoublePendulumUtils.compute_jacobian(q)
        link_mass_matrix = DoublePendulumUtils.compute_link_mass_matrix()
        M = DoublePendulumUtils.compute_full_mass_matrix(link_mass_matrix, J)
        M_inv = jp.linalg.inv(M)
        return jp.linalg.pinv(jp.matmul(J, jp.matmul(M_inv, J.T)))

    @staticmethod
    def compute_grav_torque(q: jp.ndarray) -> jp.ndarray:
        """compute the gravity torque"""
        g = 9.81
        F_grav = jp.array([0.0, 0.0, g])
        J_0 = jp.array(
            [
                [0.0, 0.0],
                [-1 / 2 * jp.cos(q[0]), 0],
                [-1 / 2 * jp.sin(q[0]), 0],
            ]
        )
        J_1 = jp.array(
            [
                [0.0, 0.0],
                [
                    -1 / 2 * jp.cos(q[0] + q[1]) - jp.cos(q[0]),
                    -1 / 2 * jp.cos(q[0] + q[1]),
                ],
                [
                    -1 / 2 * jp.sin(q[0] + q[1]) - jp.sin(q[0]),
                    -1 / 2 * jp.sin(q[0] + q[1]),
                ],
            ]
        )

        return jp.matmul(J_0.T, F_grav) + jp.matmul(J_1.T, F_grav)
