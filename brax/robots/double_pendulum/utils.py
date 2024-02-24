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

        y = -jp.sin(joint1_angle) - jp.sin(joint1_angle + joint2_angle)
        z = jp.cos(joint1_angle) + jp.cos(joint1_angle + joint2_angle)

        return y, z
