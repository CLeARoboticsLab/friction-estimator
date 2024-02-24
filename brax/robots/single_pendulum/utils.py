from brax.base import System
from etils import epath
from brax.io import mjcf


class SinglePendulumUtils:

    @staticmethod
    def get_system() -> System:
        """Returns the system for the single pendulum."""

        # load in urdf file
        path = epath.resource_path('brax')
        path /= 'robots/single_pendulum/single_pendulum.xml'
        sys = mjcf.load(path)

        return sys
