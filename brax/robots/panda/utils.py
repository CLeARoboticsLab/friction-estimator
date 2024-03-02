from brax.base import System
from etils import epath
from brax.io import mjcf


class PandaUtils:

    @staticmethod
    def get_system() -> System:
        """Returns the system for the Panda robotic manipulator."""

        # load in urdf file
        path = epath.resource_path('brax')
        path /= 'robots/panda/panda.xml'
        sys = mjcf.load(path)

        return sys
