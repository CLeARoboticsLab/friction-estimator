from brax.robots.go1.utils import Go1Utils
from flax import struct
from jax import numpy as jp


STANDING_FEET_POS = Go1Utils.standing_foot_positions()
FR_STAND = STANDING_FEET_POS[0:3]
FL_STAND = STANDING_FEET_POS[3:6]
RR_STAND = STANDING_FEET_POS[6:9]
RL_STAND = STANDING_FEET_POS[9:12]


@struct.dataclass
class Go1GaitHutterParams:
    f0: jp.ndarray       # initial frequency of gait (Hz)
    phi_i_0: jp.ndarray  # initial phase for each leg, shape (4,)
    h: jp.ndarray        # peak height of foot above ground during swing (m)


class Go1GaitHutter:

    @staticmethod
    def control(gait_params: Go1GaitHutterParams,
                cos_phase: jp.ndarray,
                sin_phase: jp.ndarray,
                f_i: jp.ndarray):

        """Compute desired foot positions in the body frame.

        Arguments:
            gait_params: gait parameters
            cos_phase: cosine of the phase variable, shape ()
            sin_phase: sine of the phase variable, shape ()
            f_i: frequency offset for each leg, shape (4,); FR, FL, RR, RL
        """

        # extract gait parameters
        f0 = gait_params.f0
        phi_i_0 = gait_params.phi_i_0
        h = gait_params.h

        # compute time from phase
        phase = jp.arctan2(sin_phase, cos_phase)
        t = phase / (2 * jp.pi * gait_params.f0)

        # calculate phase for each leg
        phi_i = jp.mod(2*jp.pi*(phi_i_0 + (f0 + f_i)*t), 2*jp.pi)

        # calculate height of foot above ground for each leg
        k = 2*(phi_i - jp.pi)/jp.pi
        z = jp.zeros_like(k)
        z = jp.where(jp.logical_and(0 <= k, k < 1),
                     h*(-2*k**3 + 3*k**2),
                     z)
        z = jp.where(jp.logical_and(1 <= k, k <= 2),
                     h*(2*k**3 - 9*k**2 + 12*k - 4),
                     z)

        # calculate foot positions in body frame
        foot_pos_FR = FR_STAND + jp.array([0, 0, z[0]])
        foot_pos_FL = FL_STAND + jp.array([0, 0, z[1]])
        foot_pos_RR = RR_STAND + jp.array([0, 0, z[2]])
        foot_pos_RL = RL_STAND + jp.array([0, 0, z[3]])

        return jp.concatenate([foot_pos_FR, foot_pos_FL,
                               foot_pos_RR, foot_pos_RL])


if __name__ == "__main__":
    gait_params = Go1GaitHutterParams(
        f0=2.0,
        phi_i_0=jp.array([0.0, 0.5, 0.5, 0.0]),
        h=0.08
    )
    phase = jp.pi/8
    cos_phase = jp.cos(phase)
    sin_phase = jp.sin(phase)
    f_i = jp.array([0.0, 0.0, 0.0, 0.0])
    pdes = Go1GaitHutter.control(gait_params, cos_phase, sin_phase, f_i)
    print(pdes)
