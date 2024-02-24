from brax.robots.go1.utils import Go1Utils
from jax import numpy as jp


class Go1Impedance:

    @staticmethod
    def control(Kp: jp.ndarray, Kd: jp.ndarray, p_des: jp.ndarray,
                q: jp.ndarray, qd: jp.ndarray) -> jp.ndarray:
        """Imedance control for the go1 robot. All arguments will need to be
        formatted in the order FR, FL, RR, RL.

        Arguments:
            Kp: proportional gain; shape (12,); Kpx, Kpy, Kpz for each leg
            Kd: derivative gain; shape (12, ); Kdx, Kdy, Kdz for each leg
            p_des: desired position of the feet, specified in the body frame;
                shape (12,)
            obs: observations
        """

        # desired positions of the feet
        p_des_FR = p_des[0:3]
        p_des_FL = p_des[3:6]
        p_des_RR = p_des[6:9]
        p_des_RL = p_des[9:12]

        # joint angles
        q_FR = q[0:3]
        q_FL = q[3:6]
        q_RR = q[6:9]
        q_RL = q[9:12]

        # joint speeds
        qd_FR = qd[0:3]
        qd_FL = qd[3:6]
        qd_RR = qd[6:9]
        qd_RL = qd[9:12]

        # estimate the current positions of the feet
        p_FR = Go1Utils.forward_kinematics('FR', q_FR)
        p_FL = Go1Utils.forward_kinematics('FL', q_FL)
        p_RR = Go1Utils.forward_kinematics('RR', q_RR)
        p_RL = Go1Utils.forward_kinematics('RL', q_RL)

        # estimate the current velocities of the feet
        pd_FR = Go1Utils.foot_vel('FR', q_FR, qd_FR)
        pd_FL = Go1Utils.foot_vel('FL', q_FL, qd_FL)
        pd_RR = Go1Utils.foot_vel('RR', q_RR, qd_RR)
        pd_RL = Go1Utils.foot_vel('RL', q_RL, qd_RL)

        # jacobians for each leg
        J_FR = Go1Utils.jacobian('FR', q_FR)
        J_FL = Go1Utils.jacobian('FL', q_FL)
        J_RR = Go1Utils.jacobian('RR', q_RR)
        J_RL = Go1Utils.jacobian('RL', q_RL)

        # compute the torques
        u_FR = jp.matmul(jp.transpose(J_FR),
                         Kp[0:3]*(p_des_FR - p_FR) - Kd[0:3]*pd_FR)
        u_FL = jp.matmul(jp.transpose(J_FL),
                         Kp[3:6]*(p_des_FL - p_FL) - Kd[3:6]*pd_FL)
        u_RR = jp.matmul(jp.transpose(J_RR),
                         Kp[6:9]*(p_des_RR - p_RR) - Kd[6:9]*pd_RR)
        u_RL = jp.matmul(jp.transpose(J_RL),
                         Kp[9:12]*(p_des_RL - p_RL) - Kd[9:12]*pd_RL)

        u = jp.concatenate([u_FR, u_FL, u_RR, u_RL])
        return u


if __name__ == '__main__':
    Kp = jp.tile(jp.array([8000, 8000, 12000]), 4)
    Kd = jp.tile(jp.array([15, 15, 30]), 4)
    q = jp.tile(jp.array([0.0, 0.67, -1.3]), 4)
    qd = jp.zeros(12)
    pdes = Go1Utils.standing_foot_positions()
    u = Go1Impedance.control(Kp, Kd, pdes, q, qd)
    print(u)
