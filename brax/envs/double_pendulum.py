from brax.robots.double_pendulum.utils import DoublePendulumUtils
from brax.envs.base import State, PipelineEnv
from brax import base
from jax import numpy as jp
from typing import Optional, Any
import jax


class DoublePendulum(PipelineEnv):
    """Double Pendulum environment."""

    def __init__(self, backend='generalized',
                 normalize_reward=True,
                 **kwargs):

        # get the brax system for the double pendulum
        sys = DoublePendulumUtils.get_system()
        self._sys_approx = DoublePendulumUtils.get_approx_system()

        # set the time step duration for the physics pipeline
        sys = sys.replace(dt=0.01)
        self._sys_approx = self._sys_approx.replace(dt=0.01)

        # the number of times to step the physics pipeline for each
        # environment step
        n_frames = 1

        super().__init__(sys=sys, backend=backend, n_frames=n_frames, **kwargs)

        """parameters for the system"""
        # element-wise multiplication of the actions to scale them from [-1, 1]
        # to [min, max] corrections to setpoints
        self._action_weights = jp.array([1.0, 1.0, 1.0, 1.0])
        self._normalize_reward = normalize_reward

        # uncorrected setpoint for q1, q2, q1d, q2d (pendulum is straight up)
        self._setpoint = jp.array([0.0, 0.0, 0.0, 0.0])

        # feedback gains for the low-level controller
        self._K = jp.array([[64., 17., 25., 9.],
                            [17., 30., 9., 8.]])

        # desired position of the end effector
        self._ydes = 1.5
        self._zdes = 1.5

        # Friction parameters
        self.friction_torque_coeff = 1.0
        self.friction_static = 0.1

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-jp.pi, maxval=jp.pi
        )
        qd = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=-0.1, maxval=0.1
        )
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)
        metrics = {}
        u = jp.zeros(self.sys.act_size())

        return State(pipeline_state, obs, reward, done, metrics, u=u)
    
    def set_state(self, q: jp.ndarray, qd: jp.ndarray) -> State:
        """Sets the state of the environment."""
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)
        metrics = {}
        u = jp.zeros(self.sys.act_size())

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        # translate state to observations
        prev_obs = self._get_obs(state.pipeline_state)

        # compute low-level control
        u = self.low_level_control(action, prev_obs)

        # take an environment step with low level control
        pipeline_state = self.pipeline_step(state.pipeline_state, u)

        # get new observations
        obs = self._get_obs(pipeline_state)

        # compute reward
        reward, _ = self.compute_reward(obs, prev_obs, u, action)

        # compute dones for resets; here we never reset
        done = jp.zeros_like(reward)
        # jp.where(jp.abs(obs[1]) > 1000.0, 1.0, 0.0)

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def step_with_friction(self, state: State, action: jp.ndarray) -> State:
        # translate state to observations
        prev_obs = self._get_obs(state.pipeline_state)

        # compute low-level control
        u = self.low_level_control(action, prev_obs)

        # Add friction
        friction = self.calculate_friction(state)

        # take an environment step with low level control
        pipeline_state = self.pipeline_step(state.pipeline_state, u + friction)

        # get new observations
        obs = self._get_obs(pipeline_state)

        # compute reward
        reward, _ = self.compute_reward(obs, prev_obs, u, action)

        # compute dones for resets; here we never reset
        done = jp.zeros_like(reward)
        # jp.where(jp.abs(obs[1]) > 1000.0, 1.0, 0.0)

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def calculate_friction(self, state: State) -> jp.ndarray:
        qd = state.pipeline_state.qd
        return jp.where(
            qd != 0, -self.friction_torque_coeff * qd, self.friction_static
        )

    def approx_dynamics(self, obs: jp.ndarray, u: jp.ndarray,
                        ext_forces: Optional[jp.ndarray] = None,
                        obs_next: Optional[jp.ndarray] = None) -> jp.ndarray:
        q = obs[:self._sys_approx.q_size()]
        qd = obs[self._sys_approx.q_size():]
        pipeline_state_start = self.pipeline_init_approx(q, qd)
        pipeline_state = self.pipeline_step_approx(pipeline_state_start, u)
        obs_new = self._get_obs(pipeline_state)

        return obs_new

    def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
        """Observations: q; qd"""
        return jp.concatenate([pipeline_state.q, pipeline_state.qd])

    def low_level_control(self, action: jp.ndarray,
                           obs: jp.ndarray) -> jp.ndarray:

        # actions are corrections to the setpoints
        corrected_setpoints = (self._setpoint
                               + action*self._action_weights)

        # feedback control
        u = jp.matmul(-self._K, (obs - corrected_setpoints))

        return u

    def compute_reward(self, obs: jp.ndarray, prev_obs: jp.ndarray,
                       u: jp.ndarray,
                       unscaled_action: jp.ndarray) -> jp.ndarray:

        # get joint angles
        q = obs[:self.sys.q_size()]

        # compute end effector position
        y, z = DoublePendulumUtils.end_effector_position(q)

        # higher reward the closer the end effector is to the desired position
        ysq = (y - self._ydes)**2
        zsq = (z - self._zdes)**2
        reward = jax.lax.select(
            self._normalize_reward,
            0.5*jp.exp(-ysq**2/4) + 0.5*jp.exp(-zsq**2/4),
            -((y - self._ydes)**2 + (z - self._zdes)**2)
        )

        return reward, {}

    def pipeline_init_approx(self, q: jp.ndarray, qd: jp.ndarray) -> base.State:
        """Initializes the pipeline state for the approximate system."""
        return self._pipeline.init(self._sys_approx, q, qd, self._debug)

    def pipeline_step_approx(
        self, pipeline_state: Any, action: jp.ndarray
    ) -> base.State:
        """Takes a physics step using the physics pipeline on the approximate
        system."""

        def f(state, _):
            return (
                self._pipeline.step(self._sys_approx, state,
                                    action, self._debug),
                None,
            )

        return jax.lax.scan(f, pipeline_state, (), self._n_frames)[0]

    @property
    def action_size(self):
        """Actions are corrections to the setpoints: q1, q2, q1d, q2d, so the
        action size is 4"""
        return 4

    @property
    def action_weights(self) -> jp.ndarray:
        return self._action_weights
