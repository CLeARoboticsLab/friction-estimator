from brax.robots.panda.utils import PandaUtils
from brax.envs.base import PipelineEnv, State
from brax import base
from jax import numpy as jp
import jax


class Panda(PipelineEnv):
    """Panda environment."""

    def __init__(
        self,
        theta_des=jp.pi / 4.0,
        action_scale=1.0,
        initial_theta_range=(-jp.pi, jp.pi),
        initial_thetadot_range=(-0.1, 0.1),
        backend="generalized",
        **kwargs
    ):

        # get the brax system for panda
        sys = PandaUtils.get_system()

        # set the time step duration for the physics pipeline
        sys = sys.replace(dt=0.01)

        # the number of times to step the physics pipeline for each
        # environment step
        n_frames = 1

        super().__init__(sys=sys, backend=backend, n_frames=n_frames, **kwargs)

        # parameters for the system
        self._theta_des = theta_des
        self._initial_theta_range = initial_theta_range
        self._initial_thetadot_range = initial_thetadot_range
        self._action_scale = action_scale

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to a random initial state."""
        rng1, rng2 = jax.random.split(rng, 2)

        q = self.sys.init_q + jax.random.uniform(
            rng1,
            (self.sys.q_size(),),
            minval=self._initial_theta_range[0],
            maxval=self._initial_theta_range[1],
        )
        qd = jax.random.uniform(
            rng2,
            (self.sys.qd_size(),),
            minval=self._initial_thetadot_range[0],
            maxval=self._initial_thetadot_range[1],
        )
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)
        metrics = {}

        return State(pipeline_state, obs, reward, done, metrics)

    def set_state(self, q, qd) -> State:
        """Sets the environment state to a specific state."""
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)
        metrics = {}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Steps the environment forward given an action. Action is a torque for every joint."""

        # get initial observation
        prev_obs = self._get_obs(state.pipeline_state)

        # open loop control
        u = action * self._action_scale

        # take an environment step with low level control
        new_pipeline_state = self.pipeline_step(state.pipeline_state, u)

        # get new observations
        obs = self._get_obs(new_pipeline_state)

        # compute reward
        reward = self._compute_reward(prev_obs, obs, action)

        # compute dones for resets; here we never reset
        done = jp.zeros_like(reward)

        return state.replace(
            pipeline_state=new_pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
        )

    def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
        """Observations: q; qd"""
        return jp.concatenate([pipeline_state.q, pipeline_state.qd])

    def _compute_reward(
        self, prev_obs: jp.ndarray, obs: jp.ndarray, action: jp.ndarray
    ) -> jp.ndarray:

        reward = -jp.linalg.norm(prev_obs - obs)
        return reward

    @property
    def action_size(self):
        """Action is open loop torque input"""
        return self.sys.act_size()
