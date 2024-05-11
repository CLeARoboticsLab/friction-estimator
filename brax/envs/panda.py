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

        # Friction parameters
        # Liming Gao, 2017
        self.fv = 288.28
        self.fc = 58.47
        self.qdv = 90.17
        self.Kt = 0.0075  # Ours

        # TODO: Get this directly from mujoco xml
        self.q_max = jp.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        )
        self.q_min = jp.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        )
        self.qd_max = jp.array(
            [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
        )
        self.qd_min = -self.qd_max

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to a random initial state."""
        rng1, rng2 = jax.random.split(rng, 2)

        q = jax.random.uniform(
            rng1,
            (7,),
            minval=self.q_min,
            maxval=self.q_max,
        )
        qd = jax.random.uniform(
            rng2,
            (7,),
            minval=self.qd_min,
            maxval=self.qd_max,
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
        u = action

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

    def step_directly(self, state: State, action: jp.ndarray) -> State:
        """Steps the environment forward given an action. Action is a torque for every joint."""

        # open loop control
        u = action

        # take an environment step with low level control
        new_pipeline_state = self.pipeline_step(state.pipeline_state, u)

        # get new observations
        obs = self._get_obs(new_pipeline_state)

        return state.replace(
            pipeline_state=new_pipeline_state,
            obs=obs,
        )

    def calculate_friction(self, state: State) -> jp.ndarray:
        # Liming Gao, 2017
        qd = state.pipeline_state.qd
        qd = qd * 180 / jp.pi
        return (
            self.Kt
            * jp.sign(qd)
            * (self.fc + self.fv * (1 - jp.exp(-(jp.abs(qd / self.qdv)))))
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
        return self.sys.act_size()#
