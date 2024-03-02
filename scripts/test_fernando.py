from brax.envs.panda import Panda
import jax

# You can try this to cache jitting functions
# from pathlib import Path
# from jax._src import compilation_cache as cc
# path = (Path(__file__).parent.parent / 'cache')
# cc.initialize_cache(path)

from jax import numpy as jp


seed = 0
env = Panda()
env_reset_jitted = jax.jit(env.reset)
env_step_jitted = jax.jit(env.step)
init_state = env_reset_jitted(jax.random.PRNGKey(seed))


# def reward(action):
#     new_state = env_step_jitted(init_state, action)
#     return new_state.reward


# action = jp.array([0.1])
# reward_gradient_fn = jax.grad(reward)
# grads = reward_gradient_fn(action)
# print(grads)


# run this script with this command to render:
# python -m streamlit run test_fernando.py
# then navigate to http://127.0.0.1:8501/ (or whatever port streamlit tells you)
from brax.io.rendering import render
# for an animation, look at: brax.evaluate import evaluate
render(env, [init_state.pipeline_state])