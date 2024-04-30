import jax
from jax import numpy as jp
from brax.envs.double_pendulum import DoublePendulum
import flax
from brax.envs import State
import time
import pickle
from jax.config import config


# -----------------------
# --- Debug Utils -------
# -----------------------
# jax.config.update("jax_disable_jit", True)
# config.update("jax_enable_x64", True)


# -----------------------
# --- Sim Parameters ----
# -----------------------
num_steps = 2 ** 15
torque_logging_interval = 100
num_joints = 2
link_length = 1.0
seed = 0

# -----------------------
# --- Brax stuff --------
# -----------------------

print("Loading Brax environment...")
start_time = time.time()

# Setup Brax environment
env = DoublePendulum()  # 2D double pendulum rotating around x-axis
step_jitted = jax.jit(env.step_with_friction)
reset_jitted = jax.jit(env.reset)

print("Brax environment loaded.")
print(f"Time taken: {time.time() - start_time}")


# ----------------------------
# --- Collect data  ----------
# ----------------------------

# Data class
@flax.struct.dataclass
class MyData:
    init_state: State
    torque: jp.ndarray
    friction: jp.ndarray
    next_state: State


def sample_os_action(key):
    # Return a point on or within a circle with radius num_joints * link_length
    # Sampling based on this discussion: https://stackoverflow.com/a/50746409
    # 2D double pendulum rotating around x-axis
    k1, k2 = jax.random.split(key)
    r = num_joints * link_length * jp.sqrt(jax.random.uniform(k1))
    theta = jax.random.uniform(k2) * 2 * jp.pi
    return jp.array([0.0, r * jp.cos(theta), r * jp.sin(theta)])


# Data collection function
def make_data(key):
    # Sample init brax state and compute friction torques
    init_state = reset_jitted(key)

    # Sample action
    action = sample_os_action(key)

    # Compute torques TODO: save these in the brax state
    torque = env.osc_control(action, env._get_obs(init_state.pipeline_state))
    friction = env.calculate_friction(init_state, torque)

    # Step in both environments
    next_state = step_jitted(init_state, action)

    return MyData(init_state, torque, friction, next_state)


# Collect data
print("Starting data collection loop...")
start_time = time.time()

key = jax.random.key(seed)
key_states = jax.random.split(key, num=num_steps)
data = jax.vmap(make_data)(key_states)

print("Data collection loop finished.")
print(f"Time taken: {time.time() - start_time}"),

# Save data
with open("data/data.pkl", "wb") as f:
    pickle.dump(data, f)
