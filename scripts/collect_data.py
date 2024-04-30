import jax
from jax import numpy as jp
import robosuite as suite
from robosuite.controllers import load_controller_config
from brax.envs.panda import Panda
import flax
from brax.envs import State
import numpy as np
import time
import pickle
from jax.config import config


# -----------------------
# --- Debug Utils -------
# -----------------------
# jax.config.update("jax_disable_jit", True)
config.update("jax_enable_x64", True)


# -----------------------
# --- Sim Parameters ----
# -----------------------
num_steps = 2 ** 15
friction_torque_coeff = 100.0
friction_static = 10.0
torque_logging_interval = 100
num_joints = 7
key = jax.random.key(0)
key_states = jax.random.split(key, num=num_steps)

# -----------------------
# --- Friction model ----
# -----------------------


# Friction torque proportional and opposite to joint velocity.
# If any element of qd is zero, the corresponding friction torque is zero.
def compute_friction_torques(q, qd):
    return jp.where(qd != 0, -friction_torque_coeff * qd, friction_static)


# -----------------------
# --- Robosuite stuff ---
# -----------------------

print("Loading Robosuite environment...")
start_time = time.time()

# Environment configuration
config = load_controller_config(default_controller="OSC_POSE")
env_config = {}
env_config["env_name"] = "Lift"
env_config["robots"] = "Panda"
env_config["camera_names"] = ["frontview"]
env_config["camera_heights"] = 480
env_config["camera_widths"] = 480
env_config["control_freq"] = 10
env_config["controller_configs"] = suite.load_controller_config(
    default_controller="OSC_POSE"
)
env_config["has_renderer"] = False
env_config["has_offscreen_renderer"] = False
env_config["ignore_done"] = True
env_config["use_camera_obs"] = False

# Make the environment
env_suite = suite.make(**env_config)

# Reset the environment
env_suite.reset()

print("Robosuite environment loaded.")
print(f"Time taken: {time.time() - start_time}")

# -----------------------
# --- Brax stuff --------
# -----------------------

print("Loading Brax environment...")
start_time = time.time()

# Setup Brax environment
seed = 0
env_brax = Panda()
env_reset_jitted = jax.jit(env_brax.reset)
env_step_jitted = jax.jit(env_brax.step)
low, high = env_suite.action_spec

print("Brax environment loaded.")
print(f"Time taken: {time.time() - start_time}")


# Data class
@flax.struct.dataclass
class MyData:
    init_state: State
    torque: jp.ndarray
    friction: jp.ndarray
    next_state: State


# ----------------------------
# --- Generate OSC torques ---
# ----------------------------
print("Generating OSC torques...")
start_time = time.time()

torques = []
for i in range(num_steps):
    # Sample brax env
    init_state = env_reset_jitted(key_states[i])

    # Set robosuite env
    env_suite.sim.data.qpos[env_suite.robots[0].joint_indexes] = (
        init_state.pipeline_state.q
    )
    env_suite.sim.data.qvel[env_suite.robots[0].joint_indexes] = (
        init_state.pipeline_state.qd
    )

    # Sample action and compute osc torques
    action = np.random.uniform(low, high)  # use jax for this?
    _, _, _, _ = env_suite.step(action)
    torques_osc = env_suite.sim.data.ctrl

    # Save torques
    torques.append(torques_osc[0:num_joints])

    if i % torque_logging_interval == 0:
        print(f"Step {i} of {num_steps}")

torques = jp.stack(torques)

print("OSC torques generated.")
print(f"Time taken: {time.time() - start_time}")

# ----------------------------
# --- Collect data  ----------
# ----------------------------


# Data collection function
def make_data(key, torque):
    # Sample init brax state and compute friction torques
    init_state = env_reset_jitted(key)
    friction = compute_friction_torques(
        init_state.pipeline_state.q, init_state.pipeline_state.qd
    )

    # Step in both environments
    next_state = env_step_jitted(init_state, torque + friction)

    return MyData(init_state, torque, friction, next_state)


# Collect data
print("Starting data collection loop...")
start_time = time.time()
data = jax.vmap(make_data)(key_states, torques)
print("Data collection loop finished.")
print(f"Time taken: {time.time() - start_time}"),


# Save data
with open("data/data.pkl", "wb") as f:
    pickle.dump(data, f)