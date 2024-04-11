from brax.envs.panda import Panda
import jax

import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np

from jax import numpy as jp

# -----------------------
# --- Initial states ----
# -----------------------

q_initial = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi / 4])
qd_initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


# -----------------------
# --- Robosuite stuff ---
# -----------------------

# Environment configuration
config = load_controller_config(default_controller="OSC_POSE")
env_config = {}
env_config["env_name"] = "Lift"
env_config["robots"] = "Panda"
env_config["camera_names"] = ["frontview"]
env_config["camera_heights"] = 480
env_config["camera_widths"] = 480
env_config["control_freq"] = 10
env_config["controller_configs"] = suite.load_controller_config(default_controller="OSC_POSE")
env_config["has_renderer"] = False
env_config["has_offscreen_renderer"] = False
env_config["ignore_done"] = True
env_config["use_camera_obs"] = False

# Make the environment
env_suite = suite.make(**env_config)

# Reset the environment
env_suite.reset()

# Extract initial robot state (in generalized coordinates)
q_0 = env_suite.sim.data.qpos[env_suite.robots[0].joint_indexes]

# Sample random perturbation (in operational space)
low, high = env_suite.action_spec
action = np.random.uniform(low, high)

# Step Robosuite environment
_, _, _, _ = env_suite.step(action)
q_new = env_suite.sim.data.qpos[env_suite.robots[0].joint_indexes] # not strictly necessary

# Extract OSC controller torques
torques_osc = env_suite.sim.data.ctrl

print(f"Action: {action}")
print(f"Torque: {osc_torques}")
print(f"Initial state: {q_0}")
print(f"New state: {q_new}")

# ------------------
# --- Brax stuff ---
# ------------------

seed = 0
env_brax = Panda()
env_reset_jitted = jax.jit(env_brax.reset)
env_step_jitted = jax.jit(env_brax.step)
brax_init_state = env_reset_jitted(jax.random.PRNGKey(seed))

# Set env_brax state to q_0
brax_init_state = env_brax.set_state(q_0, np.zeros(7))

# Add a correction to OSC torque
torques_total = torques_osc + np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Add torque to brax environment
brax_new_state = env_step_jitted(brax_init_state, torques_total)

# ------------------
# --- Learning stuff ---
# ------------------

# Gradient of state wrt actio

# action = jp.array([0.1])
# reward_gradient_fn = jax.grad(reward)
# grads = reward_gradient_fn(action)
# print(grads)

# run this script with this command to render:
# python -m streamlit run test_fernando.py
# then navigate to http://127.0.0.1:8501/ (or whatever port streamlit tells you)
# from brax.io.rendering import render
# for an animation, look at: brax.evaluate import evaluate
# render(env, [env_reset_jitted(jax.random.PRNGKey(i)).pipeline_state for i in range(100)])