import jax
from jax import numpy as jp
import robosuite as suite
from robosuite.controllers import load_controller_config
from brax.envs.panda import Panda
import numpy as np
import time

# -----------------------
# --- Sim Parameters ----
# -----------------------
num_steps = 10000
friction_torque_coeff = 0.1
friction_static = 0.5
num_joints = 7

# -----------------------
# --- Initial states ----
# -----------------------
q_initial = jp.array([0, jp.pi / 16.0, 0.00, -jp.pi / 2.0 - jp.pi / 3.0, 0.00, jp.pi - 0.2, jp.pi / 4])
qd_initial = jp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# -----------------------
# --- Friction model ----
# -----------------------


def compute_friction_torques(q, qd):
    # Friction torque proportional and opposite to joint velocity.
    # If any element of qd is zero, the corresponding friction torque is zero.
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
env_config["controller_configs"] = suite.load_controller_config(default_controller="OSC_POSE")
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
brax_init_state = env_reset_jitted(jax.random.PRNGKey(seed))
env_set_state_jitted = jax.jit(env_brax.set_state)

print("Brax environment loaded.")
print(f"Time taken: {time.time() - start_time}")

# Save initial state, torque_osc, torque_friction, and new state for each step
data_states_init = []
data_torques = []
data_states_new = []

# Set initial state in both environments

# Robosuite
env_suite.sim.data.qpos[env_suite.robots[0].joint_indexes] = q_initial
env_suite.sim.data.qvel[env_suite.robots[0].joint_indexes] = qd_initial

# Brax
brax_init_state = env_set_state_jitted(q_initial, qd_initial)

# Data collection loop
start_time = time.time()
print("Starting data collection loop...")
for step in range(num_steps):

    # --- Act in both environmenst ---

    # Sample random perturbation (from current state in operational space)
    low, high = env_suite.action_spec
    action = np.random.uniform(low, high)

    # Step Robosuite environment
    _, _, _, _ = env_suite.step(action)

    # Extract corresponding OSC controller torques
    torques_osc = env_suite.sim.data.ctrl

    # Compute friction torques given the initial state
    torques_friction = compute_friction_torques(
        brax_init_state.pipeline_state.q, brax_init_state.pipeline_state.qd
    )

    # Step Brax environment with osc torques + friction torques
    torques_total = (
        torques_osc[env_suite.robots[0].joint_indexes] + torques_friction
    )
    brax_new_state = env_step_jitted(brax_init_state, torques_total)

    # Save data
    data_states_init.append(brax_new_state)
    data_torques.append((jp.array(torques_osc[0:num_joints]), torques_friction))
    data_states_new.append(brax_new_state)
    print(f"Step: {step}")

    # Update initial state
    brax_init_state = brax_new_state

print(f"Time taken: {time.time() - start_time}"),

# Save data to file
np.save("data/data_states_init.npy", data_states_init)
np.save("data/data_torques.npy", data_torques)
np.save("data/data_states_new.npy", data_states_new)