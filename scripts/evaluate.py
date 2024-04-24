import flax
import jax
import time
import robosuite as suite
import matplotlib.pyplot as plt

from robosuite.controllers import load_controller_config
from brax.training import networks
from brax.training.types import PRNGKey
from brax.envs.panda import Panda
from brax.envs import State
from jax import numpy as jp
import numpy as np
from flax import serialization


# -----------------------
# --- Sim Parameters ----
# -----------------------

# Parameters
num_joints = 7
input_size = num_joints * 2
hidden_layer_dim = 256
hidden_layer_num = 3
output_size = num_joints
seed = 0

friction_torque_coeff = 10.
friction_static = 1.0

num_steps = 100
torque_logging_interval = 10
key = jax.random.key(0)
key_states = jax.random.split(key, num=num_steps)

perturbation_index = 0
perturbation_amount = 0.001

# -----------------------
# --- Friction model ----
# ------------
# -----------


def compute_friction_torques(q, qd):
    return jp.where(qd != 0, -friction_torque_coeff * qd, friction_static)


# -----------------------
# ----- Load model ------
# -----------------------
print("Loading model...")
start_time = time.time()

# with open('data/model_params.bin', 'rb') as f:
#     bytes_input = f.read()

network = networks.MLP(
    layer_sizes=([hidden_layer_dim] * hidden_layer_num + [output_size])
)
# dummy_params = network.init(jax.random.PRNGKey(seed), jp.zeros((input_size)))
# loaded_params = serialization.from_bytes(dummy_params, bytes_input)

loaded_params = network.init(jax.random.PRNGKey(seed), jp.zeros((input_size)))

print(f"Model loaded. Time taken: {time.time() - start_time}")

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

print(f"Robosuite environment loaded. Time taken: {time.time() - start_time}")

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

print(f"Brax environment loaded. Time taken: {time.time() - start_time}")

# ----------------------------
# --- Run tracking -----------
# ----------------------------
print("Testing tracking...")
start_time = time.time()

# Action
action = np.zeros(num_joints)
cartesian_perturbation = np.zeros(3)
cartesian_perturbation[perturbation_index] = perturbation_amount
action[0:3] = cartesian_perturbation
action = np.zeros(num_joints)

# Run the simulation
ee_positions_vanilla = []
ee_positions_corrected = []
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
    _, _, _, _ = env_suite.step(action)
    torques_control = env_suite.sim.data.ctrl
    torques_control = torques_control[0:num_joints]

    # Compute friction torques
    torques_friction = compute_friction_torques(
        init_state.pipeline_state.q, init_state.pipeline_state.qd
    )

    # Evaluate the model
    torques_compensation = network.apply(loaded_params, init_state.obs)

    # Step in both brax envs
    next_state = env_step_jitted(
        init_state, torques_control + torques_friction
    )
    next_state_corrected = env_step_jitted(
        init_state, torques_control + torques_friction + torques_compensation
    )

    # Save end effector position
    ee_positions_vanilla.append(
        next_state.pipeline_state.x.pos[6][perturbation_index]
    )
    ee_positions_corrected.append(
        next_state_corrected.pipeline_state.x.pos[6][perturbation_index]
    )

    if i % torque_logging_interval == 0:
        print(f"Step {i} of {num_steps}")


print(f"Tracking done. Time taken: {time.time() - start_time}")


# ----------------------------
# --- Plottting --------------
# ----------------------------
plt.figure()
plt.plot(ee_positions_vanilla, label="Vanilla")
plt.plot(ee_positions_corrected, label="Corrected")
plt.legend()
plt.xlabel("Step")
plt.ylabel(f"End effector {['x','y','z'][perturbation_index]}-position")
plt.title("Tracking performance")
plt.savefig("figures/tracking_performance.png")
