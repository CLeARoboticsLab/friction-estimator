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
friction_static = 0.05

num_steps = 2000
torque_logging_interval = 10
key = jax.random.key(0)
key_states = jax.random.split(key, num=num_steps)
initial_q = np.array(
    [
        0,
        np.pi / 16.0,
        0.00,
        -np.pi / 2.0 - np.pi / 3.0,
        0.00,
        np.pi - 0.2,
        np.pi / 4,
    ]
)
initial_qd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
timestep_length = 0.002

perturbation_index = 1
perturbation_amount = 0.1

# -----------------------
# --- Friction model ----
# -----------------------


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
env_suite_corrected = suite.make(**env_config)

# Reset the environment
env_suite.reset()
env_suite_corrected.reset()

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
env_set_state_jitted = jax.jit(env_brax.set_state)
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

# Sample brax env
init_state = env_set_state_jitted(initial_q, initial_qd)
init_state_corrected = init_state

# Run the simulation
ee_positions_vanilla = [init_state.pipeline_state.x.pos[6][perturbation_index]]
ee_positions_corrected = [
    init_state.pipeline_state.x.pos[6][perturbation_index]
]
for i in range(num_steps):

    # Set robosuite envs
    env_suite.sim.data.qpos[env_suite.robots[0].joint_indexes] = (
        init_state.pipeline_state.q
    )
    env_suite.sim.data.qvel[env_suite.robots[0].joint_indexes] = (
        init_state.pipeline_state.qd
    )
    env_suite_corrected.sim.data.qpos[env_suite.robots[0].joint_indexes] = (
        init_state_corrected.pipeline_state.q
    )
    env_suite_corrected.sim.data.qvel[env_suite.robots[0].joint_indexes] = (
        init_state_corrected.pipeline_state.qd
    )

    # Step robosuite and get control torques
    _, _, _, _ = env_suite.step(action)
    torques_control = env_suite.sim.data.ctrl
    torques_control = torques_control[0:num_joints]

    _, _, _, _ = env_suite_corrected.step(action)
    torques_control_corrected = env_suite_corrected.sim.data.ctrl
    torques_control_corrected = torques_control[0:num_joints]

    # Compute friction torques
    torques_friction = compute_friction_torques(
        init_state.pipeline_state.q, init_state.pipeline_state.qd
    )

    torques_friction_corrected = compute_friction_torques(
        init_state_corrected.pipeline_state.q,
        init_state_corrected.pipeline_state.qd,
    )

    # Evaluate the model
    torques_compensation = network.apply(
        loaded_params, init_state_corrected.obs
    )

    # Step in both brax envs
    init_state = env_step_jitted(
        init_state, torques_control
    )
    init_state_corrected = env_step_jitted(
        init_state_corrected,
        torques_control_corrected
    )

    # Save end effector position
    ee_positions_vanilla.append(
        init_state.pipeline_state.x.pos[6][perturbation_index]
    )
    ee_positions_corrected.append(
        init_state_corrected.pipeline_state.x.pos[6][perturbation_index]
    )

    # Log
    if i % torque_logging_interval == 0:
        print(f"Step {i} of {num_steps}")


print(f"Tracking done. Time taken: {time.time() - start_time}")


# ----------------------------
# --- Plottting --------------
# ----------------------------
total_perturbation = perturbation_amount * num_steps * timestep_length
time_vec = np.arange(num_steps + 1) * timestep_length

plt.figure()
plt.plot(time_vec, ee_positions_vanilla, label="Vanilla")
plt.plot(time_vec, ee_positions_corrected, label="Corrected")
# plt.axhline(y=total_perturbation, color='r', linestyle='--', label='Reference')
plt.legend()
plt.xlabel("Time")
plt.ylabel(f"End effector {['x','y','z'][perturbation_index]}-position [m]")
plt.title("Tracking performance")
plt.savefig("figures/tracking_performance.png")
