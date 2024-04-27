import flax
import jax
import time
import robosuite as suite
import matplotlib.pyplot as plt
import copy

from robosuite.controllers import load_controller_config
from brax.training import networks
from brax.training.types import PRNGKey
from brax.envs.panda import Panda
from brax.envs import State
from jax import numpy as jp
import numpy as np
from flax import serialization
import imageio
import cv2


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

num_steps = 100
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

perturbation_index = 0
perturbation_amount = 0.1

plot_lim = 0.5

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
env_config = {}
env_config["env_name"] = "Lift"
env_config["robots"] = "Panda"
env_config["robots"]
env_config["camera_names"] = ["sideview"]
env_config["camera_heights"] = 480
env_config["camera_widths"] = 480
env_config["control_freq"] = 10
env_config["has_renderer"] = False
env_config["has_offscreen_renderer"] = True
env_config["ignore_done"] = True
env_config["use_camera_obs"] = True

controller_config = suite.load_controller_config(
    default_controller="OSC_POSITION"
)
controller_config['control_delta'] = True
env_config["controller_configs"] = controller_config

# Make the environment
env_rs_vanilla = suite.make(**env_config)
env_rs_vanilla.sim.model.opt.timestep = timestep_length

env_rs_brax = suite.make(**copy.deepcopy(env_config))
env_rs_brax.sim.model.opt.timestep = timestep_length

# Reset the environment
obs_vanilla = env_rs_vanilla.reset()
obs_brax = env_rs_brax.reset()

# Set initial robosuite state
env_rs_vanilla.sim.data.qpos[env_rs_vanilla.robots[0].joint_indexes] = initial_q
env_rs_vanilla.sim.data.qvel[env_rs_vanilla.robots[0].joint_indexes] = initial_qd

env_rs_brax.sim.data.qpos[env_rs_brax.robots[0].joint_indexes] = initial_q
env_rs_brax.sim.data.qvel[env_rs_brax.robots[0].joint_indexes] = initial_qd

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

# Set initial state
state_brax = env_set_state_jitted(initial_q, initial_qd)

# Set integrator


print(f"Brax environment loaded. Time taken: {time.time() - start_time}")

# ----------------------------
# --- Run tracking -----------
# ----------------------------
print("Testing tracking...")
start_time = time.time()

# Action
action = np.zeros(3 + 1)
cartesian_perturbation = np.zeros(3)
cartesian_perturbation[perturbation_index] = perturbation_amount
action[0:3] = cartesian_perturbation

# Run the simulation
ee_pos_vanilla = []
ee_pos_brax = []
frames_vanilla = []
frames_brax = []

for i in range(num_steps):

    # Save end effecttor position within robosuite
    ee_pos_vanilla.append(obs_vanilla['robot0_eef_pos'])
    ee_pos_brax.append(obs_brax['robot0_eef_pos'])

    # Record frame
    frames_vanilla.append(obs_vanilla["sideview_image"])
    frames_brax.append(obs_brax["sideview_image"])

    # Step robosuite and get control torques
    obs_vanilla, _, _, _ = env_rs_vanilla.step(action)
    obs_brax, _, _, _ = env_rs_brax.step(action)

    # Record torques
    torques_control_vanilla = env_rs_vanilla.sim.data.ctrl
    torques_control_brax = env_rs_brax.sim.data.ctrl

    # Step brax
    state_brax = env_step_jitted(
        state_brax, torques_control_brax[0:num_joints]
    )

    # Set brax robosuite env to the same state as stepped brax
    # env_rs_brax.sim.data.qpos[env_rs_brax.robots[0].joint_indexes] = (
    #     state_brax.pipeline_state.q
    # )
    # env_rs_brax.sim.data.qvel[env_rs_brax.robots[0].joint_indexes] = (
    #     state_brax.pipeline_state.qd
    # )

    # Log
    if i % torque_logging_interval == 0:
        print(f"Step {i} of {num_steps}")

ee_pos_vanilla = np.array(ee_pos_vanilla)
ee_pos_brax = np.array(ee_pos_brax)

print(f"Tracking done. Time taken: {time.time() - start_time}")


# ----------------------------
# --- Plot -----------------
# ----------------------------

# Make a subplot of torque over time for each torque
# They should all be in the same figure
total_perturbation = perturbation_amount * num_steps * timestep_length
time_vec = np.arange(num_steps) * timestep_length

# Plot showing displacement of end effector coordinate given by perturbation_index
plt.figure()
plt.plot(time_vec, ee_pos_vanilla[:, perturbation_index], label="Robosuite")
plt.plot(time_vec, ee_pos_brax[:, perturbation_index], label="Brax")
plt.xlabel("Time (s)")
plt.ylabel(f"End effector {['x', 'y', 'z'][perturbation_index]} position")
plt.legend()

plt.savefig("figures/tracking_performance.png")

# ----------------------------
# --- Video ------------------
# ----------------------------


def save_video(frames, title="video"):
    time_vec = np.arange(num_steps + 1) * timestep_length
    video_writer = imageio.get_writer("figures/" + title + ".mp4", fps=20)
    for idx, frame in enumerate(frames):
        frame = frame.astype(np.uint8)
        frame = cv2.flip(frame, 0)
        cv2.putText(
            frame,
            f"t = {time_vec[idx]}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        video_writer.append_data(frame)

    video_writer.close()


save_video(frames_vanilla, "vanilla")
save_video(frames_brax, "brax")
