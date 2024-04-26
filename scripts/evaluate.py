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
config = load_controller_config(default_controller="OSC_POSE")
env_config = {}
env_config["env_name"] = "Lift"
env_config["robots"] = "Panda"
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
env_suite = suite.make(**env_config)
env_suite.sim.model.opt.timestep = timestep_length

# Reset the environment
obs = env_suite.reset()

# Set initial robosuite state
env_suite.sim.data.qpos[env_suite.robots[0].joint_indexes] = initial_q
env_suite.sim.data.qvel[env_suite.robots[0].joint_indexes] = initial_qd

print(f"Robosuite environment loaded. Time taken: {time.time() - start_time}")

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
ee_positions_robosuite = []
frames = []
torques = []
ee_positions_robosuite.append(obs['robot0_eef_pos'])
frames.append(obs["sideview_image"])
for i in range(num_steps):

    # Step robosuite and get control torques
    obs, _, _, _ = env_suite.step(action)

    # Save end effecttor position within robosuite
    ee_positions_robosuite.append(obs['robot0_eef_pos'])

    # Record frame 
    frames.append(obs["sideview_image"])

    # Record torques 
    torques.append(env_suite.sim.data.ctrl)

    # Log
    if i % torque_logging_interval == 0:
        print(f"Step {i} of {num_steps}")

ee_positions_robosuite = np.array(ee_positions_robosuite)
torques = np.array(torques)

print(f"Tracking done. Time taken: {time.time() - start_time}")


# ----------------------------
# --- Plot -----------------
# ----------------------------

# Make a subplot of torque over time for each torque
# They should all be in the same figure
total_perturbation = perturbation_amount * num_steps * timestep_length
time_vec = np.arange(num_steps + 1) * timestep_length

# Plot showing displacement of end effector coordinate given by perturbation_index
plt.figure()
plt.plot(time_vec, ee_positions_robosuite[:, perturbation_index], label="Robosuite")
plt.xlabel("Time (s)")
plt.ylabel(f"End effector {['x', 'y', 'z'][perturbation_index]} position")
plt.legend()

plt.savefig("figures/tracking_performance.png")

# ----------------------------
# --- Video ------------------
# ----------------------------
time_vec = np.arange(num_steps + 1) * timestep_length
video_writer = imageio.get_writer("figures/video.mp4", fps=20)
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
