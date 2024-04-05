import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np

# Load the desired controller's default config as a dict
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
env = suite.make(**env_config)

# Reset the environment
env.reset()

# Get action limits
low, high = env.action_spec

# do visualization
for i in range(10000):
    action = np.random.uniform(low, high)
    obs, reward, done, _ = env.step(action)

    # Print action and corresponding torque from the environment. Format them so it's clear
    print(f"Action: {action}")
    print(f"Torque: {env.sim.data.ctrl}")