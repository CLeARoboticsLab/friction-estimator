import jax
from jax import numpy as jp
import robosuite as suite
from robosuite.controllers import load_controller_config
from brax.envs.panda import Panda
import numpy as np

# -----------------------
# --- Sim Parameters ----
# -----------------------
num_steps = 10
friction_torque_coeff = 0.1
friction_static = 0.5

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

# -----------------------
# --- Brax stuff --------
# -----------------------

# Setup Brax environment
seed = 0
env_brax = Panda()
env_reset_jitted = jax.jit(env_brax.reset)
env_step_jitted = jax.jit(env_brax.step)
brax_init_state = env_reset_jitted(jax.random.PRNGKey(seed))

# Save initial state, torque_osc, torque_friction, and new state for each step
data = []

for step in range(num_steps):

    # --- Set initial state in both environments ---

    # Robosuite
    env_suite.sim.data.qpos[env_suite.robots[0].joint_indexes] = q_initial
    env_suite.sim.data.qvel[env_suite.robots[0].joint_indexes] = qd_initial

    # Brax
    brax_init_state = env_brax.set_state(q_initial, qd_initial)

    # Assert that both environments have the same initial state
    assert jp.allclose(q_initial, env_suite.sim.data.qpos[env_suite.robots[0].joint_indexes])
    assert jp.allclose(q_initial, brax_init_state.pipeline_state.q)
    assert jp.allclose(qd_initial, env_suite.sim.data.qvel[env_suite.robots[0].joint_indexes])
    assert jp.allclose(qd_initial, brax_init_state.pipeline_state.qd)

    # --- Act in both environmenst ---

    # Sample random perturbation (from current state in operational space)
    low, high = env_suite.action_spec
    action = np.random.uniform(low, high)

    # Step Robosuite environment
    _, _, _, _ = env_suite.step(action)

    # Extract corresponding OSC controller torques
    torques_osc = env_suite.sim.data.ctrl

    # Compute friction torques given the initial state
    torques_friction = compute_friction_torques(q_initial, qd_initial)

    # Step Brax environment with osc torques + friction torques
    torques_total = torques_osc[env_suite.robots[0].joint_indexes] + torques_friction
    brax_new_state = env_step_jitted(brax_init_state, torques_total)

    # Save data
    data.append((step,
                 q_initial,
                 qd_initial,
                 torques_osc,
                 torques_friction,
                 brax_new_state.pipeline_state.q,
                 brax_new_state.pipeline_state.qd))

    print(f"Step: {step}")
    print(f"Initial q: {q_initial}")
    print(f"Initial qd: {qd_initial}")
    print(f"Action: {action}")
    print(f"Torque_o: {torques_osc}")    
    print(f"Torque_f: {torques_friction}")
    print(f"New q_brax: {brax_new_state.pipeline_state.q}")
    print(f"New qd_brax: {brax_new_state.pipeline_state.qd}")

# Save data to file 
np.save("data/data.npy", data)
