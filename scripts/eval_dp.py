import jax
import flax
import time
import pickle
from brax.training import networks
from brax.envs.double_pendulum import DoublePendulum
from brax.robots.double_pendulum.utils import DoublePendulumUtils
from jax import numpy as jp
import matplotlib.pyplot as plt
from brax.envs import State
from flax import serialization

# jax.config.update("jax_disable_jit", True)

# -----------------------
# ----- Parameters ------
# -----------------------

# Sim parameters
q_init = jp.array([-jp.pi, 0.0])  # straight down
qd_init = jp.array([0.0, 0.0])
steps = 1000
action = jp.array([0.0, 1.0, 1.0])  # position control only
plot_start = 0
num_joints = 2

# Neural network parameters
hidden_layer_dim = 256
hidden_layer_num = 3
seed = 0
input_size = 2 * num_joints
output_size = num_joints
add_corrected_trajectory = True


# -----------------------
# ----- Load model ------
# -----------------------

# Load model and params
network = networks.MLP(
        layer_sizes=([hidden_layer_dim] * hidden_layer_num + [output_size])
    )
dummy_params = network.init(jax.random.PRNGKey(seed), jp.zeros((input_size)))
loaded_params = dummy_params


@flax.struct.dataclass
class NormalizationParameters:
    translation: jp.ndarray
    scaling: jp.ndarray


norm_params = NormalizationParameters(
    translation=jp.zeros(input_size), scaling=jp.ones(input_size)
)
if add_corrected_trajectory:
    print("Loading model...")
    start_time = time.time()

    with open('data/model_params.bin', 'rb') as f:
        bytes_input = f.read()

    # with open('brax/scripts/data/model_params.bin', 'rb') as f:
    #     bytes_input = f.read()

    loaded_params = serialization.from_bytes(dummy_params, bytes_input)

    print(f"Model loaded. Time taken: {time.time() - start_time}")

    print("Loading parameters...")
    start_time = time.time()

    with open("data/norm_params.pkl", "rb") as f:
        norm_params = pickle.load(f)

    # with open("brax/scripts/data/norm_params.pkl", "rb") as f:
    #     norm_params = pickle.load(f)

    print(f"Parameters loaded. Time taken: {time.time() - start_time}")


# Run model
def normalize_joint_state(joint_state):
    return jax.tree_util.tree_map(
        lambda state: (state - norm_params.translation) / norm_params.scaling,
        joint_state,
    )


def compute_friction_torques(params, obs):
    obs = normalize_joint_state(obs)
    return network.apply(params, obs)


# -----------------------
# ----- Brax env --------
# -----------------------

# Load env and functions
env_nf = DoublePendulum()  # No friction
env_yf = DoublePendulum()  # With friction
env_cf = DoublePendulum()  # Corrected

step_nf_jitted = jax.jit(env_nf.step)
step_yf_jitted = jax.jit(env_yf.step_with_friction)
step_cf_jitted = jax.jit(env_nf.step_directly)
set_nf_jitted = jax.jit(env_nf.set_state)
set_yf_jitted = jax.jit(env_yf.set_state)
set_cf_jitted = jax.jit(env_cf.set_state)


# Define and init step functions
@flax.struct.dataclass
class MyData:
    init_state: State
    new_state: State
    action: jp.ndarray
    x: jp.ndarray
    xd: jp.ndarray
    x_error: jp.float32
    xd_error: jp.float32
    control_torque: jp.ndarray = jp.zeros(num_joints)


def make_trajectory_step(action, step_fn):
    def trajectory_step(carry, in_element):
        init_state = carry
        x = DoublePendulumUtils.end_effector_position(
            init_state.pipeline_state.q
        )
        xd = DoublePendulumUtils.end_effector_velocity(
            init_state.pipeline_state.q, init_state.pipeline_state.qd
        )
        x_error = jp.linalg.norm(x - action)
        xd_error = jp.linalg.norm(xd)
        new_state = step_fn(init_state, action)
        control_torque = env_cf.osc_control(
            action, init_state.obs
        )  # only for plotting purposes. Not used in simulation.
        return new_state, MyData(
            init_state,
            new_state,
            action,
            x,
            xd,
            x_error,
            xd_error,
            control_torque,
        )

    return trajectory_step


def make_cf_trajectory_step(action, step_fn, params):
    def trajectory_step(carry, in_element):
        init_state = carry
        x = DoublePendulumUtils.end_effector_position(
            init_state.pipeline_state.q
        )
        xd = DoublePendulumUtils.end_effector_velocity(
            init_state.pipeline_state.q, init_state.pipeline_state.qd
        )
        x_error = jp.linalg.norm(x - action)
        xd_error = jp.linalg.norm(xd)
        control_torque = env_cf.osc_control(action, init_state.obs)
        friction = env_cf.calculate_friction(init_state)
        friction_correction = compute_friction_torques(
            params, init_state.obs
        )  # i/o both (batch_size, num_joints)
        new_state = step_fn(
            init_state, control_torque + friction + friction_correction
        )
        return new_state, MyData(
            init_state,
            new_state,
            action,
            x,
            xd,
            x_error,
            xd_error,
            control_torque,
        )
    return trajectory_step


trajectory_step_nf = make_trajectory_step(action, step_nf_jitted)
trajectory_step_yf = make_trajectory_step(action, step_yf_jitted)
trajectory_step_cf = make_cf_trajectory_step(
    action, step_cf_jitted, loaded_params
)


# -----------------------
# ----- Simulation ------
# -----------------------

# Reset the environment
state_nf = set_nf_jitted(q_init, qd_init)
state_yf = set_yf_jitted(q_init, qd_init)
state_cf = set_cf_jitted(q_init, qd_init)

# Run the simulation
print("Generating trajectory")
start_time = time.time()

_, trajectory_nf = jax.lax.scan(
    trajectory_step_nf, state_nf, (), steps
)
_, trajectory_yf = jax.lax.scan(
    trajectory_step_yf, state_yf, (), steps
)
_, trajectory_cf = jax.lax.scan(
    trajectory_step_cf, state_cf, (), steps
)

print("Trajectory generated.")
print(f"Time taken: {time.time() - start_time}")


# -----------------------
# -------- Plot ---------
# -----------------------

# Plot everything
x_nf = trajectory_nf.x
x_yf = trajectory_yf.x
xd_nf = trajectory_nf.xd
xd_yf = trajectory_yf.xd
x_error_nf = trajectory_nf.x_error
x_error_yf = trajectory_yf.x_error
xd_error_nf = trajectory_nf.xd_error
xd_error_yf = trajectory_yf.xd_error
q_nf = trajectory_nf.init_state.pipeline_state.q
q_yf = trajectory_yf.init_state.pipeline_state.q
qd_nf = trajectory_nf.init_state.pipeline_state.qd
qd_yf = trajectory_yf.init_state.pipeline_state.qd

fig, axs = plt.subplots(5, 2, figsize=(7.5, 7.5))
fig.suptitle("Double Pendulum State")
margin = 0.1

# Joint state
axs[0, 0].plot(q_nf[:, 0], color="tab:blue", linestyle="dashed")
axs[0, 0].plot(q_nf[:, 1], color="tab:blue", linestyle="dashdot")
axs[0, 0].plot(q_yf[:, 0], color="tab:orange", linestyle="dashed")
axs[0, 0].plot(q_yf[:, 1], color="tab:orange", linestyle="dashdot")
axs[0, 0].set_ylabel("q [rad]")
# axs[0, 0].set_ylim([-2*jp.pi - margin, 2*jp.pi + margin])

axs[0, 1].plot(qd_nf[:, 0], color="tab:blue", linestyle="dashed")
axs[0, 1].plot(qd_nf[:, 1], color="tab:blue", linestyle="dashdot")
axs[0, 1].plot(qd_yf[:, 0], color="tab:orange", linestyle="dashed")
axs[0, 1].plot(qd_yf[:, 1], color="tab:orange", linestyle="dashdot")
axs[0, 1].set_ylabel("qd [rad/s]")

# End effector state
axs[1, 0].plot(x_nf[:, 1], label="NF")
axs[1, 0].plot(x_yf[:, 1], label="YF")
axs[1, 0].hlines(
    y=action[1],
    xmin=0,
    xmax=len(x_nf),
    colors="r",
    linestyles="dotted",
    label="Reference",
)
axs[1, 0].set_ylabel("y [m]")
axs[1, 0].set_ylim([-2 - margin, 2 + margin])

axs[2, 0].plot(x_nf[:, 2], label="NF")
axs[2, 0].plot(x_yf[:, 2], label="YF")
axs[2, 0].hlines(
    y=action[2],
    xmin=0,
    xmax=len(x_nf),
    colors="r",
    linestyles="dotted",
    label="Reference",
)
axs[2, 0].set_ylabel("z [m]")
axs[2, 0].set_xlabel("Steps")
axs[2, 0].set_ylim([-2 - margin, 2 + margin])

axs[1, 1].plot(xd_nf[:, 1], label="NF")
axs[1, 1].plot(xd_yf[:, 1], label="YF")
axs[1, 1].hlines(
    y=0.0,
    xmin=0,
    xmax=len(x_nf),
    colors="r",
    linestyles="dotted",
    label="Reference",
)
axs[1, 1].set_ylabel("yd [m/s]")

axs[2, 1].plot(xd_nf[:, 2], label="NF")
axs[2, 1].plot(xd_yf[:, 2], label="YF")
axs[2, 1].hlines(
    y=0.0,
    xmin=0,
    xmax=len(x_nf),
    colors="r",
    linestyles="dotted",
    label="Reference",
)
axs[2, 1].set_ylabel("zd [m/s]")

# Error
axs[3, 0].plot(x_error_nf, label="NF")
axs[3, 0].plot(x_error_yf, label="YF")
axs[3, 0].hlines(
    y=0.0,
    xmin=0,
    xmax=len(x_nf),
    colors="r",
    linestyles="dotted",
    label="Reference",
)
axs[3, 0].set_ylabel("Pos. error [m]")
axs[3, 0].set_xlabel("Steps")
axs[3, 0].set_ylim(bottom=0 - margin)

axs[3, 1].plot(xd_error_nf, label="NF")
axs[3, 1].plot(xd_error_yf, label="YF")
axs[3, 1].hlines(
    y=0.0,
    xmin=0,
    xmax=len(x_nf),
    colors="r",
    linestyles="dotted",
    label="Reference",
)
axs[3, 1].set_ylabel("Vel. error [m/s]")
axs[3, 1].set_xlabel("Steps")
axs[3, 1].set_ylim(bottom=0 - margin)

# Add corrected trajectory
if add_corrected_trajectory:
    x_cf = trajectory_cf.x
    xd_cf = trajectory_cf.xd
    x_error_cf = trajectory_cf.x_error
    xd_error_cf = trajectory_cf.xd_error
    q_cf = trajectory_cf.init_state.pipeline_state.q
    qd_cf = trajectory_cf.init_state.pipeline_state.qd

    # Joint state
    axs[0, 0].plot(q_cf[:, 0], color="tab:green", linestyle="dashed")
    axs[0, 0].plot(q_cf[:, 1], color="tab:green", linestyle="dashdot")
    axs[0, 1].plot(qd_cf[:, 0], color="tab:green", linestyle="dashed")
    axs[0, 1].plot(qd_cf[:, 1], color="tab:green", linestyle="dashdot")

    # End effector state
    axs[1, 0].plot(x_cf[:, 1], label="CF")
    axs[2, 0].plot(x_cf[:, 2], label="CF")
    axs[1, 1].plot(xd_cf[:, 1], label="CF")
    axs[2, 1].plot(xd_cf[:, 2], label="CF")

    # Error
    axs[3, 0].plot(x_error_cf, label="CF")
    axs[3, 1].plot(xd_error_cf, label="CF")

# Set x-axis limits
axs[0, 0].set_xlim(left=plot_start)
axs[1, 0].set_xlim(left=plot_start)
axs[2, 0].set_xlim(left=plot_start)
axs[0, 1].set_xlim(left=plot_start)
axs[1, 1].set_xlim(left=plot_start)
axs[2, 1].set_xlim(left=plot_start)
axs[3, 0].set_xlim(left=plot_start)
axs[3, 1].set_xlim(left=plot_start)

# Create legend in the new subplot
handles, labels = axs[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)

# Hide the new subplot
axs[4, 0].axis("off")
axs[4, 1].axis("off")

plt.tight_layout()
plt.savefig("figures/double_pend.png")


# Plot friction correction torques
if add_corrected_trajectory:
    friction_correction = compute_friction_torques(
        loaded_params, trajectory_cf.init_state.obs
    )
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.75))
    fig.suptitle("Friction Correction Torques")
    axs[0].plot(friction_correction[:, 0], label="Joint 1")
    axs[0].set_ylabel("Torque [Nm]")
    axs[0].set_xlabel("Steps")
    axs[0].set_title("Joint 1")
    axs[1].plot(friction_correction[:, 1], label="Joint 2")
    axs[1].set_ylabel("Torque [Nm]")
    axs[1].set_xlabel("Steps")
    axs[1].set_title("Joint 2")
    plt.tight_layout()
    plt.savefig("figures/friction_correction.png")


# Plot friction and control torques for the trajectory with friction
# One subplot for each joint
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.75))
fig.suptitle("Friction Torques")
friction = env_yf.calculate_friction(trajectory_yf.init_state)
control_torque = trajectory_yf.control_torque
axs[0].plot(control_torque[:, 0], label="Control")
axs[0].plot(friction[:, 0], label="Friction")
axs[0].set_ylabel("Torque [Nm]")
axs[0].set_xlabel("Steps")
axs[0].set_title("Joint 1")
axs[1].plot(control_torque[:, 1], label="Control")
axs[1].plot(friction[:, 1], label="Friction")
axs[1].set_ylabel("Torque [Nm]")
axs[1].set_xlabel("Steps")
axs[1].set_title("Joint 2")
plt.legend()
plt.tight_layout()
plt.savefig("figures/control_and_friction.png")