import jax
import flax
import time
from brax.envs.double_pendulum import DoublePendulum
from brax.robots.double_pendulum.utils import DoublePendulumUtils
from jax import numpy as jp
import matplotlib.pyplot as plt
from brax.envs import State

# jax.config.update("jax_disable_jit", True)

# Sim parameters
q_init = jp.array([-jp.pi, 0.0])
qd_init = jp.array([0.0, 0.0])
steps = 1000
action = jp.array([0.0, 1.0, 1.0])  # position control only
plot_start = 0

# Setup Brax environment
env_nf = DoublePendulum()
env_yf = DoublePendulum()

step_nf_jitted = jax.jit(env_nf.step)
step_yf_jitted = jax.jit(env_yf.step_with_friction)
set_nf_jitted = jax.jit(env_nf.set_state)
set_yf_jitted = jax.jit(env_yf.set_state)


@flax.struct.dataclass
class MyData:
    init_state: State
    new_state: State
    action: jp.ndarray
    x: jp.ndarray
    xd: jp.ndarray
    x_error: jp.float32
    xd_error: jp.float32


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
        return new_state, MyData(
            init_state, new_state, action, x, xd, x_error, xd_error
        )

    return trajectory_step


trajectory_step_nf = make_trajectory_step(action, step_nf_jitted)
trajectory_step_yf = make_trajectory_step(action, step_yf_jitted)

# Reset the environment
state_nf = set_nf_jitted(q_init, qd_init)
state_yf = set_yf_jitted(q_init, qd_init)

# Run the simulation
print("Generating trajectory")
start_time = time.time()

_, trajectory_nf = jax.lax.scan(
    trajectory_step_nf, state_nf, (), steps
)
_, trajectory_yf = jax.lax.scan(
    trajectory_step_yf, state_yf, (), steps
)

print("Trajectory generated.")
print(f"Time taken: {time.time() - start_time}")

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
axs[3, 0].set_ylabel("Pos. error [m]")
axs[3, 0].set_xlabel("Steps")
axs[3, 0].set_ylim(bottom=0 - margin)

axs[3, 1].plot(xd_error_nf, label="NF")
axs[3, 1].plot(xd_error_yf, label="YF")
axs[3, 1].set_ylabel("Vel. error [m/s]")
axs[3, 1].set_xlabel("Steps")
axs[3, 1].set_ylim(bottom=0 - margin)

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
