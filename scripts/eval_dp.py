import jax
from brax.envs.double_pendulum import DoublePendulum
from brax.robots.double_pendulum.utils import DoublePendulumUtils
from jax import numpy as jp
import matplotlib.pyplot as plt

# jax.config.update("jax_disable_jit", True)

# Sim parameters
q_init = jp.array([-jp.pi, 0.0])
qd_init = jp.array([0.0, 0.0])
steps = 400
action = jp.array(
    [0.0, 2.0, 0.0]
)  # position control only

# Setup Brax environment
env_nf = DoublePendulum()
env_yf = DoublePendulum()

step_nf_jitted = jax.jit(env_nf.step)
step_yf_jitted = jax.jit(env_yf.step_with_friction)
set_nf_jitted = jax.jit(env_nf.set_state)
set_yf_jitted = jax.jit(env_yf.set_state)

# Reset the environment
state_nf = set_nf_jitted(q_init, qd_init)
state_yf = set_yf_jitted(q_init, qd_init)

# Save q and qd
x_nf = DoublePendulumUtils.end_effector_position(
        state_nf.pipeline_state.q
    )
x_yf = DoublePendulumUtils.end_effector_position(
    state_yf.pipeline_state.q
)
xd_nf = DoublePendulumUtils.end_effector_velocity(
    state_nf.pipeline_state.q, state_nf.pipeline_state.qd
)
xd_yf = DoublePendulumUtils.end_effector_velocity(
    state_yf.pipeline_state.q, state_yf.pipeline_state.qd
)
for i in range(steps):
    state_nf = step_nf_jitted(state_nf, action)
    state_yf = step_yf_jitted(state_yf, action)

    # Compute the end effector position
    x_nf = jp.vstack(
        (
            x_nf,
            DoublePendulumUtils.end_effector_position(
                state_nf.pipeline_state.q
            ),
        )
    )

    x_yf = jp.vstack(
        (
            x_yf,
            jp.array(DoublePendulumUtils.end_effector_position(
                state_yf.pipeline_state.q
            )),
        )
    )

    # Compute the end effector velocity
    xd_nf = jp.vstack(
        (
            xd_nf,
            DoublePendulumUtils.end_effector_velocity(
                state_nf.pipeline_state.q, state_nf.pipeline_state.qd
            ),
        )
    )

    xd_yf = jp.vstack(
        (
            xd_yf,
            DoublePendulumUtils.end_effector_velocity(
                state_yf.pipeline_state.q, state_yf.pipeline_state.qd
            ),
        )
    )

    # Print
    if i % 10 == 0:
        print(f"Step: {i}")

# Plot everything in subplots.
# The first row of subplots shows the q values for the NF and YF environments.
# The second row of subplots shows the qd values for the NF and YF environments.
fig, axs = plt.subplots(4, 2)
fig.suptitle("Double Pendulum Environment")
margin = 0.1

axs[0, 0].plot(x_nf[:, 0], label="NF")
axs[0, 0].plot(x_yf[:, 0], label="YF")
axs[0, 0].hlines(
    y=action[0],
    xmin=0,
    xmax=len(x_nf),
    colors="r",
    linestyles="--",
    label="Reference",
)
axs[0, 0].set_ylabel("x [m]")
axs[0, 0].set_ylim([-2-margin, 2+margin])

axs[1, 0].plot(x_nf[:, 1], label="NF")
axs[1, 0].plot(x_yf[:, 1], label="YF")
axs[1, 0].hlines(
    y=action[1],
    xmin=0,
    xmax=len(x_nf),
    colors="r",
    linestyles="--",
    label="Reference",
)
axs[1, 0].set_ylabel("y [m]")
axs[1, 0].set_ylim([-2-margin, 2+margin])

axs[2, 0].plot(x_nf[:, 2], label="NF")
axs[2, 0].plot(x_yf[:, 2], label="YF")
axs[2, 0].hlines(
    y=action[2],
    xmin=0,
    xmax=len(x_nf),
    colors="r",
    linestyles="--",
    label="Reference",
)
axs[2, 0].set_ylabel("z [m]")
axs[2, 0].set_xlabel("Steps")
axs[2, 0].set_ylim([-2-margin, 2+margin])

axs[0, 1].plot(xd_nf[:, 0], label="NF")
axs[0, 1].plot(xd_yf[:, 0], label="YF")
axs[0, 1].hlines(
    y=action[3],
    xmin=0,
    xmax=len(x_nf),
    colors="r",
    linestyles="--",
    label="Reference",
)
axs[0, 1].set_ylabel("xd [m/s]")

axs[1, 1].plot(xd_nf[:, 1], label="NF")
axs[1, 1].plot(xd_yf[:, 1], label="YF")
axs[1, 1].hlines(
    y=action[4],
    xmin=0,
    xmax=len(x_nf),
    colors="r",
    linestyles="--",
    label="Reference",
)
axs[1, 1].set_ylabel("yd [m/s]")

axs[2, 1].plot(xd_nf[:, 2], label="NF")
axs[2, 1].plot(xd_yf[:, 2], label="YF")
axs[2, 1].hlines(
    y=action[5],
    xmin=0,
    xmax=len(x_nf),
    colors="r",
    linestyles="--",
    label="Reference",
)
axs[2, 1].set_ylabel("zd [m/s]")
axs[2, 1].set_xlabel("Steps")


for i in range(3):
    for j in range(2):
        axs[i, j].margins(x=0, y=margin)

# Create legend in the new subplot
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2)

# Hide the new subplot
axs[3, 0].axis('off')
axs[3, 1].axis('off')

plt.tight_layout()
plt.savefig("figures/double_pend.png")
