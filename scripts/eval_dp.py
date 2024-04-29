import jax
from brax.envs.double_pendulum import DoublePendulum
from brax.robots.double_pendulum.utils import DoublePendulumUtils
from jax import numpy as jp
import matplotlib.pyplot as plt

# jax.config.update("jax_disable_jit", True)

# Sim parameters
q_init = jp.array([jp.pi/4, 0.0])
qd_init = jp.array([0.0, 0.0])
steps = 150
action = jp.array(
    [0.0, 2.0 * jp.sin(jp.pi / 4), jp.sin(jp.pi / 4), 0.0, 0.0, 0.0]
)  # 2 o'clock position

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
            DoublePendulumUtils.end_effector_position(
                state_yf.pipeline_state.q
            ),
        )
    )

    # Print
    print(f"Step: {i}")

# Plot everything in subplots.
# The first row of subplots shows the q values for the NF and YF environments.
# The second row of subplots shows the qd values for the NF and YF environments.
fig, axs = plt.subplots(3, 1)
fig.suptitle("Double Pendulum Environment")
axs[0].plot(x_nf[:, 0], label="NF")
axs[0].plot(x_yf[:, 0], label="YF")
axs[0].hlines(y=action[0], xmin=0, xmax=len(x_nf), colors='r', linestyles='--', label='Reference')
axs[0].set_ylabel("x [m]")
axs[0].legend()
axs[1].plot(x_nf[:, 1], label="NF")
axs[1].plot(x_yf[:, 1], label="YF")
axs[1].hlines(y=action[1], xmin=0, xmax=len(x_nf), colors='r', linestyles='--', label='Reference')
axs[1].set_ylabel("y [m]")
axs[1].legend()
axs[2].plot(x_nf[:, 2], label="NF")
axs[2].plot(x_yf[:, 2], label="YF")
axs[2].hlines(y=action[2], xmin=0, xmax=len(x_nf), colors='r', linestyles='--', label='Reference')
axs[2].set_ylabel("z [m]")
axs[2].legend()

plt.tight_layout()
plt.savefig("figures/double_pend.png")
