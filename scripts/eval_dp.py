import jax
from brax.envs.double_pendulum import DoublePendulum
from jax import numpy as jp
import matplotlib.pyplot as plt

# Sim parameters
q_init = jp.array([0.0, 0.0])
qd_init = jp.array([0.0, 0.0])
steps = 200
action = jp.array([jp.pi / 4, 0.0, 0.0, 0.0])  # [q1, q2, qd1, qd2]

# Setup Brax environment
seed = 0
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
q_nf = state_nf.pipeline_state.q
qd_nf = state_nf.pipeline_state.qd
q_yf = state_yf.pipeline_state.q
qd_yf = state_yf.pipeline_state.qd
for i in range(steps):
    state_nf = step_nf_jitted(state_nf, action)
    state_yf = step_yf_jitted(state_yf, action)

    print(f"Step {i}")
    print(f"NF: {state_nf.obs}")
    print(f"YF: {state_yf.obs}")
    print("\n")

    q_nf = jp.vstack((q_nf, state_nf.pipeline_state.q))
    qd_nf = jp.vstack((qd_nf, state_nf.pipeline_state.qd))
    q_yf = jp.vstack((q_yf, state_yf.pipeline_state.q))
    qd_yf = jp.vstack((qd_yf, state_yf.pipeline_state.qd))


# Plot everything in subplots.
# The first row of subplots shows the q values for the NF and YF environments.
# The second row of subplots shows the qd values for the NF and YF environments.
fig, axs = plt.subplots(2, 2)
fig.suptitle("Double Pendulum Environment")
axs[0, 0].plot(q_nf[:, 0], label="NF")
axs[0, 0].plot(q_yf[:, 0], label="YF")
axs[0, 0].set_title("q1")
axs[0, 0].set_ylabel("Value")
axs[0, 0].legend()
axs[0, 1].plot(q_nf[:, 1], label="NF")
axs[0, 1].plot(q_yf[:, 1], label="YF")
axs[0, 1].set_title("q2")
axs[0, 1].set_ylabel("Value")
axs[0, 1].legend()
axs[1, 0].plot(qd_nf[:, 0], label="NF")
axs[1, 0].plot(qd_yf[:, 0], label="YF")
axs[1, 0].set_title("qd1")
axs[1, 0].set_xlabel("Step")
axs[1, 0].set_ylabel("Value")
axs[1, 0].legend()
axs[1, 1].plot(qd_nf[:, 1], label="NF")
axs[1, 1].plot(qd_yf[:, 1], label="YF")
axs[1, 1].set_title("qd2")
axs[1, 1].set_xlabel("Step")
axs[1, 1].set_ylabel("Value")
axs[1, 1].legend()

plt.savefig("figures/double_pend.png")
