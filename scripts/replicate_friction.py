import jax
import matplotlib.pyplot as plt
from jax import numpy as jp
from brax.envs.double_pendulum import DoublePendulum

# Trying to replicate figures in https://ieeexplore.ieee.org/document/8206141

qds = jp.linspace(-30.0, 30.0, 100)
env = DoublePendulum()  # No friction
friction = jax.vmap(env.calculate_friction)(qds)

plt.plot(qds, friction)
plt.xlabel("qdot")
plt.ylabel("friction")
plt.title("Friction vs qdot")

plt.savefig("figures/friction_test.png")