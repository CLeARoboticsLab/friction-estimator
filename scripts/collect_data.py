import jax
from jax import numpy as jp
from brax.envs.panda import Panda
from brax.envs import State
import flax
import time
import pickle
from jax.config import config
import jax.tree_util as jtu

import matplotlib.pyplot as plt

# -----------------------
# --- Debug Utils -------
# -----------------------
# jax.config.update("jax_disable_jit", True)


# -----------------------
# --- Sim Parameters ----
# -----------------------

# Collection limits
batch_size = 32
num_batches = 2
torque_logging_interval = 100
num_joints = 7
seed = 0

# Control params
torque_max = 10.0
torque_lims = jp.array([-torque_max, torque_max])
joint_lims_max = jp.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
)
joint_lims_min = jp.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
)

# -----------------------
# --- Brax stuff --------
# -----------------------

print("Loading Brax environment...")
start_time = time.time()

env = Panda()
step_jitted = jax.jit(env.step_directly)
reset_jitted = jax.jit(env.reset)
friction_jitted = jax.jit(env.calculate_friction)

print(f"Done. Time taken: {time.time() - start_time}")


# ----------------------------
# --- Collect data  ----------
# ----------------------------


# Data class
@flax.struct.dataclass
class MyData:
    init_state: State
    torque: jp.ndarray
    friction: jp.ndarray
    next_state: State


# Data collection function
def make_data(key):
    # Split key
    rng1, rng2 = jax.random.split(key, num=2)

    # Sample init brax state and compute friction torques
    init_state = reset_jitted(rng1)

    # Sample random torque
    torque = jax.random.uniform(
                rng2,
                (num_joints,),
                minval=torque_lims[0],
                maxval=torque_lims[1],
            )

    # Calculate friction
    friction = friction_jitted(init_state)

    # Step in both environments
    next_state = step_jitted(init_state, torque + friction)

    return MyData(
        init_state=init_state.obs,
        torque=torque,
        friction=friction,
        next_state=next_state.obs,
    )


# Data stacking function to append batches
def tree_stack(trees):
    return jtu.tree_map(lambda *v: jp.vstack(v), *trees)


# Collect data
print("Starting data collection loop...")
start_time = time.time()

batch_keys_init = jax.random.split(jax.random.key(seed), num=num_batches)
batch_start_time = time.time()

batch_keys_1 = jax.random.split(batch_keys_init[0], num=batch_size)
data = jax.vmap(make_data)(batch_keys_1)

print(
        f"  Batch {1}/{num_batches}. Time taken: {time.time() - batch_start_time}"
    )

for batch_i, batch_key in enumerate(batch_keys_init[1:]):
    batch_start_time = time.time()

    batch_keys = jax.random.split(batch_key, num=batch_size)
    batch_data = jax.vmap(make_data)(batch_keys)
    data = tree_stack([data, batch_data])

    print(
        f"  Batch {batch_i + 2}/{num_batches}. Time taken: {time.time() - batch_start_time}"
    )

print(f"Done. Time taken: {time.time() - start_time}"),

# Save data
print("Saving data...")
start_time = time.time()

with open("data/data_panda.pkl", "wb") as f:
    pickle.dump(data, f)

print(f"Done. Time taken: {time.time() - start_time}")
