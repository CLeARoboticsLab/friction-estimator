import jax
import optax
import flax
import pickle
import time
import matplotlib.pyplot as plt

from brax.training import networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.envs import State
from brax.envs.double_pendulum import DoublePendulum
from jax import numpy as jp
from jax.config import config

# Debugging utilities
# jax.config.update("jax_disable_jit", True)
# config.update("jax_enable_x64", True)


# ----------------------------
# --- Load data --------------
# ----------------------------

# Define the data class
@flax.struct.dataclass
class MyData:
    init_state: State
    torque: jp.ndarray
    friction: jp.ndarray
    next_state: State


# Load data
# with open('brax/scripts/data/data.pkl', 'rb') as f:
#     data = pickle.load(f)
with open("data/data.pkl", "rb") as f:
    data = pickle.load(f)

# Training parameters
num_joints = 2
batch_size = 256
data_length = data.torque.shape[0]
test_length = 1024
num_epochs = 1000
learning_rate = 1e-3
log_interval = 10
input_size = num_joints * 2
hidden_layer_dim = 256
hidden_layer_num = 3
output_size = num_joints
seed = 0


# ----------------------------
# --- Brax -------------------
# ----------------------------

# Setup Brax environment
env_brax = DoublePendulum()
env_step_jitted = jax.jit(env_brax.step_directly)


# ----------------------------
# --- Network ----------------
# ----------------------------

# Define the network
network = networks.MLP(
    layer_sizes=([hidden_layer_dim] * hidden_layer_num + [output_size]),
    activation=flax.linen.gelu
)
optimizer = optax.adam(learning_rate)
key = jax.random.key(seed)


# Define training state
# Holds the optimizer state and the network parameters
@flax.struct.dataclass
class TrainingState:
    opt_state: optax.OptState
    params: Params


# Initialize the training state
def _init_training_state(
    key: PRNGKey,
    network: networks.MLP,
    optimizer: optax.GradientTransformation,
) -> TrainingState:
    dummy_X = jp.zeros((input_size))
    params = network.init(key, dummy_X)
    opt_state = optimizer.init(params)

    training_state = TrainingState(
        opt_state=opt_state,
        params=params,
    )

    return training_state


# ----------------------------
# --- Normalization ----------
# ----------------------------


# Normalization utils
# Normalizes inputs so they're all within the range [0, 1]
# Note that the first 2 elements of the input are the joint angles and will be normalized differently

# Normalization parameters
# Joint means and standard deviations
# A mean and std dev for each joint angle and joint velocity

@flax.struct.dataclass
class NormalizationParameters:
    translation: jp.ndarray
    scaling: jp.ndarray


def compute_norm_params(data):
    joint_pos = jp.concatenate(
        [
            jax.vmap(lambda state: state[0:num_joints])(data.init_state.obs),
        ]
    )
    joint_vel = jp.concatenate(
        [
            jax.vmap(lambda state: state[num_joints:])(data.init_state.obs),
        ]
    )
    joint_pos_mean = jp.mean(joint_pos, axis=0)
    joint_pos_std = jp.std(joint_pos, axis=0)
    joint_vel_mean = jp.mean(joint_vel, axis=0)
    joint_vel_std = jp.std(joint_vel, axis=0)  

    return NormalizationParameters(
        translation=jp.concatenate([joint_pos_mean, joint_vel_mean]),
        scaling=jp.concatenate([joint_pos_std, joint_vel_std]),
    )


norm_params = compute_norm_params(data)


def normalize_joint_state(joint_state):
    return jax.tree_util.tree_map(
        lambda state: (state - norm_params.translation) / norm_params.scaling,
        joint_state,
    )


def compute_friction_torques(params, obs):
    obs = normalize_joint_state(obs)
    return network.apply(params, obs)


# Save
with open("data/norm_params.pkl", "wb") as f:
    pickle.dump(norm_params, f)


# ----------------------------
# --- Training ---------------
# ----------------------------

# Loss function
def loss_fn(params, data):

    # Compute the friction torques
    correction = compute_friction_torques(
        params, data.init_state.obs
    )  # i/o both (batch_size, num_joints)

    # Compute the next state
    next_state = jax.vmap(env_step_jitted)(
        data.init_state, data.torque + correction
    )

    return jp.mean((next_state.obs - data.next_state.obs) ** 2)


def sgd_step(carry, in_element):
    # carry is the carry, in_element is the input data
    params, opt_state = carry
    data = in_element
    sgd_loss, params_grad = jax.value_and_grad(loss_fn)(params, data)
    updates, opt_state = optimizer.update(params_grad, opt_state)
    new_params = optax.apply_updates(params, updates)
    return (new_params, opt_state), sgd_loss


# Define training epoch
@jax.jit
def train_epoch(training_state: TrainingState, data, key):

    # Setup batch
    permuted_idx = jax.random.permutation(key, data_length)
    num_batches = data_length // batch_size
    shuffled_data = jax.tree_util.tree_map(lambda x: x[permuted_idx], data)
    batched_data = jax.tree_util.tree_map(
        lambda x: jp.reshape(x, (num_batches, batch_size) + x.shape[1:]),
        shuffled_data,
    )

    # Training for epoch
    (new_params, new_opt_state), losses = jax.lax.scan(
        sgd_step,
        (training_state.params, training_state.opt_state),
        batched_data,
    )

    return TrainingState(opt_state=new_opt_state, params=new_params), jp.mean(
        losses
    )


# Split data
data_train = jax.tree_util.tree_map(
    lambda x: x[: data_length - test_length], data
)
data_test = jax.tree_util.tree_map(
    lambda x: x[data_length - test_length:], data
)


# Eval function for test data
# Requires knowledge of friction
def eval_loss_fn(params, data):

    # Compute the friction torques
    correction = compute_friction_torques(
        params, data.init_state.obs
    )  # i/o both (batch_size, num_joints)

    # Compute the next state
    next_state = jax.vmap(env_step_jitted)(
        data.init_state, data.torque + correction
    )

    return jp.mean((next_state.obs - data.next_state.obs) ** 2)


def eval_step(carry, in_element):
    params = carry
    data = in_element
    eval_loss = eval_loss_fn(params, data)
    return params, eval_loss


@jax.jit
def evaluate_epoch(training_state: TrainingState, data):
    batched_data = jax.tree_util.tree_map(
        lambda x: jp.reshape(x, (1, test_length) + x.shape[1:]),
        data,
    )
    _, eval_losses = jax.lax.scan(
        eval_step, training_state.params, batched_data
    )
    return jp.mean(eval_losses)


# Run training loop
key_init = jax.random.PRNGKey(seed)
training_state = _init_training_state(key_init, network, optimizer)
losses = []
eval_losses = []
print("Training started...")
start_time = time.time()
for epoch in range(num_epochs):
    start_time_epoch = time.time()
    # Train
    key, key_init = jax.random.split(key_init, 2)
    training_state, epoch_loss = train_epoch(training_state, data_train, key)
    losses.append(epoch_loss)

    # Eval
    epoch_eval_loss = evaluate_epoch(training_state, data_test)
    eval_losses.append(epoch_eval_loss)

    print(
        f"epoch {epoch}, loss: {epoch_loss}, eval loss: {eval_losses[-1]}, time taken: {time.time() - start_time_epoch}"
    )
print(f"Training finished. Time taken: {time.time() - start_time}")


# Save model
with open("data/model_params.pkl", "wb") as f:
    pickle.dump(training_state.params, f)


# ----------------------------
# --- Plotting ---------------
# ----------------------------

# Plot validation and training loss in the same plot
plt.figure()
plt.plot(losses[1:], label="Training Loss")
plt.plot(eval_losses[1:], label="Evaluation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")  # Set y-axis to logarithmic scale
plt.legend()
plt.title(
    f"Training and Evaluation Loss\n"
    f"LR: {learning_rate}, "
    f"Batch Size: {batch_size}, "
    f"Hidden Layers: {hidden_layer_num}, "
    f"Layer Dims: {hidden_layer_dim}"
)
plt.savefig("figures/training_and_evaluation_loss.png")
