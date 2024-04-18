import jax
import optax
import flax
import pickle

from brax.training import networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.envs import State
from brax.envs.panda import Panda
from jax import numpy as jp
from jax.config import config

# jax.config.update("jax_disable_jit", True)
config.update("jax_enable_x64", True)

# Define the dataclass
@flax.struct.dataclass
class MyData:
    init_state: State
    torque: jp.ndarray
    friction: jp.ndarray
    next_state: State


# Load data
# with open('brax/scripts/data/data.pkl', 'rb') as f:
#     data = pickle.load(f)
with open('data/data.pkl', 'rb') as f:
    data = pickle.load(f)

# Training parameters
num_joints = 7
batch_size = 256
data_length = 1024
num_epochs = 20
input_dim = 28 * 28
learning_rate = 1e-3
log_interval = 10
input_size = num_joints * 2
hidden_layer_dim = 256
hidden_layer_num = 3
output_size = num_joints
seed = 0

# Define the network
network = networks.MLP(
    layer_sizes=([hidden_layer_dim] * hidden_layer_num + [output_size])
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


# Setup Brax environment
seed = 0
env_brax = Panda()
env_reset_jitted = jax.jit(env_brax.reset)
env_step_jitted = jax.jit(env_brax.step)


# Loss function
def loss_fn(params, data):

    initial_pose = jp.concatenate(
        [data.init_state.pipeline_state.q, data.init_state.pipeline_state.qd],
        axis=1,
    )
    torques_friction = network.apply(
        params, initial_pose
    )  # i/o both (batch_size, num_joints)

    # Compute the next state
    next_state = jax.vmap(env_step_jitted)(
        data.init_state, data.torque + torques_friction
    )

    return jp.mean(
        (next_state.pipeline_state.q - data.next_state.pipeline_state.q) ** 2
        + (next_state.pipeline_state.qd - data.next_state.pipeline_state.qd)
        ** 2
    )


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

    print(f"epoch losses {losses}")

    return TrainingState(
        opt_state=new_opt_state, params=new_params
    ), jp.mean(losses)


# Run training loop
key_init = jax.random.PRNGKey(seed)
training_state = _init_training_state(key_init, network, optimizer)
for epoch in range(num_epochs):
    key, key_init = jax.random.split(key_init)
    training_state, epoch_loss = train_epoch(
        training_state, data, key
    )
    print(f"epoch {epoch}, loss {epoch_loss:.2f}")