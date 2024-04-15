import jax
import optax
import flax
import numpy as np
from brax.training import networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.envs.base import State
from brax.training import gradients
from brax.envs.panda import Panda
from jax import numpy as jp


# Training parameters
num_joints = 7
num_epochs = 2
batch_size = 256
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
brax_init_state = env_reset_jitted(jax.random.PRNGKey(seed))
env_set_state_jitted = jax.jit(env_brax.set_state)


# Loss function
def make_loss(network: networks.MLP):
    def loss(
        params: Params,
        state_init: State,
        torques_osc: jp.ndarray,
        state_new: State,
    ):
        # Compute friction
        torques_friction = network.apply(
            params,
            jp.concatenate(
                [
                    state_init.pipeline_state.q_0,
                    state_init.pipeline_state.qd_0,
                ]
            ),
        )
        
        # Step brax
        state_torqued = env_step_jitted(
            state_init, torques_osc + torques_friction
        )

        # Compute loss
        loss = jp.mean(
            (
                jp.concatenate(
                    [
                        state_new.pipeline_state.q,
                        state_new.pipeline_state.qd,
                    ]
                )
                - jp.concatenate(
                    [
                        state_torqued.pipeline_state.q,
                        state_torqued.pipeline_state.qd,
                    ]
                )
            )
            ** 2
        )
        return loss

    return loss


# Init training state and function
loss_fn = make_loss(network)
update = gradients.gradient_update_fn(loss_fn, optimizer,
                                      pmap_axis_name=None,
                                      has_aux=False)
key, key_init = jax.random.split(key)
training_state = _init_training_state(key_init, network, optimizer)


# Define gradient descent
def sgd_step(carry, in_element):
    params, opt_state = carry
    states_init, torques_osc, states_new = in_element
    loss, new_params, opt_state = update(
        params, states_init, torques_osc, states_new, optimizer_state=opt_state
    )
    return (new_params, opt_state), loss


# Define training epoch
@jax.jit
def train_epoch(
    training_state: TrainingState,
    states_init: jp.ndarray,
    torques_osc: jp.ndarray,
    states_new: jp.ndarray,
):
    # normally you would shuffle data here
    # reshape into (num_batches, batch_size, dim)
    num_batches = states_init.shape[0] // batch_size
    states_init_batch = states_init.reshape(num_batches, batch_size)
    torques_osc_batch = torques_osc.reshape(num_batches, batch_size)
    states_new_batch = states_new.reshape(num_batches, batch_size)

    print("test")

    (new_params, new_opt_state), losses = jax.lax.scan(
        sgd_step,
        (training_state.params, training_state.opt_state),
        (states_init_batch, torques_osc_batch, states_new_batch))

    new_training_state = training_state.replace(
        params=new_params, opt_state=new_opt_state)

    return new_training_state, jp.mean(losses)


# Load data
states_init = jp.load('data_states_init.npy')
torques_osc = jp.load('data_torques_osc.npy')
states_new = jp.load('data_states_new.npy')


# Run training loop
for epoch in range(num_epochs):
    training_state, loss = train_epoch(
        training_state, states_init, torques_osc, states_new
    )
    print(f"epoch {epoch}, loss {loss:.2f}")
