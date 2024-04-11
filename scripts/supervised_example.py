import optax
import flax
import jax
from jax import numpy as jp
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.training import networks
from brax.training import gradients


@flax.struct.dataclass
class TrainingState:
    opt_state: optax.OptState
    params: Params


num_data = 1000
batch_size = 100
num_epochs = 20
input_size = 10
hidden_size = 256
num_hidden = 2
output_size = 4
learning_rate = 1e-3
seed = 0

network = networks.MLP(layer_sizes=([hidden_size]*num_hidden + [output_size]))
optimizer = optax.adam(learning_rate)
key = jax.random.PRNGKey(seed)


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


def make_loss(network: networks.MLP):
    def loss(params: Params, x: jp.ndarray, y: jp.ndarray):
        y_pred = network.apply(params, x)
        loss = jp.mean((y - y_pred)**2)
        return loss
    return loss


loss_fn = make_loss(network)

update = gradients.gradient_update_fn(loss_fn, optimizer,
                                      pmap_axis_name=None,
                                      has_aux=False)

key, key_init = jax.random.split(key)
training_state = _init_training_state(key_init, network, optimizer)

key, key_inputs, key_labels = jax.random.split(key, 3)
inputs = jax.random.uniform(key_inputs, (num_data, input_size))
labels = jax.random.uniform(key_labels, (num_data, output_size))


def sgd_step(carry, in_element):
    params, opt_state = carry
    x, y = in_element
    loss, new_params, opt_state = update(
        params, x, y, optimizer_state=opt_state)
    return (new_params, opt_state), loss


@jax.jit
def train_epoch(training_state: TrainingState, inputs: jp.ndarray,
                labels: jp.ndarray):
    # normally you would shuffle data here
    # reshape into (num_batches, batch_size, dim)
    inputs = inputs.reshape(-1, batch_size, input_size)
    labels = labels.reshape(-1, batch_size, output_size)

    (new_params, new_opt_state), losses = jax.lax.scan(
        sgd_step,
        (training_state.params, training_state.opt_state),
        (inputs, labels))

    new_training_state = training_state.replace(
        params=new_params, opt_state=new_opt_state)

    return new_training_state, jp.mean(losses)


for epoch in range(num_epochs):
    training_state, loss = train_epoch(training_state, inputs, labels)
    print(f'epoch {epoch}, loss {loss:.2f}')


def make_inference_fn(network: networks.MLP):
    def inference_fn(params: Params, x: jp.ndarray):
        return network.apply(params, x)
    return inference_fn


# do inference
inference_fn = make_inference_fn(network)
x = jp.ones((input_size))
y_pred = inference_fn(training_state.params, x)
print(y_pred)

# def fernando_loss(params: Params, starting_state: envs.State,
#                   actual_obs: jp.ndarray):
    
#     action = network.apply(params, starting_state)
#     nstate = env.step(starting_state, action)
#     new_obs = nstate.obs
#     loss = jp.mean((new_obs - actual_obs)**2)
#     return loss