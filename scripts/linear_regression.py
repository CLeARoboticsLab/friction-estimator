import jax
from jax import random, numpy as jnp
import flax
from flax import linen as nn
import optax

# Parameters
input_dim = 10
output_dim = 5
n_samples = 20
noise_level = 0.1

learning_rate = 0.3

model = nn.Dense(features=output_dim)

# Initialize parameters 
key1, key2 = random.split(random.PRNGKey(0))
dummy_input = random.normal(key1, (input_dim,))
params = model.init(key2, dummy_input)  # Init params with a dummy input
para_dims = jax.tree_util.tree_map(lambda x: x.shape, params)
dummy_output = model.apply(params, dummy_input)
print(f"Parameter shapes: {para_dims}")

# Generate random ground truth W and b
key = random.key(0)
key1, key2 = random.split(key)
W = random.normal(key1, (input_dim, output_dim))
b = random.normal(key2, (output_dim,))
true_params = flax.core.freeze({'params': {'bias': b, 'kernel': W}})

# Generate data
key_sample, key_noise = random.split(key)
input_samples = random.normal(key_sample, (n_samples, input_dim))
noisy_targets = jnp.dot(input_samples, W) + b + noise_level * random.normal(key_noise, (n_samples, output_dim))

# Define loss
@jax.jit
def mse(params, inputs_batched, targets_batched):
    def squared_error(input, target):
        pred = model.apply(params, input)
        return jnp.inner(target - pred, target - pred)/2
    return jnp.mean(jax.vmap(squared_error)(inputs_batched, targets_batched), axis=0)


print(f"Initial MSE: {mse(params, input_samples, noisy_targets)}") 


# ---- Manual training loop ----

# # Define gradient of loss
# loss_grad_fn = jax.value_and_grad(mse)


# # Define gradient descent update
# @jax.jit
# def update_params(params, learning_rate, grads):
#     params = jax.tree_util.tree_map(
#         lambda p, g: p - learning_rate * g, params, grads)
#     return params


# # Training loop
# for i in range(101):
#     # Perform one gradient update
#     loss_val, grads = loss_grad_fn(params, input_samples, noisy_targets)
#     params = update_params(params, learning_rate, grads)
#     if i % 10 == 0:
#         print(f"Iteration {i}, Loss: {loss_val}")


# ---- Training using Optax ---- 

# Now using optax
tx = optax.adam(learning_rate)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(mse)

# Training loop
for i in range(101):
    # Perform one gradient update
    loss_val, grads = loss_grad_fn(params, input_samples, noisy_targets)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss_val}")