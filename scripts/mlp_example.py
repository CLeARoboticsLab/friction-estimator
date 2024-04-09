import jax
from jax import random, numpy as jnp
import flax
from flax import linen as nn
from typing import Sequence, Callable

input_dim = 4
feature_dims = [3, 4, 5]
n_samples = 20
noise_level = 0.1


# Define simple MLP using Flax
class ExplicitMLP(nn.Module):
    features: Sequence[int]

    # Setup layer kernels with output size according to features
    def setup(self):
        self.layers = [nn.Dense(feature) for feature in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x


# Generate a dummy input
key1, key2 = random.split(random.key(0))
x = random.uniform(key1, (n_samples, input_dim))

model = ExplicitMLP(features=feature_dims)
params = model.init(key2, x)
y = model.apply(params, x)

print(f"Parameter shapes:\n{jax.tree_util.tree_map(lambda x: x.shape, params)}")
print(f"Input:\n{x}")
print(f"Output:\n{y}")


# Simple MLP
class SimpleMLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, num_of_features in enumerate(self.features):
            x = nn.Dense(num_of_features, name=f"Layer_{i}")(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x


model = SimpleMLP(features=[3,4,5])
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, flax.core.unfreeze(params)))
print(f"Input:\n{x}")


# Define our own dense layer
class SimpleDense(nn.Module):
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact 
    def __call__(self, inputs):
        kernel = self.param('kernel',
                        self.kernel_init,  # Initialization function
                        (inputs.shape[-1], self.features))  # shape info.
        y = jnp.dot(inputs, kernel)
        bias = self.param('bias', self.bias_init, (self.features,))
        y = y + bias
        return y

key1, key2 = random.split(random.key(0), 2)
x = random.uniform(key1, (n_samples, input_dim))

model = SimpleDense(features=3)
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameters:\n', params)
print('output:\n', y)


# Define a stateful layer
class BiasAdderWithRunningMean(nn.Module):
    decay: float = 0.99

    @nn.compact
    def __call__(self, x):
        is_initalized = self.has_variable('batch_stats', 'mean')
        ra_mean = self.variable('batch_stats', 'mean',
                                lambda s: jnp.zeros(s),
                                x.shape[1:])
        bias = self.param('bias', lambda rng, shape: jnp.zeros(shape), x.shape[1:])
        if is_initalized:
            ra_mean.value = self.decay * ra_mean.value + (1.0 - self.decay) * jnp.mean(x, axis=0, keepdims=True)

        return x - ra_mean.value + bias

key1, key2 = random.split(random.key(0), 2)
x = jnp.ones((10, 5))
model = BiasAdderWithRunningMean()
variables = model.init(key1, x)
print(f"Initialized variables:\n{variables}")
y, updated_state = model.apply(variables, x, mutable=['batch_stats'])
print(f"updated state:\n{updated_state}")

for val in [1.0, 2.0, 3.0]:
    x = val * jnp.ones((10, 5))
    y, updated_state = model.apply(variables, x, mutable=['batch_stats'])
    old_state, params = flax.core.pop(variables, 'params')
    variables = flax.core.freeze({'params': params, **updated_state})
    print(f"updated state:\n{updated_state}")