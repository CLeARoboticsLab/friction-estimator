from matplotlib import pyplot as plt
from jax import random, numpy as jnp
from jax import jit, value_and_grad
from flax import linen as nn
from typing import Sequence
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import optax


# Parameters 
n_epochs = 2
batch_size_train = 256
batch_size_test = 64
input_dim = 28*28
learning_rate = 0.01
log_interval = 10
layer_features = [256, 10]


# Helper function for images
def show_img(img, ax=None, title=None):
    """Show a single image."""
    if ax is None:
        ax = plt.gca()
    ax.imshow(img, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if not title is None:
        ax.set_title(f"{title}")


def show_img_grd(imgs, titles):
    """Shows a grid of images."""
    n = int(jnp.ceil(len(imgs)**0.5))
    _, axs = plt.subplots(n, n, figsize=(3*n, 3*n))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        show_img(img, axs[i // n][i % n], title)
    plt.savefig('figures/grid.png', dpi=300)


# Define MLP 
class SimpleMLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, num_of_features in enumerate(self.features):
            x = nn.Dense(num_of_features, name=f"Layer_{i}")(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
            else:
                x = nn.softmax(x)
        return x


# Data loaders
train_loader = DataLoader(
    MNIST('~/data/', train=True, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
                                batch_size=batch_size_train, shuffle=True)

test_loader = DataLoader(
    MNIST('~/data/', train=False, download=True,
                            transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
                            batch_size=batch_size_test, shuffle=True)

# Show a single image
# train_examples = enumerate(train_loader)
# batch_idx, (example_batch, example_targets) = next(train_examples)
# show_img_grd(example_batch.squeeze(1)[:9], example_targets[:9].tolist())


# Initialize model 
key1, key2 = random.split(random.key(0))
x = random.uniform(key1, (batch_size_train, input_dim))

model = SimpleMLP(features=layer_features)
params = model.init(key2, x)
y = model.apply(params, x)


# Helper function to convert and reshape PyTorch tensors to JAX arrays
def prepare_inputs(pytorch_tensor):
    # Convert PyTorch tensor to numpy, then to JAX array
    jax_array = jnp.array(pytorch_tensor.numpy())
    # Reshape [B, 1, 28, 28] to [B, 784]
    return jax_array.reshape(jax_array.shape[0], -1)


# Define loss
def make_loss(model: SimpleMLP):
    def batched_cross_entropy(params, inputs_batched, targets_batched):
        targets_one_hot = jnp.eye(10)[targets_batched]
        preds_batched = model.apply(params, inputs_batched)
        return -jnp.mean(jnp.sum(targets_one_hot * jnp.log(preds_batched + 1e-10), axis=1))
    
    return batched_cross_entropy


loss_fn = make_loss(model)


# Initial loss for a single batch
train_examples = enumerate(train_loader)
batch_idx, (example_batch, example_targets) = next(train_examples)
jax_example_batch = prepare_inputs(example_batch)
jax_example_targets = jnp.array(example_targets.numpy())
loss = loss_fn(params, jax_example_batch, jax_example_targets)
print(f"Initial loss: {loss}")

# Setup optimizer 
tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)
loss_grad_fn = value_and_grad(loss_fn)

# Run training loop 
history_loss_train = []
history_loss_test = []
for i in range(n_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = prepare_inputs(inputs)
        targets = jnp.array(targets.numpy())
        loss, grads = loss_grad_fn(params, inputs, targets)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        history_loss_train.append(loss)
        if batch_idx % log_interval == 0:
            print(f"Epoch {i}, Batch {batch_idx}, Training Loss: {loss}")

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = prepare_inputs(inputs)
        targets = jnp.array(targets.numpy())
        loss = loss_fn(params, inputs, targets)
        print(f"Epoch {i}, Batch {batch_idx}, Validation Loss: {loss}")
        history_loss_test.append(loss)

# Plot loss history
plt.plot(history_loss_train, label='train')
plt.plot(history_loss_test, label='test')
plt.legend()
plt.savefig('figures/loss.png', dpi=300)

# Run predictions on a random test batch
test_examples = enumerate(test_loader)
batch_idx, (example_batch, example_targets) = next(test_examples)
jax_example_batch = prepare_inputs(example_batch)
jax_example_targets = jnp.array(example_targets.numpy())
preds = model.apply(params, jax_example_batch)
preds = jnp.argmax(preds, axis=1)
print(f"Predictions: {preds}")
print(f"True labels: {jax_example_targets}")

show_img_grd(example_batch.squeeze(1)[:9], preds[:9].tolist())