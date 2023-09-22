import flax.linen as nn
import jax.numpy as jnp
from math import prod


class Linear(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""
    # Linear layer in spacial dimension, nonlinear in time
    @nn.compact
    def __call__(self, x, t):
        x_shape = x.shape
        in_size = prod(x_shape[1:])
        n_hidden = 256
        t = t.reshape((t.shape[0], -1))
        x = x.reshape((x.shape[0], -1))  # flatten
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=-1)
        t = nn.Dense(n_hidden)(t)
        t = nn.relu(t)
        t = nn.Dense(in_size)(t)
        t = nn.relu(t)
        x = jnp.concatenate([x, t], axis=-1)
        x = nn.Dense(in_size)(x)
        return x.reshape(x_shape)

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)  # score_t * std_t
