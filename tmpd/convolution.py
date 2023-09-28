from scipy import misc
import jax.scipy as jsp
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random

fig, ax = plt.subplots(1, 3, figsize=(12, 5))

# Load a sample image; compute mean() to convert from RGB to grayscale.
image = jnp.array(misc.face().mean(-1))
ax[0].imshow(image, cmap='binary_r')
ax[0].set_title('original')

# Create a noisy version by adding random Gaussian noise
key = random.PRNGKey(1701)
noisy_image = image + 50 * random.normal(key, image.shape)
ax[1].imshow(noisy_image, cmap='binary_r')
ax[1].set_title('noisy')

# Smooth the noisy image with a 2D Gaussian smoothing kernel.
x = jnp.linspace(-3, 3, 7)
window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
smooth_image = jsp.signal.convolve(noisy_image, window, mode='same')
ax[2].imshow(smooth_image, cmap='binary_r')
ax[2].set_title('smoothed')

plt.plot()
plt.savefig('test')
plt.close()
