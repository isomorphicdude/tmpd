"""Inpainting utilities."""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import jax


def display_sample(sample):
    image_processed = sample.cpu().permute(1, 2, 0)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = Image.fromarray(image_processed)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image_pil)
    #.title(f"Image at step {i}")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def save_jpg(sample, path, format='PDF'):
    if sample.shape[0] == 3:
        image_processed = sample.cpu().permute(1, 2, 0)
    else:
        image_processed = sample[0].cpu()
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = Image.fromarray(image_processed, mode='L' if len(sample.shape)==2 else None)
    image_pil.save(fp=path, format=format)
    return None


def get_mask(image_size, mask_name='square'):
    mask_file = Path(__file__).parent / Path(f'masks/{mask_name}.npy')
    mask = np.load(mask_file)
    mask_size = mask.shape[0]
    assert mask_size % image_size == 0
    sub_sample = mask_size // image_size
    mask = mask[::sub_sample, ::sub_sample]
    num_obs = np.count_nonzero(mask)
    mask = np.tile(mask, (3, 1, 1))
    mask = mask.transpose(1, 2, 0)
    mask = mask.flatten()
    num_obs *= 3
    return mask, num_obs


def get_random_mask(rng, image_size, num_channels):
    num_obs = int(image_size)
    idx_obs = jax.random.choice(
    rng, image_size**2, shape=(num_obs,), replace=False)
    mask = np.zeros((image_size**2,), dtype=int)
    mask = mask.at[idx_obs].set(1)
    mask = mask.reshape((image_size, image_size))
    mask = np.tile(mask, (num_channels, 1, 1)).transpose(1, 2, 0)
    num_obs = num_obs * 3  # because of tile
    mask = mask.flatten()
    return mask, num_obs
