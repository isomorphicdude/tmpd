"""
Module for performing the forward and backward pass through a deep neural network with a batch of data stored in a dictionary.
Training and evaluation for score-based generative models.
"""
import jax
from jax import jit
import jax.random as random
import jax.numpy as jnp

from diffusionjax.utils import batch_mul, get_loss, get_score, get_sampler
from diffusionjax.models import MLP
import diffusionjax.sde as sde_lib
from diffusionjax.solvers import EulerMaruyama, Annealed

from torch.utils.data import DataLoader

import numpy as np
import optax
from functools import partial
import flax
import flax.jax_utils as flax_utils
from flax.training import checkpoints

from absl import flags

from tqdm import tqdm, trange
import gc
import io
import os
import time
from typing import Any
import datetime
import logging
from collections import defaultdict
import wandb
import orbax


# Google cloud and wandb stuff
# from google.cloud import secretmanager
def read_wandb_secret():
    """ Read weights and biases secrets """
    wandb_secret = os.environ.get("WANDB_SECRET", "")
    if wandb_secret:
        logger.info("Found W&B secret in the environment, attempting to retrieve API key")
        wandb_secret_location = os.environ.get("WANDB_SECRET")
        secrets_client = secretmanager.SecretManagerServiceClient()
        client_response = secrets_client.access_secret_version(
            request={"name": wandb_secret_location}
        )
        api_key = client_response.payload.data.decode("UTF-8")
        os.environ["WANDB_API_KEY"] = api_key
    return wandb_secret
