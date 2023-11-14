"""TMPD runlib."""
import gc
import io
import os
import time

# import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
from jax import lax
import jax.scipy as jsp
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
from flax.training import checkpoints
# TODO: Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
from evaluation import get_inception_model, load_dataset_stats, run_inception_distributed
from models import utils as mutils
import datasets
from absl import flags
FLAGS = flags.FLAGS
from diffusionjax.sde import VP, VE
from diffusionjax.solvers import EulerMaruyama
from diffusionjax.utils import get_sampler
from diffusionjax.run_lib import get_solver, get_markov_chain, get_ddim_chain
from tmpd.samplers import get_cs_sampler
from tmpd.inpainting import get_mask
import matplotlib.pyplot as plt


num_devices = 1


unconditional_ddim_methods = ['DDIMVE', 'DDIMVP', 'DDIMVEplus', 'DDIMVPplus']
unconditional_markov_methods = ['DDIM', 'DDIMplus', 'SMLD', 'SMLDplus']
ddim_methods = ['PiGDMVP', 'PiGDMVE', 'PiGDMVPplus', 'PiGDMVEplus',
  'KGDMVP', 'KGDMVE', 'KGDMVPplus', 'KGDMVEplus']
markov_methods = ['KPDDPM', 'KPDDPMplus', 'KPSMLD', 'KPSMLDplus']

from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms


__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img


def get_asset_sample(config):
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  dataset = get_dataset(config.data.dataset.lower(),
                        root='./assets/',
                        transforms=transform)
  loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
  ref_img = next(enumerate(loader))
  ref_img = ref_img.detach().cpu().numpy()[0].transpose(1, 2, 0)
  return ref_img


def get_prior_sample(rng, score_fn, epsilon_fn, sde, inverse_scaler, sampling_shape, config, eval_folder):
  if config.solver.outer_solver in unconditional_ddim_methods:
    outer_solver = get_ddim_chain(config, epsilon_fn)
  elif config.solver.outer_solver in unconditional_markov_methods:
    outer_solver = get_markov_chain(config, score_fn)
  else:
    outer_solver, _ = get_solver(config, sde, score_fn)

  sampler = get_sampler(
    sampling_shape,
    outer_solver, inverse_scaler=None)

  if config.eval.pmap:
    sampler = jax.pmap(sampler, axis_name='batch')
    rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
    sample_rng = jnp.asarray(sample_rng)
  else:
    rng, sample_rng = jax.random.split(rng, 2)

  q_samples, _ = sampler(sample_rng)
  q_images = inverse_scaler(q_samples.copy())
  q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])

  plot_samples(
    inverse_scaler(q_images),
    image_size=config.data.image_size,
    num_channels=config.data.num_channels,
    fname=eval_folder + "/_{}_prior_{}".format(config.data.dataset, config.solver.outer_solver))

  plot_samples(
    inverse_scaler(q_images[0]),
    image_size=config.data.image_size,
    num_channels=config.data.num_channels,
    fname=eval_folder + "/_{}_groundtruth_{}".format(config.data.dataset, config.solver.outer_solver))

  return q_samples[0]


def get_eval_sample(scaler, inverse_scaler, config, eval_folder):
  # Build data pipeline
  _, eval_ds, _ = datasets.get_dataset(num_devices,
                                       config,
                                       uniform_dequantization=config.data.uniform_dequantization,
                                       evaluation=True)
                        
  eval_iter = iter(eval_ds)
  batch = next(eval_iter)
  batch = next(eval_iter)
  # for i, batch in enumerate(eval_iter): #   for i in batch: print(i)
  eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)  # pylint: disable=protected-access

  plot_samples(
    inverse_scaler(eval_batch['image'][0]),
    image_size=config.data.image_size,
    num_channels=config.data.num_channels,
    fname=eval_folder + "/_{}_data_{}".format(config.data.dataset, config.solver.outer_solver))

  return eval_batch['image'][0, 0]


def get_inpainting_observation(rng, x, config, mask_name='square'):
  " mask_name in ['square', 'half', 'inverse_square', 'lorem3'"
  mask, num_obs = get_mask(config.data.image_size, mask_name)
  y = x + jax.random.normal(rng, x.shape) * config.sampling.noise_std
  y = y * mask
  return y, mask, num_obs


def get_superresolution_observation(rng, x, config, shape, method='square'):
  y = jax.image.resize(x, shape, method)
  y = y + jax.random.normal(rng, y.shape) * config.sampling.noise_std
  num_obs = jnp.size(y)
  return y, num_obs


def get_convolve_observation(rng, x, config):
  width = 11
  k = jnp.linspace(-3, 3, width)
  print(x.shape)
  x = jnp.transpose(x,[2,1,0])    # lhs = NCHW image tensor
  print(x.shape)
  x = jnp.expand_dims(x, 0)
  print(x.shape)
  window = jsp.stats.norm.pdf(k) * jsp.stats.norm.pdf(k[:, None])
  kernel = jnp.zeros((width, width, 3, 3), dtype=jnp.float32)
  kernel += window[:, :, jnp.newaxis, jnp.newaxis]
  print(kernel.shape)
  k = jnp.transpose(kernel, [3, 2, 0, 1])  # rhs = OIHW conv kernel tensor
  print(k.shape)
  y = lax.conv(x,  # lhs = NCHW image tensor
               k,  # rhs = OIHW conv kernel tensor
               (1, 1),  # window strides
               'SAME')  # padding mode
  print("out", y.shape)
  y = jnp.transpose(y, [0, 2, 3, 1])
  y = y[0]
  print("out", y.shape)
  # y = y + jax.random.normal(rng, y.shape) * config.sampling.noise_std
  num_obs = jnp.size(y)
  return y, num_obs


def image_grid(x, image_size, num_channels):
    img = x.reshape(-1, image_size, image_size, num_channels)
    w = int(np.sqrt(img.shape[0]))
    img = img[:w**2, :, :, :]
    return img.reshape((w, w, image_size, image_size, num_channels)).transpose((0, 2, 1, 3, 4)).reshape((w * image_size, w * image_size, num_channels))


def plot_samples(x, image_size=32, num_channels=3, fname="samples"):
    img = image_grid(x, image_size, num_channels)
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(fname + '.png', bbox_inches='tight', pad_inches=0.0)
    plt.savefig(fname + '.pdf', bbox_inches='tight', pad_inches=0.0)
    plt.close()


def deblur(config, workdir, eval_folder="eval"):
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.eval.pmap else 1

  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 1)

  # Initialize model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config, num_devices)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  checkpoint_dir = workdir

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = VP(beta_min=config.model.beta_min, beta_max=config.model.beta_max)
    if 'plus' not in config.sampling.cs_method:
      # VP/DDPM Methods with matrix H
      cs_methods = [
                    'TMPD2023avjp',
                    'TMPD2023b',
                    'Song2023',
                    'Chung2022',  
                    'PiGDMVP',
                    'KPDDPM'
                    'DPSDDPM']
    else:
      # VP/DDM methods with mask
      cs_methods = [
                    'KPDDPMplus',
                    'PiGDMVPplus',
                    'DPSDDPMplus',
                    'Song2023plus',
                    'TMPD2023bvjpplus',
                    'chung2022scalarplus',  
                    'chung2022plus',  
                    ]
  elif config.training.sde.lower() == 'vesde':
    sde = VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    if 'plus' not in config.sampling.cs_method:
      # VE/SMLD Methods with matrix H
      cs_methods = [
                    'TMPD2023ajvp',
                    'TMPD2023b',
                    'Song2023',
                    'Chung2022',  
                    'PiGDMVE',
                    'KPSMLD'
                    'DPSSMLD']
    else:
      # VE/SMLD methods with mask
      cs_methods = [
                    'KPSMLDplus',
                    'PiGDMVEplus',
                    'DPSSMLDplus',
                    'Song2023plus',
                    'TMPD2023bvjpplus',
                    'chung2022scalarplus',  
                    'chung2022plus',  
                    ]
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = jax.random.fold_in(rng, jax.host_id())

  ckpt = config.eval.begin_ckpt

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Get model state from checkpoint file
  ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))
  if not tf.io.gfile.exists(ckpt_filename):
    raise FileNotFoundError("{} does not exist".format(ckpt_filename))
  state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)

  epsilon_fn = mutils.get_epsilon_fn(
    sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)
  score_fn = mutils.get_score_fn(
    sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)

  batch_size = config.eval.batch_size
  print("\nbatch_size={}".format(batch_size))
  sampling_shape = (
    config.eval.batch_size//num_devices,
    config.data.image_size, config.data.image_size, config.data.num_channels)
  print("sampling shape", sampling_shape)

  obs_shape = (config.data.image_size//2, config.data.image_size//2, config.data.num_channels)
  method = 'nearest'

  num_sampling_rounds = 1
  xs = []
  ys = []
  np.savez(eval_folder + "/{}_{}_eval_{}.npz".format(
    config.sampling.noise_std, config.data.dataset, config.solver.outer_solver),
    noise_std=config.sampling.noise_std)
  for i in range(num_sampling_rounds):
    x = get_eval_sample(scaler, inverse_scaler, config, eval_folder)
    # x = get_prior_sample(rng, score_fn, epsilon_fn, sde, inverse_scaler, sampling_shape, config, eval_folder)
    y, num_obs = get_convolve_observation(
        rng, x, config)
    plot_samples(
      inverse_scaler(x.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_ground_{}_{}".format(
        config.data.dataset, config.solver.outer_solver, i))
    plot_samples(
      inverse_scaler(y.copy()),
      image_size=obs_shape[1],
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_observed_{}_{}".format(
        config.data.dataset, config.solver.outer_solver, i))

    xs.append(x)
    ys.append(y)
    np.savez(eval_folder + "/{}_{}_eval_{}.npz".format(
      config.sampling.noise_std, config.data.dataset, config.solver.outer_solver),
      xs=jnp.array(xs), ys=jnp.array(ys))

    def observation_map(x):
      print(x.shape)
      x = x.reshape(sampling_shape[1:])
      print(x.shape)
      y = jax.image.resize(x, obs_shape, method)
      print(y.shape)
      assert 0
      return y.flatten()

    def adjoint_observation_map(x):
      y = y.reshape(obs_shape)
      x = jax.image.resize(y, sampling_shape[1:], method)
      return x.flatten()

    H = None
    y = y.flatten()

    cs_method = config.sampling.cs_method

    for cs_method in cs_methods:
      config.sampling.cs_method = cs_method
      if cs_method in ddim_methods:
        sampler = get_cs_sampler(config, sde, epsilon_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=False)
      else:
        sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=False)

      rng, sample_rng = jax.random.split(rng, 2)
      if config.eval.pmap:
        # sampler = jax.pmap(sampler, axis_name='batch')
        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
      else:
        rng, sample_rng = jax.random.split(rng, 2)

      q_samples, nfe = sampler(sample_rng)
      q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])
      print(q_samples, "\nconfig.sampling.cs_method")
      plot_samples(
        q_samples,
        image_size=config.data.image_size,
        num_channels=config.data.num_channels,
        fname=eval_folder + "/{}_{}_{}_{}".format(config.data.dataset, config.sampling.noise_std, config.sampling.cs_method.lower(), i))


def super_resolution(config, workdir, eval_folder="eval"):
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.eval.pmap else 1

  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 1)

  # Initialize model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config, num_devices)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  checkpoint_dir = workdir

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = VP(beta_min=config.model.beta_min, beta_max=config.model.beta_max)
    if 'plus' not in config.sampling.cs_method:
      # VP/DDPM Methods with matrix H
      cs_methods = [
                    'TMPD2023avjp',
                    'TMPD2023b',
                    'Song2023',
                    'Chung2022',  
                    'PiGDMVP',
                    'KPDDPM'
                    'DPSDDPM']
    else:
      # VP/DDM methods with mask
      cs_methods = [
                    'KPDDPMplus',
                    'PiGDMVPplus',
                    'DPSDDPMplus',
                    'Song2023plus',
                    'TMPD2023bvjpplus',
                    'chung2022scalarplus',  
                    'chung2022plus',  
                    ]
  elif config.training.sde.lower() == 'vesde':
    sde = VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    if 'plus' not in config.sampling.cs_method:
      # VE/SMLD Methods with matrix H
      cs_methods = [
                    'TMPD2023ajvp',
                    'TMPD2023b',
                    'Song2023',
                    'Chung2022',  
                    'PiGDMVE',
                    'KPSMLD'
                    'DPSSMLD']
    else:
      # VE/SMLD methods with mask
      cs_methods = [
                    'KPSMLDplus',
                    'PiGDMVEplus',
                    'DPSSMLDplus',
                    'Song2023plus',
                    'TMPD2023bvjpplus',
                    'chung2022scalarplus',  
                    'chung2022plus',  
                    'kgdmveplus',
                    'tmpd2023avjp',
                    'tmpd2023ajvp'
                    'tmpd2023ajacrev',
                    ]
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = jax.random.fold_in(rng, jax.host_id())

  ckpt = config.eval.begin_ckpt

  # Create data normalizer and its inverse
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Get model state from checkpoint file
  ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))
  if not tf.io.gfile.exists(ckpt_filename):
    raise FileNotFoundError("{} does not exist".format(ckpt_filename))
  state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)

  epsilon_fn = mutils.get_epsilon_fn(
    sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)
  score_fn = mutils.get_score_fn(
    sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)

  batch_size = config.eval.batch_size
  print("\nbatch_size={}".format(batch_size))
  sampling_shape = (
    config.eval.batch_size//num_devices,
    config.data.image_size, config.data.image_size, config.data.num_channels)
  print("sampling shape", sampling_shape)

  # obs_shape = (config.data.image_size//2, config.data.image_size//2, config.data.num_channels)
  # method = 'nearest'
  obs_shape = (config.data.image_size//16, config.data.image_size//16, config.data.num_channels)
  method = 'bicubic'

  num_sampling_rounds = 10
  xs = []
  ys = []
  np.savez(eval_folder + "/{}_{}_eval_{}.npz".format(
    config.sampling.noise_std, config.data.dataset, config.solver.outer_solver),
    noise_std=config.sampling.noise_std)
  for i in range(num_sampling_rounds):
    # x = get_eval_sample(scaler, inverse_scaler, config, eval_folder)
    x = get_prior_sample(rng, score_fn, epsilon_fn, sde, inverse_scaler, sampling_shape, config, eval_folder)
    y, num_obs = get_superresolution_observation(
        rng, x, config, obs_shape, method=method)
    plot_samples(
      inverse_scaler(x.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_ground_{}_{}".format(
        config.data.dataset, config.solver.outer_solver, i))
    plot_samples(
      inverse_scaler(y.copy()),
      image_size=obs_shape[1],
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_observed_{}_{}".format(
        config.data.dataset, config.solver.outer_solver, i))

    xs.append(x)
    ys.append(y)
    np.savez(eval_folder + "/{}_{}_eval_{}.npz".format(
      config.sampling.noise_std, config.data.dataset, config.solver.outer_solver),
      xs=jnp.array(xs), ys=jnp.array(ys))

    def observation_map(x):
      x = x.reshape(sampling_shape[1:])
      y = jax.image.resize(x, obs_shape, method)
      return y.flatten()

    def adjoint_observation_map(x):
      y = y.reshape(obs_shape)
      x = jax.image.resize(y, sampling_shape[1:], method)
      return x.flatten()

    H = None
    y = y.flatten()

    cs_method = config.sampling.cs_method

    for cs_method in cs_methods:
      config.sampling.cs_method = cs_method
      if cs_method in ddim_methods:
        sampler = get_cs_sampler(config, sde, epsilon_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=False)
      else:
        sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=False)

      rng, sample_rng = jax.random.split(rng, 2)
      if config.eval.pmap:
        # sampler = jax.pmap(sampler, axis_name='batch')
        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
      else:
        rng, sample_rng = jax.random.split(rng, 2)

      q_samples, _ = sampler(sample_rng)
      q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])
      print(q_samples, "\nconfig.sampling.cs_method")
      plot_samples(
        q_samples,
        image_size=config.data.image_size,
        num_channels=config.data.num_channels,
        fname=eval_folder + "/{}_{}_{}_{}".format(config.data.dataset, config.sampling.noise_std, config.sampling.cs_method.lower(), i))


def inpainting(config, workdir, eval_folder="eval"):
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.eval.pmap else 1

  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 1)

  # Initialize model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config, num_devices)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  checkpoint_dir = workdir

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = VP(beta_min=config.model.beta_min, beta_max=config.model.beta_max)
    if 'plus' not in config.sampling.cs_method:
      # VP/DDPM Methods with matrix H
      cs_methods = [
                    'TMPD2023b',
                    'TMPD2023bjacfwd',
                    'TMPD2023bvjp',
                    'Song2023',
                    'Chung2022',  
                    'PiGDMVP',
                    'KPDDPM'
                    'DPSDDPM']
    else:
      # VP/DDM methods with mask
      cs_methods = [
                    'KPDDPMplus',
                    'PiGDMVPplus',
                    'DPSDDPMplus',
                    'Song2023plus',
                    'TMPD2023bvjp',
                    'chung2022plus',  
                    ]
  elif config.training.sde.lower() == 'vesde':
    sde = VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    if 'plus' not in config.sampling.cs_method:
      # VE/SMLD Methods with matrix H
      cs_methods = [
                    # 'KPSMLD',
                    # 'TMPD2023avjp',
                    # 'TMPD2023ajacrev',
                    'TMPD2023ajacfwd',
                    # 'TMPD2023b',
                    # 'Song2023',
                    # 'Chung2022',  
                    # 'PiGDMVE',
                    # 'KGDMVE',
                    # 'KPSMLD'
                    # 'DPSSMLD'
                    ]
    else:
      # VE/SMLD methods with mask
      cs_methods = [
                    # 'KGDMVEplus',
                    # 'KPSMLDplus',
                    # 'PiGDMVEplus',
                    'DPSSMLDplus',
                    # 'Song2023plus',
                    # 'TMPD2023bvjpplus',
                    # 'TMPD2023ajacrevplus',
                    # 'chung2022plus',  
                    # 'chung2022scalarplus',  
                    ]
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Get model state from checkpoint file
  ckpt = config.eval.begin_ckpt
  ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))
  if not tf.io.gfile.exists(ckpt_filename):
    raise FileNotFoundError("{} does not exist".format(ckpt_filename))
  state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)

  epsilon_fn = mutils.get_epsilon_fn(
    sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)
  score_fn = mutils.get_score_fn(
    sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)

  batch_size = config.eval.batch_size
  print("\nbatch_size={}".format(batch_size))
  sampling_shape = (
    config.eval.batch_size//num_devices,
    config.data.image_size, config.data.image_size, config.data.num_channels)
  print("sampling shape", sampling_shape)

  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = jax.random.fold_in(rng, jax.host_id())

  num_sampling_rounds = 10
  xs = []
  ys = []
  np.savez(eval_folder + "/{}_{}_eval_{}.npz".format(
    config.sampling.noise_std, config.data.dataset, config.solver.outer_solver),
    noise_std=config.sampling.noise_std)
  for i in range(num_sampling_rounds):
    x = get_eval_sample(scaler, inverse_scaler, config, eval_folder)
    # x = get_prior_sample(rng, score_fn, epsilon_fn, sde, inverse_scaler, sampling_shape, config, eval_folder)
    # x = get_asset_sample(config)
    x = x.flatten()
    mask_y, mask, num_obs = get_inpainting_observation(rng, x, config, mask_name='half')
    plot_samples(
      inverse_scaler(x.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_ground_{}_{}".format(
        config.data.dataset, config.solver.outer_solver, i))
    plot_samples(
      inverse_scaler(mask_y.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_observed_{}_{}".format(
        config.data.dataset, config.solver.outer_solver, i))
    xs.append(x)
    ys.append(mask_y)
    np.savez(eval_folder + "/{}_{}_eval_{}.npz".format(
      config.sampling.noise_std, config.data.dataset, config.solver.outer_solver),
      xs=jnp.array(xs), ys=jnp.array(ys))

    if 'plus' not in config.sampling.cs_method:
      logging.warning(
        "Using full H matrix H.shape={} which may be too large to fit in memory ".format(
          (num_obs, config.data.image_size**2 * config.data.num_channels)))
      idx_obs = np.nonzero(mask)[0]
      H = jnp.zeros((num_obs, config.data.image_size**2 * config.data.num_channels))
      ogrid = np.arange(num_obs, dtype=int)
      H = H.at[ogrid, idx_obs].set(1.0)
      y = H @ mask_y

      def observation_map(x):
          x = x.flatten()
          return H @ x
      adjoint_observation_map = None
    else:
      y = mask_y
      def observation_map(x):
          x = x.flatten()
          return mask * x
      adjoint_observation_map = None
      H = None

    for cs_method in cs_methods:
      config.sampling.cs_method = cs_method
      if cs_method in ddim_methods:
        sampler = get_cs_sampler(config, sde, epsilon_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=False)
      else:
        sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=False)

      rng, sample_rng = jax.random.split(rng, 2)
      if config.eval.pmap:
        sampler = jax.pmap(sampler, axis_name='batch')
        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
      else:
        rng, sample_rng = jax.random.split(rng, 2)

      time_prev = time.time()
      q_samples, _ = sampler(sample_rng)
      sample_time = time.time() - time_prev
      print("{}: {}s".format(cs_method, sample_time))

      q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])
      # print(q_samples, "\n{}".format(config.sampling.cs_method))
      plot_samples(
        q_samples,
        image_size=config.data.image_size,
        num_channels=config.data.num_channels,
        fname=eval_folder + "/{}_{}_{}_{}".format(config.data.dataset, config.sampling.noise_std, config.sampling.cs_method.lower(), i))
      assert 0


def sample(config,
          workdir,
          eval_folder="eval"):
  """
  Sample trained models using diffusionjax.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 1)

  # Create data normalizer and its inverse
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config, num_devices)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  checkpoint_dir = workdir

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = VP(beta_min=config.model.beta_min, beta_max=config.model.beta_max)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    # sampling_eps = 1e-3
    raise NotImplementedError("The sub-variance-preserving SDE was not implemented.")
  elif config.training.sde.lower() == 'vesde':
    sde = VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  input_shape = (
    num_devices, config.data.image_size, config.data.image_size, config.data.num_channels)
  x_0 = jax.random.normal(rng, input_shape)
  t_0 = 0.999 * jnp.ones((1,))

  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = jax.random.fold_in(rng, jax.host_id())

  ckpt = config.eval.begin_ckpt

  # Wait if the target checkpoint doesn't exist yet
  ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))

  if not tf.io.gfile.exists(ckpt_filename):
    raise FileNotFoundError("{} does not exist".format(ckpt_filename))

  state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)

  # score_fn is vmap'd
  score_fn = mutils.get_score_fn(
    sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)

  # rsde = sde.reverse(score_fn)
  # drift, diffusion = rsde.sde(x_0, t_0)

  sampler = get_sampler(
    (4, config.data.image_size, config.data.image_size, config.data.num_channels),
    EulerMaruyama(sde.reverse(score_fn), num_steps=config.model.num_scales))
  q_samples, num_function_evaluations = sampler(rng)
  print("num_function_evaluations", num_function_evaluations)
  q_samples = inverse_scaler(q_samples)
  print(q_samples.shape)
  plot_samples(
    q_samples,
    image_size=config.data.image_size,
    num_channels=config.data.num_channels,
    fname="{} samples".format(config.data.dataset))


def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.eval.pmap else 1

  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 1)

  # Build data pipeline
  _, eval_ds, _ = datasets.get_dataset(num_devices,
                                       config,
                                       uniform_dequantization=config.data.uniform_dequantization,
                                       evaluation=True)

  eval_iter = iter(eval_ds)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config, num_devices)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  checkpoint_dir = workdir

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = VP(beta_min=config.model.beta_min, beta_max=config.model.beta_max)
    if 'plus' not in config.sampling.cs_method:
      # VP/DDPM Methods with matrix H
      cs_methods = [
                    'TMPD2023b',
                    'TMPD2023bjacfwd',
                    'TMPD2023bvjp',
                    'Song2023',
                    'Chung2022',  
                    'PiGDMVP',
                    'KPDDPM'
                    'DPSDDPM']
    else:
      # VP/DDM methods with mask
      cs_methods = [
                    'KPDDPMplus',
                    'PiGDMVPplus',
                    'DPSDDPMplus',
                    'Song2023plus',
                    'TMPD2023bvjp',
                    'chung2022plus',  
                    ]
  elif config.training.sde.lower() == 'vesde':
    sde = VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    if 'plus' not in config.sampling.cs_method:
      # VE/SMLD Methods with matrix H
      cs_methods = [
                    # 'KPSMLD',
                    # 'TMPD2023avjp',
                    # 'TMPD2023ajacrev',
                    'TMPD2023ajacfwd',
                    # 'TMPD2023b',
                    # 'Song2023',
                    # 'Chung2022',  
                    # 'PiGDMVE',
                    # 'KGDMVE',
                    # 'KPSMLD'
                    # 'DPSSMLD'
                    ]
    else:
      # VE/SMLD methods with mask
      cs_methods = [
                    # 'KPSMLDplus',
                    # 'PiGDMVEplus',
                    'DPSSMLDplus',
                    # 'Song2023plus',
                    # 'TMPD2023bvjpplus',
                    # 'TMPD2023ajacrevplus',
                    # 'chung2022plus',  
                    # 'chung2022scalarplus',  
                    # 'KGDMVEplus',
                    # 'KPSMLDplus',
                    # 'PiGDMVEplus',
                    # 'DPSSMLDplus',
                    # 'Song2023plus',
                    ]
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Get model state from checkpoint file
  ckpt = config.eval.begin_ckpt
  ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))
  if not tf.io.gfile.exists(ckpt_filename):
    raise FileNotFoundError("{} does not exist".format(ckpt_filename))
  state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)

  epsilon_fn = mutils.get_epsilon_fn(
    sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)
  score_fn = mutils.get_score_fn(
    sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)

  batch_size = config.eval.batch_size
  print("\nbatch_size={}".format(batch_size))
  sampling_shape = (
    config.eval.batch_size//num_devices,
    config.data.image_size, config.data.image_size, config.data.num_channels)
  print("sampling shape", sampling_shape)

  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = jax.random.fold_in(rng, jax.host_id())

  num_sampling_rounds = 0
  xs = []
  ys = []
  np.savez(eval_folder + "/{}_{}_eval_{}.npz".format(
    config.sampling.noise_std, config.data.dataset, config.solver.outer_solver),
    noise_std=config.sampling.noise_std)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = get_inception_model(inceptionv3=inceptionv3)
  for i in range(num_sampling_rounds):
    batch = next(eval_iter)
    eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)  # pylint: disable=protected-access
    x = eval_batch['image'][0, 0]
    x = x.flatten()
    mask_y, mask, num_obs = get_inpainting_observation(rng, x, config, mask_name='half')
    plot_samples(
      inverse_scaler(x.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_ground_{}_{}".format(
        config.data.dataset, config.solver.outer_solver, i))
    plot_samples(
      inverse_scaler(mask_y.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_observed_{}_{}".format(
        config.data.dataset, config.solver.outer_solver, i))
    xs.append(x)
    ys.append(mask_y)
    np.savez(eval_folder + "/{}_{}_eval_{}.npz".format(
      config.sampling.noise_std, config.data.dataset, config.solver.outer_solver),
      xs=jnp.array(xs), ys=jnp.array(ys))

    if 'plus' not in config.sampling.cs_method:
      logging.warning(
        "Using full H matrix H.shape={} which may be too large to fit in memory ".format(
          (num_obs, config.data.image_size**2 * config.data.num_channels)))
      idx_obs = np.nonzero(mask)[0]
      H = jnp.zeros((num_obs, config.data.image_size**2 * config.data.num_channels))
      ogrid = np.arange(num_obs, dtype=int)
      H = H.at[ogrid, idx_obs].set(1.0)
      y = H @ mask_y

      def observation_map(x):
          x = x.flatten()
          return H @ x
      adjoint_observation_map = None
    else:
      y = mask_y
      def observation_map(x):
          x = x.flatten()
          return mask * x
      adjoint_observation_map = None
      H = None

    # TODO: SSIM, MSE (distance)
    # Generate samples and compute IS/FID/KID when enabled
    for cs_method in cs_methods:
      config.sampling.cs_method = cs_method
      if cs_method in ddim_methods:
        sampler = get_cs_sampler(config, sde, epsilon_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=False)
      else:
        sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=False)

      rng, sample_rng = jax.random.split(rng, 2)
      if config.eval.pmap:
        sampler = jax.pmap(sampler, axis_name='batch')
        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
      else:
        rng, sample_rng = jax.random.split(rng, 2)

      time_prev = time.time()
      q_samples, _ = sampler(sample_rng)
      sample_time = time.time() - time_prev
      print("{}: {}s".format(cs_method, sample_time))

      q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])
      # print(q_samples, "\n{}".format(config.sampling.cs_method))
      plot_samples(
        q_samples,
        image_size=config.data.image_size,
        num_channels=config.data.num_channels,
        fname=eval_folder + "/{}_{}_{}_{}".format(config.data.dataset, config.sampling.noise_std, config.sampling.cs_method.lower(), i))

      samples = np.clip(q_samples * 255., 0, 255).astype(np.uint8)

      eval_file = "{}_{}_eval_{}".format(
        config.sampling.noise_std, config.data.dataset, config.sampling.cs_method)
      eval_path = eval_folder + eval_file
      np.savez(eval_path + ".npz", xs=q_samples, y=mask_y)

      # Evaluate FID scores
      # Force garbage collection before calling TensorFlow code for Inception network
      gc.collect()
      latents = run_inception_distributed(samples, inception_model, inceptionv3=inceptionv3)
      # Force garbage collection again before returning to JAX code
      gc.collect()
      # Save latent represents of the Inception network to disk
      np.savez_compressed(
        eval_path + "_stats_{}.npz".format(i), pool_3=latents["pool_3"], logits=latents["logits"])

  for cs_method in cs_methods:
    config.sampling.cs_method = cs_method
    eval_file = "{}_{}_eval_{}".format(
      config.sampling.noise_std, config.data.dataset, config.sampling.cs_method)
    eval_path = eval_folder + eval_file
    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved
    all_logits = []
    all_pools = []

    stats = tf.io.gfile.glob(os.path.join(eval_folder, eval_file + "_stats_*.npz"))

    for stat_file in stats:
      with tf.io.gfile.GFile(stat_file, "rb") as fin:
        stat = np.load(fin)
        if not inceptionv3:
          all_logits.append(stat["logits"])
        all_pools.append(stat["pool_3"])

    print(len(all_logits))
    print(len(all_pools))
    if not inceptionv3:
      if len(all_logits) != 1:
        all_logits = np.concatenate(all_logits, axis=0)
      else:
        all_logits = np.array(all_logits)
    if len(all_pools) != 1:
      all_pools = np.concatenate(all_pools, axis=0)
    else:
      all_pools = np.array(all_logits)

    print(all_logits.shape)
    print(all_pools.shape)
    # Load pre-computed dataset statistics.
    data_stats = load_dataset_stats(config)
    print(data_stats)
    print(data_stats.files)
    assert 0
    data_pools = data_stats["pool_3"]

    # Compute FID/KID/IS on all samples together.
    if not inceptionv3:
      inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
    else:
      inception_score = -1

    fid = tfgan.eval.frechet_classifier_distance_from_activations(
      data_pools, all_pools)
    # Hack to get tfgan KID work for eager execution.
    tf_data_pools = tf.convert_to_tensor(data_pools)
    tf_all_pools = tf.convert_to_tensor(all_pools)
    kid = tfgan.eval.kernel_classifier_distance_from_activations(
      tf_data_pools, tf_all_pools).numpy()
    del tf_data_pools, tf_all_pools

    logging.info("cs_method-{} --- inception_score: {%.6e}, FID: {%.6e}, KID: {%.6e}".format(
        cs_method, inception_score, fid, kid))

    np.savez_compressed(
      eval_file + "_reports.npz", IS=inception_score, fid=fid, kid=kid)
