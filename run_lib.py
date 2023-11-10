"""TMPD runlib."""
import os
import flax
# import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
from jax import lax
import jax.scipy as jsp
import numpy as np
import tensorflow as tf
import logging
from flax.training import checkpoints
# TODO: Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
# import utils
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

    q_samples, num_function_evaluations = sampler(sample_rng)
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
  train_ds, eval_ds, _ = datasets.get_dataset(num_devices,
                                              config,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)
                        
  eval_iter = iter(eval_ds)
  batch = next(eval_iter)
  batch = next(eval_iter)
  # for i, batch in enumerate(eval_iter):
  #   for i in batch: print(i)
  # print(batch['image'].shape, "batch image shape")
  # print(batch['label'].shape, "batch label shape")
  eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)  # pylint: disable=protected-access
  # print(eval_batch['image'].shape)
  # print(eval_batch['label'].shape)

  plot_samples(
    inverse_scaler(eval_batch['image'][0]),
    image_size=config.data.image_size,
    num_channels=config.data.num_channels,
    fname=eval_folder + "/_{}_data_{}".format(config.data.dataset, config.solver.outer_solver))

  return eval_batch['image'][0, 0]


def get_inpainting_observation(rng, x, config, mask_name='square'):
  " mask_name in ['square', 'half', 'inverse_square', 'lorem3'"
  mask, num_obs = get_mask(config.data.image_size, mask_name)
  y = x + jax.random.normal(rng, x.shape) * jnp.sqrt(config.sampling.noise_std)
  y = y * mask
  return y, mask, num_obs


def get_superresolution_observation(rng, x, config, shape, method='square'):
  y = jax.image.resize(x, shape, method)
  y = y + jax.random.normal(rng, y.shape) * jnp.sqrt(config.sampling.noise_std)
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
  # y = y + jax.random.normal(rng, y.shape) * jnp.sqrt(config.sampling.noise_std)
  num_obs = jnp.size(y)
  return y, num_obs


def image_grid(x, image_size, num_channels):
    img = x.reshape(-1, image_size, image_size, num_channels)
    w = int(np.sqrt(img.shape[0]))
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

  num_examples = 1
  xs = []
  ys = []
  np.savez(eval_folder + "/{}_{}_eval_{}.npz".format(
    config.sampling.noise_std, config.data.dataset, config.solver.outer_solver),
    noise_std=config.sampling.noise_std)
  for i in range(num_examples):
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
          y, num_obs, H, observation_map, adjoint_observation_map, stack_samples=False)
      else:
        sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
          y, num_obs, H, observation_map, adjoint_observation_map, stack_samples=False)

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

  # obs_shape = (config.data.image_size//2, config.data.image_size//2, config.data.num_channels)
  # method = 'nearest'
  obs_shape = (config.data.image_size//16, config.data.image_size//16, config.data.num_channels)
  method = 'bicubic'

  num_examples = 10
  xs = []
  ys = []
  np.savez(eval_folder + "/{}_{}_eval_{}.npz".format(
    config.sampling.noise_std, config.data.dataset, config.solver.outer_solver),
    noise_std=config.sampling.noise_std)
  for i in range(num_examples):
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
          y, num_obs, H, observation_map, adjoint_observation_map, stack_samples=False)
      else:
        sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
          y, num_obs, H, observation_map, adjoint_observation_map, stack_samples=False)

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


def inpainting(config, workdir, eval_folder="eval"):
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

  num_examples = 10
  xs = []
  ys = []
  np.savez(eval_folder + "/{}_{}_eval_{}.npz".format(
    config.sampling.noise_std, config.data.dataset, config.solver.outer_solver),
    noise_std=config.sampling.noise_std)
  for i in range(num_examples):
    # x = get_eval_sample(scaler, inverse_scaler, config, eval_folder)
    x = get_prior_sample(rng, score_fn, epsilon_fn, sde, inverse_scaler, sampling_shape, config, eval_folder)
    x = x.flatten()
    mask_y, mask, num_obs = get_inpainting_observation(rng, x, config, mask_name='square')
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
      observation_map = lambda x: H @ x
      adjoint_observation_map = None
    else:
      y = mask_y
      observation_map = lambda x: mask * x
      adjoint_observation_map = lambda y: y
      H = None

    cs_method = config.sampling.cs_method

    for cs_method in cs_methods:
      config.sampling.cs_method = cs_method
      if cs_method in ddim_methods:
        sampler = get_cs_sampler(config, sde, epsilon_fn, sampling_shape, inverse_scaler,
          y, num_obs, H, observation_map, adjoint_observation_map, stack_samples=False)
      else:
        sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
          y, num_obs, H, observation_map, adjoint_observation_map, stack_samples=False)

      rng, sample_rng = jax.random.split(rng, 2)
      if config.eval.pmap:
        # sampler = jax.pmap(sampler, axis_name='batch')
        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
      else:
        rng, sample_rng = jax.random.split(rng, 2)

      q_samples, nfe = sampler(sample_rng)
      q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])
      print(q_samples, "\n{}".format(config.sampling.cs_method))
      plot_samples(
        q_samples,
        image_size=config.data.image_size,
        num_channels=config.data.num_channels,
        fname=eval_folder + "/{}_{}_{}_{}".format(config.data.dataset, config.sampling.noise_std, config.sampling.cs_method.lower(), i))


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
  # scaler = datasets.get_data_scaler(config)
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

  rsde = sde.reverse(score_fn)
  drift, diffusion = rsde.sde(x_0, t_0)

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
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 1)

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              additional_dim=1,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)
  print(train_ds)
  print(eval_ds)
  assert 0

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, score_model,
                                   train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous, likelihood_weighting=likelihood_weighting)
    # Pmap (and jit-compile) multiple evaluation steps together for faster execution
    p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step), axis_name='batch', donate_argnums=1)

  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                      additional_dim=None,
                                                      uniform_dequantization=True, evaluation=True)
  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = 5
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, score_model, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size // jax.local_device_count(),
                      config.data.image_size, config.data.image_size,
                      config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = jax.random.fold_in(rng, jax.host_id())

  # A data class for storing intermediate results to resume evaluation after pre-emption
  @flax.struct.dataclass
  class EvalMeta:
    ckpt_id: int
    sampling_round_id: int
    bpd_round_id: int
    rng: Any

  # Add one additional round to get the exact number of samples as required.
  num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
  num_bpd_rounds = len(ds_bpd) * bpd_num_repeats

  # Restore evaluation after pre-emption
  eval_meta = EvalMeta(ckpt_id=config.eval.begin_ckpt, sampling_round_id=-1, bpd_round_id=-1, rng=rng)
  eval_meta = checkpoints.restore_checkpoint(
    eval_dir, eval_meta, step=None, prefix=f"meta_{jax.host_id()}_")

  if eval_meta.bpd_round_id < num_bpd_rounds - 1:
    begin_ckpt = eval_meta.ckpt_id
    begin_bpd_round = eval_meta.bpd_round_id + 1
    begin_sampling_round = 0

  elif eval_meta.sampling_round_id < num_sampling_rounds - 1:
    begin_ckpt = eval_meta.ckpt_id
    begin_bpd_round = num_bpd_rounds
    begin_sampling_round = eval_meta.sampling_round_id + 1

  else:
    begin_ckpt = eval_meta.ckpt_id + 1
    begin_bpd_round = 0
    begin_sampling_round = 0

  rng = eval_meta.rng

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))
    while not tf.io.gfile.exists(ckpt_filename):
      if not waiting_message_printed and jax.host_id() == 0:
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
      time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    try:
      state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
    except:
      time.sleep(60)
      try:
        state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
      except:
        time.sleep(120)
        state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)

    # Replicate the training state for executing on multiple devices
    pstate = flax.jax_utils.replicate(state)
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in enumerate(eval_iter):
        eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)  # pylint: disable=protected-access
        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        next_rng = jnp.asarray(next_rng)
        (_, _), p_eval_loss = p_eval_step((next_rng, pstate), eval_batch)
        eval_loss = flax.jax_utils.unreplicate(p_eval_loss)
        all_losses.extend(eval_loss)
        if (i + 1) % 1000 == 0 and jax.host_id() == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = jnp.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      begin_repeat_id = begin_bpd_round // len(ds_bpd)
      begin_batch_id = begin_bpd_round % len(ds_bpd)
      # Repeat multiple times to reduce variance when needed
      for repeat in range(begin_repeat_id, bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for _ in range(begin_batch_id):
          next(bpd_iter)
        for batch_id in range(begin_batch_id, len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)
          rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
          step_rng = jnp.asarray(step_rng)
          bpd = likelihood_fn(step_rng, pstate, eval_batch['image'])[0]
          bpd = bpd.reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, jnp.mean(jnp.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())

          eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=bpd_round_id, rng=rng)
          # Save intermediate states to resume evaluation after pre-emption
          checkpoints.save_checkpoint(
            eval_dir,
            eval_meta,
            step=ckpt * (num_sampling_rounds + num_bpd_rounds) + bpd_round_id,
            keep=1,
            prefix=f"meta_{jax.host_id()}_")
    else:
      # Skip likelihood computation and save intermediate states for pre-emption
      eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=num_bpd_rounds - 1)
      checkpoints.save_checkpoint(
        eval_dir,
        eval_meta,
        step=ckpt * (num_sampling_rounds + num_bpd_rounds) + num_bpd_rounds - 1,
        keep=1,
        prefix=f"meta_{jax.host_id()}_")

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      state = jax.device_put(state)
      # Run sample generation for multiple rounds to create enough samples
      # Designed to be pre-emption safe. Automatically resumes when interrupted
      for r in range(begin_sampling_round, num_sampling_rounds):
        if jax.host_id() == 0:
          logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(
          eval_dir, f"ckpt_{ckpt}_host_{jax.host_id()}")
        tf.io.gfile.makedirs(this_sample_dir)

        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
        samples, n = sampling_fn(sample_rng, pstate)
        samples = np.clip(samples * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
          fout.write(io_buffer.getvalue())

        # Update the intermediate evaluation state
        eval_meta = eval_meta.replace(ckpt_id=ckpt, sampling_round_id=r, rng=rng)
        # Save an intermediate checkpoint directly if not the last round.
        # Otherwise save eval_meta after computing the Inception scores and FIDs
        if r < num_sampling_rounds - 1:
          checkpoints.save_checkpoint(
            eval_dir,
            eval_meta,
            step=ckpt * (num_sampling_rounds + num_bpd_rounds) + r + num_bpd_rounds,
            keep=1,
            prefix=f"meta_{jax.host_id()}_")

      # Compute inception scores, FIDs and KIDs.
      if jax.host_id() == 0:
        # Load all statistics that have been previously computed and saved for each host
        all_logits = []
        all_pools = []
        for host in range(jax.host_count()):
          this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}_host_{host}")

          stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
          wait_message = False
          while len(stats) < num_sampling_rounds:
            if not wait_message:
              logging.warning("Waiting for statistics on host %d" % (host,))
              wait_message = True
            stats = tf.io.gfile.glob(
              os.path.join(this_sample_dir, "statistics_*.npz"))
            time.sleep(30)

          for stat_file in stats:
            with tf.io.gfile.GFile(stat_file, "rb") as fin:
              stat = np.load(fin)
              if not inceptionv3:
                all_logits.append(stat["logits"])
              all_pools.append(stat["pool_3"])

        if not inceptionv3:
          all_logits = np.concatenate(
            all_logits, axis=0)[:config.eval.num_samples]
        all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

        # Load pre-computed dataset statistics.
        data_stats = evaluation.load_dataset_stats(config)
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

        logging.info(
          "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
            ckpt, inception_score, fid, kid))

        with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                               "wb") as f:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
          f.write(io_buffer.getvalue())
      else:
        # For host_id() != 0.
        # Use file existence to emulate synchronization across hosts
        while not tf.io.gfile.exists(os.path.join(eval_dir, f"report_{ckpt}.npz")):
          time.sleep(1.)

      # Save eval_meta after computing IS/KID/FID to mark the end of evaluation for this checkpoint
      checkpoints.save_checkpoint(
        eval_dir,
        eval_meta,
        step=ckpt * (num_sampling_rounds + num_bpd_rounds) + r + num_bpd_rounds,
        keep=1,
        prefix=f"meta_{jax.host_id()}_")

    else:
      # Skip sampling and save intermediate evaluation states for pre-emption
      eval_meta = eval_meta.replace(ckpt_id=ckpt, sampling_round_id=num_sampling_rounds - 1, rng=rng)
      checkpoints.save_checkpoint(
        eval_dir,
        eval_meta,
        step=ckpt * (num_sampling_rounds + num_bpd_rounds) + num_sampling_rounds - 1 + num_bpd_rounds,
        keep=1,
        prefix=f"meta_{jax.host_id()}_")

    begin_bpd_round = 0
    begin_sampling_round = 0

  # Remove all meta files after finishing evaluation
  meta_files = tf.io.gfile.glob(
    os.path.join(eval_dir, f"meta_{jax.host_id()}_*"))
  for file in meta_files:
    tf.io.gfile.remove(file)
