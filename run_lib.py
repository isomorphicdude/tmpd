"""Evaluation for score-based generative models."""
import os
import time
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
    # sampling_eps = 1e-3  # TODO: add this in numerical solver
    if 'plus' not in config.sampling.cs_method:
      # VP/DDPM Methods with matrix H
      cs_methods = [
                    'Boys2023avjp',
                    'Boys2023b',
                    'Song2023',
                    'Chung2022',  # Unstable
                    'PiGDMVP',
                    'KGDMVP',
                    'KPDDPM'
                    'DPSDDPM']
    else:
      # VP/DDM methods with mask
      cs_methods = [
                    # 'KGDMVPplus',
                    # 'KPDDPMplus',
                    # 'PiGDMVPplus',
                    # 'DPSDDPMplus',
                    'Song2023plus',
                    # 'Boys2023ajacrevplus',
                    # 'Boys2023ajacfwd',
                    # 'Boys2023b',
                    # 'Boys2023bjacfwd',  # OOM, but can try on batch_size=1
                    'Boys2023bvjpplus',
                    'chung2022scalarplus',  # Unstable
                    'chung2022plus',  # Unstable
                    ]
  elif config.training.sde.lower() == 'vesde':
    sde = VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    # sampling_eps = 1e-5  # TODO: add this in numerical solver
    if 'plus' not in config.sampling.cs_method:
      # VE/SMLD Methods with matrix H
      cs_methods = [
                    'Boys2023ajvp',
                    'Boys2023b',
                    'Song2023',
                    # 'Chung2022',  # Unstable
                    'PiGDMVE',
                    'KGDMVE',
                    'KPSMLD'
                    'DPSSMLD']
    else:
      # VE/SMLD methods with mask
      cs_methods = [
                    # 'KGDMVEplus',
                    'KPSMLDplus',
                    # 'PiGDMVEplus',
                    # 'DPSSMLDplus',
                    'Song2023plus',
                    # 'Boys2023ajacrevplus',  # OOM, but can try on batch_size=1
                    'Boys2023b',  # OOM, but can try on batch_size=1
                    # 'Boys2023bjacfwd',  # OOM, but can try on batch_size=1
                    # 'Boys2023bvjpplus',
                    # 'chung2022scalarplus',  # Unstable
                    # 'chung2022plus',  # Unstable
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
    # x = get_eval_sample(scaler, inverse_scaler, config, eval_folder)
    x = get_prior_sample(rng, score_fn, epsilon_fn, sde, inverse_scaler, sampling_shape, config, eval_folder)
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
    assert 0

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

    num_repeats = 1
    for j in range(num_repeats):
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
          fname=eval_folder + "/{}_{}_{}_{}_{}".format(config.data.dataset, config.sampling.noise_std, config.sampling.cs_method.lower(), i, j))
  assert 0


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
    # sampling_eps = 1e-3  # TODO: add this in numerical solver
    if 'plus' not in config.sampling.cs_method:
      # VP/DDPM Methods with matrix H
      cs_methods = [
                    'Boys2023avjp',
                    'Boys2023b',
                    'Song2023',
                    # 'Chung2022',  # Unstable
                    'PiGDMVP',
                    'KGDMVP',
                    'KPDDPM'
                    'DPSDDPM']
    else:
      # VP/DDM methods with mask
      cs_methods = [
                    'KGDMVPplus',
                    'KPDDPMplus',
                    'PiGDMVPplus',
                    'DPSDDPMplus',
                    'Song2023plus',
                    # 'Boys2023ajacrevplus',
                    # 'Boys2023ajacfwd',
                    # 'Boys2023b',
                    # 'Boys2023bjacfwd',  # OOM, but can try on batch_size=1
                    'Boys2023bvjpplus',
                    'chung2022scalarplus',  # Unstable
                    'chung2022plus',  # Unstable
                    ]
  elif config.training.sde.lower() == 'vesde':
    sde = VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    # sampling_eps = 1e-5  # TODO: add this in numerical solver
    if 'plus' not in config.sampling.cs_method:
      # VE/SMLD Methods with matrix H
      cs_methods = [
                    'Boys2023ajvp',
                    'Boys2023b',
                    'Song2023',
                    # 'Chung2022',  # Unstable
                    'PiGDMVE',
                    'KGDMVE',
                    'KPSMLD'
                    'DPSSMLD']
    else:
      # VE/SMLD methods with mask
      cs_methods = [
                    'KGDMVEplus',
                    'KPSMLDplus',
                    'PiGDMVEplus',
                    'DPSSMLDplus',
                    'Song2023plus',
                    # 'Boys2023ajacrevplus',  # OOM, but can try on batch_size=1
                    # 'Boys2023b',  # OOM, but can try on batch_size=1
                    # 'Boys2023bjacfwd',  # OOM, but can try on batch_size=1
                    'Boys2023bvjpplus',
                    'chung2022scalarplus',  # Unstable
                    'chung2022plus',  # Unstable
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

    num_repeats = 3
    for j in range(num_repeats):
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
          fname=eval_folder + "/{}_{}_{}_{}_{}".format(config.data.dataset, config.sampling.noise_std, config.sampling.cs_method.lower(), i, j))
  assert 0


def inverse_problem(config, workdir, eval_folder="eval"):
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
    # sampling_eps = 1e-3  # TODO: add this in numerical solver
    if 'plus' not in config.sampling.cs_method:
      # VP/DDPM Methods with matrix H
      cs_methods = [
                    # 'Boys2023avjp',  # Unstable
                    'Boys2023b',
                    'Boys2023bjacfwd',
                    'Boys2023bvjp',
                    'Song2023',
                    # 'Chung2022',  # Unstable
                    'PiGDMVP',
                    'KGDMVP',
                    'KPDDPM'
                    'DPSDDPM']
    else:
      # VP/DDM methods with mask
      cs_methods = [
                    # 'KGDMVPplus',
                    # 'KPDDPMplus',
                    # 'PiGDMVPplus',
                    # 'DPSDDPMplus',
                    # 'Song2023plus',
                    # 'Boys2023ajacrevplus',
                    # 'Boys2023ajacfwd',
                    # 'Boys2023b',
                    # 'Boys2023bjacfwd',
                    'Boys2023bvjp',
                    # 'Boys2023bvjpplus',
                    # 'chung2022scalarplus',  # Unstable
                    # 'chung2022plus',  # Unstable
                    ]
  elif config.training.sde.lower() == 'vesde':
    sde = VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    # sampling_eps = 1e-5  # TODO: add this in numerical solver
    if 'plus' not in config.sampling.cs_method:
      # VE/SMLD Methods with matrix H
      cs_methods = [
                    'Boys2023ajvp',
                    'Boys2023b',
                    'Song2023',
                    # 'Chung2022',  # Unstable
                    'PiGDMVE',
                    'KGDMVE',
                    'KPSMLD'
                    'DPSSMLD']
    else:
      # VE/SMLD methods with mask
      cs_methods = [
                    # 'KGDMVEplus',
                    'KPSMLDplus',
                    # 'PiGDMVEplus',
                    # 'DPSSMLDplus',
                    # 'Song2023plus',
                    # 'Boys2023ajacrevplus',  # OOM, but can try on batch_size=1
                    # 'Boys2023b',  # OOM, but can try on batch_size=1
                    # 'Boys2023bjacfwd',  # OOM, but can try on batch_size=1
                    # 'Boys2023bvjpplus',
                    # 'chung2022scalarplus',  # Unstable
                    # 'chung2022plus',  # Unstable
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

    num_repeats = 10
    for j in range(num_repeats):
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
          fname=eval_folder + "/{}_{}_{}_{}_{}".format(config.data.dataset, config.sampling.noise_std, config.sampling.cs_method.lower(), i, j))
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

  score_fn = mutils.get_score_fn(
    sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)

  # is score fn a vectorized function? yes

  rsde = sde.reverse(score_fn)
  drift, diffusion = rsde.sde(x_0, t_0)
  print(drift.shape)
  print(diffusion.shape)
  print(drift[0, 0, 0, 0])
  print(drift[-1, -1, -1, -1])
  print(jnp.sum(drift))
  print(diffusion[0])

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

