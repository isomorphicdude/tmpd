"""Evaluation for score-based generative models."""
import os
import time
import flax
# import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
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
from grfjax.samplers import get_cs_sampler
from grfjax.inpainting import get_mask
from grfjax.super_resolution import Resizer
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
      q_images,
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_prior_{}".format(config.data.dataset, config.solver.outer_solver))

    plot_samples(
      q_images[0],
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_groundtruth_{}".format(config.data.dataset, config.solver.outer_solver))

    x = q_samples[0].flatten()
    return x


def get_eval_sample(rng, scaler, inverse_scaler, config, eval_folder):
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
  print(batch['image'].shape, "batch image shape")
  print(batch['label'].shape, "batch label shape")
  eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)  # pylint: disable=protected-access
  # print(eval_batch['image'].shape)
  # print(eval_batch['label'].shape)

  plot_samples(
    eval_batch['image'][0],
    image_size=config.data.image_size,
    num_channels=config.data.num_channels,
    fname=eval_folder + "/_{}_data_{}".format(config.data.dataset, config.solver.outer_solver))

  x = eval_batch['image'][0, 0].flatten()
  return x


def get_observation(rng, x, config, mask_name='square'):
  " mask_name in ['square', 'half', 'inverse_square', 'lorem3'"
  mask, num_obs = get_mask(config.data.image_size, mask_name)
  y = x + jax.random.normal(rng, x.shape) * jnp.sqrt(config.sampling.noise_std)  # noise
  y = y * mask
  return y, mask, num_obs


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


def super_resolution(config, workdir, eval_folder="eval"):
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.eval.pmap else 1

  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 1)

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

  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = jax.random.fold_in(rng, jax.host_id())

  begin_ckpt = config.eval.begin_ckpt

  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  print("begin checkpoint: {}".format(begin_ckpt))
  print("end checkpoint: {}".format(config.eval.end_ckpt))

  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))

    if not tf.io.gfile.exists(ckpt_filename):
      raise FileNotFoundError("{} does not exist".format(ckpt_filename))

    state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)

    epsilon_fn = mutils.get_epsilon_fn(
      sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)
    score_fn = mutils.get_score_fn(
      sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)
    if config.solver.outer_solver in unconditional_ddim_methods:
      outer_solver = get_ddim_chain(config, epsilon_fn)
    elif config.solver.outer_solver in unconditional_markov_methods:
      outer_solver = get_markov_chain(config, score_fn)
    else:
      # rsde = sde.reverse(score_fn)
      # outer_solver = EulerMaruyama(rsde, num_steps=config.model.num_scales)
      outer_solver, _ = get_solver(config, sde, score_fn)
 
    batch_size = config.eval.batch_size
    print("\nbatch_size={}".format(batch_size))
    sampling_shape = (
      config.eval.batch_size//num_devices,
      config.data.image_size, config.data.image_size, config.data.num_channels)
    print("sampling shape", sampling_shape)

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
    print("num_function_evaluations", num_function_evaluations)
    print("sampling shape", q_samples.shape)
    print("before inverse scaler ", q_samples)
    q_images = inverse_scaler(q_samples.copy())
    print("after inverse scaler ", q_images)
    q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])
  
    plot_samples(
      q_images,
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_prior_{}".format(config.data.dataset, config.solver.outer_solver))

    plot_samples(
      q_images[0],
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_ground_{}".format(config.data.dataset, config.solver.outer_solver))
    x = q_samples[0]

    scale_factor = 2.
    observation_map = Resizer(sampling_shape[1:], 1. / scale_factor)
    print("x shape", x.shape, "sampling_shape", sampling_shape[1:])
    y = observation_map(x)
    print("y shape", y.shape)
    plot_samples(
      y,
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_observed_{}".format(
        config.data.dataset, config.solver.outer_solver))


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
  elif config.training.sde.lower() == 'vesde':
    sde = VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    # sampling_eps = 1e-5  # TODO: add this in numerical solver
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

  num_examples = 4
  xs = []
  ys = []
  np.savez(eval_folder + "/{}_{}_eval_{}.npz".format(
    config.sampling.noise_std, config.data.dataset, config.solver.outer_solver),
    noise_std=config.sampling.noise_std)
  for i in range(num_examples):
    x = get_eval_sample(rng, scaler, inverse_scaler, config, eval_folder)
    y, mask, num_obs = get_observation(rng, x, config, mask_name='square')
    plot_samples(
      x,
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_ground_{}_{}".format(
        config.data.dataset, config.solver.outer_solver, i))
    plot_samples(
      y,
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_observed_{}_{}".format(
        config.data.dataset, config.solver.outer_solver, i))
    xs.append(x)
    ys.append(y)
    np.savez(eval_folder + "/{}_{}_eval_{}.npz".format(
      config.sampling.noise_std, config.data.dataset, config.solver.outer_solver),
      xs=jnp.array(xs), ys=jnp.array(ys))

    H = None
    observation_map = None
    adjoint_observation_map = None
    # num_obs = int(config.data.image_size**2 / 16)
    if 'plus' not in config.sampling.cs_method:
      logging.warning(
        "Using full H matrix H.shape={} which may be too large to fit in memory ".format(
          (num_obs, config.data.image_size**2 * config.data.num_channels)))
      idx_obs = np.nonzero(mask)[0]
      H = jnp.zeros((num_obs, config.data.image_size**2 * config.data.num_channels))
      ogrid = np.arange(num_obs, dtype=int)
      H = H.at[ogrid, idx_obs].set(1.0)
      def observation_map(x, t):
          return H @ x

      def adjoint_observation_map(y, t):
          return H.T @ y

      y = H @ y
      # can get indices from a flat mask
      mask = None

    cs_method = config.sampling.cs_method

    # Methods with matrix H
    # cs_methods = ['Boys2023ajvp',
    #               'Boys2023avjp',
    #               'Boys2023ajac',
    #               'Boys2023b',
    #               'Song2023',
    #               'Chung2022',
    #               'ProjectionKalmanFilter',
    #               'PiGDMVE',
    #               'KGDMVE',
    #               'KPSMLD'
    #               'DPSSMLD']

    # Methods with mask
    cs_methods = [
                  'KGDMVEplus',
                  'KPSMLDplus',
                  'PiGDMVEplus',
                  'DPSSMLDplus',
                  'Song2023plus',
                  'Boys2023bvjpplus',
                  'Boys2023bjvpplus',
                  'Boys2023cplus',
                  # 'chung2022scalarplus',
                  # 'chung2022plus',
                  ]

    num_repeats = 1
    for j in range(num_repeats):
      for cs_method in cs_methods:
        config.sampling.cs_method = cs_method
        if cs_method in ddim_methods:
          sampler = get_cs_sampler(config, sde, epsilon_fn, sampling_shape, inverse_scaler,
            y, H, mask, observation_map, adjoint_observation_map, stack_samples=False)
        elif cs_method in markov_methods:
          sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
            y, H, mask, observation_map, adjoint_observation_map, stack_samples=False)
        else:
          sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
            y, H, mask, observation_map, adjoint_observation_map, stack_samples=False)

        rng, sample_rng = jax.random.split(rng, 2)
        if config.eval.pmap:
          # sampler = jax.pmap(sampler, axis_name='batch')
          rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
          sample_rng = jnp.asarray(sample_rng)
        else:
          rng, sample_rng = jax.random.split(rng, 2)

        q_samples, nfe = sampler(sample_rng)
        q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])
        print("q_samples ",q_samples)
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

  begin_ckpt = config.eval.begin_ckpt

  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  print("begin checkpoint: {}".format(begin_ckpt))
  print("end checkpoint: {}".format(config.eval.end_ckpt))

  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
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

