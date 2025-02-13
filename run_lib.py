"""TMPD runlib."""
import gc
import os
import time
import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
from flax.training import checkpoints
# NOTE: Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
from evaluation import get_inception_model, load_dataset_stats, run_inception_distributed
from models import utils as mutils
import datasets
from absl import flags
FLAGS = flags.FLAGS
from diffusionjax.sde import VP, VE
from diffusionjax.solvers import EulerMaruyama
from diffusionjax.utils import get_sampler, batch_matmul_A
from diffusionjax.run_lib import get_solver, get_markov_chain, get_ddim_chain
from tmpd.samplers import get_cs_sampler
from tmpd.inpainting import get_mask
from tmpd.plot import plot_animation
import matplotlib.pyplot as plt
from tensorflow.image import ssim as tf_ssim
from tensorflow.image import psnr as tf_psnr
from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
import torch
import lpips
import yaml


logging.basicConfig(filename=str(float(time.time())) + ".log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

mse_vmap = vmap(lambda a, b: jnp.mean((a - b)**2))
flatten_vmap = vmap(lambda x: x.flatten())

unconditional_ddim_methods = ['DDIMVE', 'DDIMVP', 'DDIMVEplus', 'DDIMVPplus']
unconditional_markov_methods = ['DDIM', 'DDIMplus', 'SMLD', 'SMLDplus']
ddim_methods = ['PiGDMVP', 'PiGDMVE', 'PiGDMVPplus', 'PiGDMVEplus',
  'KGDMVP', 'KGDMVE', 'KGDMVPplus', 'KGDMVEplus']
markov_methods = ['KPDDPM', 'KPDDPMplus', 'KPSMLD', 'KPSMLDplus']

__DATASET__ = {}


def torch_lpips(loss_fn_vgg, x, samples):
  axes = (0, 3, 1, 2)
  delta = samples.transpose(axes)
  label = x.transpose(axes)
  delta = torch.from_numpy(np.array(delta))
  label = torch.from_numpy(np.array(label))
  delta = delta * 2. - 1.
  label = label * 2. - 1.
  lpips = loss_fn_vgg(delta, label)
  return lpips.detach().cpu().numpy().flatten()


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
  # transform = transforms.Compose([transforms.ToTensor(),
  #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  transform = transforms.ToTensor()
  dataset = get_dataset(config.data.dataset.lower(),
                        root='./assets/',
                        transforms=transform)
  loader = get_dataloader(dataset, batch_size=3, num_workers=0, train=False)
  ref_img = next(iter(loader))
  print(ref_img.shape)
  ref_img = ref_img.detach().cpu().numpy()[2].transpose(1, 2, 0)
  print(np.max(ref_img), np.min(ref_img))
  ref_img = np.tile(ref_img, (config.eval.batch_size, 1, 1, 1))
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
    rng, *sample_rng = random.split(rng, jax.local_device_count() + 1)
    sample_rng = jnp.asarray(sample_rng)
  else:
    rng, sample_rng = random.split(rng, 2)

  q_samples, _ = sampler(sample_rng)
  # q_images = inverse_scaler(q_samples.copy())
  q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])

  return q_samples


def get_eval_sample(scaler, config, num_devices):
  # Build data pipeline
  _, eval_ds, _ = datasets.get_dataset(num_devices,
                                       config,
                                       uniform_dequantization=config.data.uniform_dequantization,
                                       evaluation=True)
  eval_iter = iter(eval_ds)
  batch = next(eval_iter)
  # TODO: can tree_map be used to pmap across observation data?
  eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)  # pylint: disable=protected-access
  return eval_batch['image'][0]


def get_sde(config):
  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = VP(beta_min=config.model.beta_min, beta_max=config.model.beta_max)
    if 'plus' not in config.sampling.cs_method:
      # VP/DDPM Methods with matrix H
      cs_methods = [
                    'KPDDPM',
                    'DPSDDPM',
                    'PiGDMVP',
                    'TMPD2023b',
                    'Chung2022scalar',  
                    'Song2023',
                    ]
    else:
      # VP/DDM methods with mask
      cs_methods = [
                    'KPDDPMplus',
                    'DPSDDPMplus',
                    'PiGDMVPplus',
                    'TMPD2023bvjpplus',
                    'chung2022scalarplus',  
                    'Song2023plus',
                    ]
  elif config.training.sde.lower() == 'vesde':
    sde = VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    if 'plus' not in config.sampling.cs_method:
      # VE/SMLD Methods with matrix H
      cs_methods = [
                    # 'TMPD2023bvjp'
                    'KPSMLDdiag',
                    # 'KPSMLD',
                    # 'DPSSMLD',
                    # 'PiGDMVE',
                    # 'TMPD2023b',
                    # 'Chung2022scalar',
                    # 'Song2023',
                    ]
    else:
      # VE/SMLD methods with mask
      cs_methods = [
                    # 'KPSMLDdiag',
                    # 'TMPD2023bvjpplus',
                    'KPSMLDplus',
                    'DPSSMLDplus',
                    'PiGDMVEplus',
                    'KGDMVEplus',
                    # 'chung2022scalarplus',  
                    # 'chung2022plus',  
                    # 'Song2023plus',
                    ]
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
  return cs_methods, sde


def get_inpainting_observation(rng, x, config, mask_name='square'):
  " mask_name in ['square', 'half', 'inverse_square', 'lorem3'"
  x = flatten_vmap(x)
  mask, num_obs = get_mask(config.data.image_size, mask_name)
  y = x + random.normal(rng, x.shape) * config.sampling.noise_std
  y = y * mask
  return x, y, mask, num_obs


def get_superresolution_observation(rng, x, config, shape, method='square'):
  y = jax.image.resize(x, shape, method)
  y = y + random.normal(rng, y.shape) * config.sampling.noise_std
  num_obs = jnp.size(y)
  y = flatten_vmap(y)
  x = flatten_vmap(x)
  mask = None  # TODO
  return x, y, mask, num_obs


def get_blur_observation(rng, x, config, device=None):
  from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

  def get_blur_model(opt_yml_path):
    with open(opt_yml_path, "r") as f:
      opt = yaml.safe_load(f)["KernelWizard"]
      model_path = opt["pretrained"]
    blur_model = KernelWizard(opt)
    blur_model.eval()
    blur_model.load_state_dict(torch.load(model_path))
    # blur_model = blur_model.to(device)
    return blur_model

  opt_yml_path = './bkse/options/generate_blur/default.yml'
  blur_model = get_blur_model(opt_yml_path)

  def observation_map(x):
    random_kernel = random.randn(1, config.image_size, 2, 2) * 1.2
    # x = (x + 1.) / 2.  # do I need?
    blurred = blur_model(x, kernel=random_kernel)
    return blurred

  y = observation_map(x)
  y = y + random.normal(rng, y.shape) * config.sampling.noise_std
  num_obs = jnp.size(y)
  return x, y, observation_map, num_obs


def image_grid(x, image_size, num_channels):
    img = x.reshape(-1, image_size, image_size, num_channels)
    w = int(np.sqrt(img.shape[0]))
    img = img[:w**2, :, :, :]
    return img.reshape((w, w, image_size, image_size, num_channels)).transpose((0, 2, 1, 3, 4)).reshape((w * image_size, w * image_size, num_channels))


def plot_samples(x, image_size=32, num_channels=3, fname="samples"):
    img = image_grid(x, image_size, num_channels)
    plt.figure(figsize=(8,8))
    plt.axis('off')
    # NOTE: imshow resamples images so that the display image may not be the same resolution as the input
    plt.imshow(img, interpolation=None)
    plt.savefig(fname + '.png', bbox_inches='tight', pad_inches=0.0)
    plt.savefig(fname + '.pdf', bbox_inches='tight', pad_inches=0.0)
    plt.close()


def plot(train_data, mean, std, xlabel='x', ylabel='y',
         fname="plot.png"):
  BG_ALPHA = 1.0
  MG_ALPHA = 1.0
  FG_ALPHA = 0.3
  X, y = train_data
  # Plot result
  fig, ax = plt.subplots(1, 1)
  ax.scatter(X, y, label="Observations", color="black", s=20)
  ax.fill_between(
      X.flatten(), mean - 2. * std,
      mean + 2. * std, alpha=FG_ALPHA, color="blue")
  ax.set_xlim((X[0], X[-1]))
  # ax.set_ylim((, ))
  ax.grid(visible=True, which='major', linestyle='-')
  ax.set_xlabel('x', fontsize=10)
  ax.set_ylabel('y', fontsize=10)
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  fig.patch.set_facecolor('white')
  fig.patch.set_alpha(BG_ALPHA)
  ax.patch.set_alpha(MG_ALPHA)
  ax.legend()
  fig.savefig(fname)
  plt.close()


def compute_metrics_inner(config, cs_method, eval_path, x, q_samples, data_pools, inception_model,
                          compute_lpips=True, save=True):
  samples = np.clip(q_samples * 255., 0, 255).astype(np.uint8)
  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  # LPIPS - Need to permute and rescale to calculate correctly
  if compute_lpips: loss_fn_vgg = lpips.LPIPS(net='vgg')
  # Evaluate FID scores
  # Force garbage collection before calling TensorFlow code for Inception network
  gc.collect()
  latents = run_inception_distributed(samples, inception_model, inceptionv3=inceptionv3)
  # Force garbage collection again before returning to JAX code
  gc.collect()
  # Save latent represents of the Inception network to disk
  tmp_logits = latents["logits"].numpy()
  tmp_pool_3 = latents["pool_3"].numpy()

  # Compute PSNR, SSIM, MSE, LPIPS across images, and save them in stats files
  if compute_lpips:
    _lpips = torch_lpips(loss_fn_vgg, x, q_samples)
  else:
    _lpips = -1 * np.ones(x.shape[0])
  _psnr = tf_psnr(x, q_samples, max_val=1.0).numpy()
  _ssim = tf_ssim(x, q_samples, max_val=1.0).numpy()
  _mse = mse_vmap(x, q_samples)
  lpips_mean = np.mean(_lpips)
  lpips_std = np.std(_lpips)
  psnr_mean = np.mean(_psnr)
  psnr_std = np.std(_psnr)
  ssim_mean = np.mean(_ssim)
  ssim_std = np.std(_ssim)
  mse_mean = np.mean(_mse)
  mse_std = np.std(_mse)
  idx = np.argwhere(_mse < 1.0).flatten()  # mse is scalar, so flatten is okay
  fraction_stable = len(idx) / jnp.shape(_mse)[0]
  all_stable_lpips = _lpips[idx]
  all_stable_psnr = _psnr[idx]
  all_stable_ssim = _ssim[idx]
  all_stable_mse = _mse[idx]
  stable_lpips_mean = np.mean(all_stable_lpips)
  stable_lpips_std = np.std(all_stable_lpips)
  stable_psnr_mean = np.mean(all_stable_psnr)
  stable_psnr_std = np.std(all_stable_psnr)
  stable_ssim_mean = np.mean(all_stable_ssim)
  stable_ssim_std = np.std(all_stable_ssim)
  stable_mse_mean = np.mean(all_stable_mse)
  stable_mse_std = np.std(all_stable_mse)

  print(tmp_pool_3.shape)
  # must have rank 2 to calculate distribution distances
  if tmp_pool_3.shape[0] > 1:
    # Compute FID/KID/IS on individual inverse problem
    if not inceptionv3:
      _inception_score = tfgan.eval.classifier_score_from_logits(tmp_logits)
      stable_inception_score = tfgan.eval.classifier_score_from_logits(tmp_logits[idx])
    else:
      _inception_score = -1
    _fid = tfgan.eval.frechet_classifier_distance_from_activations(
      data_pools, tmp_pool_3)
    stable_fid = tfgan.eval.frechet_classifier_distance_from_activations(
      data_pools, tmp_pool_3[idx])
    # Hack to get tfgan KID work for eager execution.
    _tf_data_pools = tf.convert_to_tensor(data_pools)
    _tf_tmp_pools = tf.convert_to_tensor(tmp_pool_3)
    stable_tf_tmp_pools = tf.convert_to_tensor(tmp_pool_3[idx])
    _kid = tfgan.eval.kernel_classifier_distance_from_activations(
      _tf_data_pools, _tf_tmp_pools).numpy()
    stable_kid = tfgan.eval.kernel_classifier_distance_from_activations(
      _tf_data_pools, stable_tf_tmp_pools).numpy()
    del _tf_data_pools, _tf_tmp_pools, stable_tf_tmp_pools

    logging.info("cs_method-{} - stable: {}, \
                  IS: {:6e}, FID: {:6e}, KID: {:6e}, \
                  SIS: {:6e}, SFID: {:6e}, SKID: {:6e}, \
                  LPIPS {:6e}+/-{:3e}, PSNR: {:6e}+/-{:3e}, SSIM: {:6e}+/-{:3e}, MSE: {:6e}+/-{:3e}, \
                  SLPIPS {:6e}+/-{:3e}, SPSNR: {:6e}+/-{:3e}, SSSIM: {:6e}+/-{:3e}, SMSE: {:6e}+/-{:3e}".format(
        cs_method, fraction_stable, _inception_score, _fid, _kid,
        stable_inception_score, stable_fid, stable_kid,
        lpips_mean, lpips_std, psnr_mean, psnr_std, ssim_mean, ssim_std, mse_mean, mse_std,
        stable_lpips_mean, stable_lpips_std, stable_psnr_mean, stable_psnr_std, stable_ssim_mean, stable_ssim_std, stable_mse_mean, stable_mse_std,
        ))

    if save: np.savez_compressed(
      eval_path + "_stats.npz",
      pool_3=tmp_pool_3, logits=tmp_logits,
      lpips=_lpips,
      psnr=_psnr,
      ssim=_ssim,
      mse=_mse,
      IS=_inception_score, fid=_fid, kid=_kid,
      stable_IS=stable_inception_score, stable_fid=stable_fid, stable_kid=stable_kid,
      lpips_mean=lpips_mean, lpips_std=lpips_std,
      psnr_mean=psnr_mean, psnr_std=psnr_std,
      ssim_mean=ssim_mean, ssim_std=ssim_std,
      mse_mean=mse_mean, mse_std=mse_std,
      stable_lpips_mean=stable_lpips_mean, stable_lpips_std=stable_lpips_std,
      stable_psnr=stable_psnr_mean, stable_psnr_std=stable_psnr_std,
      stable_ssim=stable_ssim_mean, stable_ssim_std=stable_ssim_std,
      stable_mse=stable_mse_mean, stable_mse_std=stable_mse_std
      )
  else:
    logging.info("cs_method-{} - stable: {}, \
                  PSNR: {:6e}+/-{:3e}, SSIM: {:6e}+/-{:3e}, MSE: {:6e}+/-{:3e}, \
                  SPSNR: {:6e}+/-{:3e}, SSSIM: {:6e}+/-{:3e}, SMSE: {:6e}+/-{:3e}".format(
        cs_method, fraction_stable,
        psnr_mean, psnr_std, ssim_mean, ssim_std, mse_mean, mse_std,
        stable_psnr_mean, stable_psnr_std, stable_ssim_mean, stable_ssim_std, stable_mse_mean, stable_mse_std,
        ))
    if save: np.savez_compressed(
      eval_path + "_stats.npz",
      pool_3=tmp_pool_3, logits=tmp_logits,
      lpips=_lpips,
      psnr=_psnr,
      ssim=_ssim,
      mse=_mse,
      psnr_mean=psnr_mean, psnr_std=psnr_std,
      ssim_mean=ssim_mean, ssim_std=ssim_std,
      mse_mean=mse_mean, mse_std=mse_std,
      stable_psnr=stable_psnr_mean, stable_psnr_std=stable_psnr_std,
      stable_ssim=stable_ssim_mean, stable_ssim_std=stable_ssim_std,
      stable_mse=stable_mse_mean, stable_mse_std=stable_mse_std
      )
  return (psnr_mean, psnr_std), (lpips_mean, lpips_std), (mse_mean, mse_std), (ssim_mean, ssim_std)


def compute_metrics(config, cs_methods, eval_folder):
  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  # Load pre-computed dataset statistics.
  data_stats = load_dataset_stats(config)
  data_pools = data_stats["pool_3"]
  for cs_method in cs_methods:
    config.sampling.cs_method = cs_method
    # eval_file = "{}_{}_eval_{}".format(
    #   config.sampling.noise_std, config.data.dataset, config.sampling.cs_method)  # OLD
    eval_file = "{}_{}_{}".format(
      config.sampling.noise_std, config.data.dataset, config.sampling.cs_method.lower())  # NEW
    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved
    all_logits = []
    all_pools = []
    all_lpips = []
    all_psnr = []
    all_ssim = []
    all_mse = []
    stats = tf.io.gfile.glob(os.path.join(eval_folder, eval_file + "_*_stats.npz"))
    flag = 1
    print(os.path.join(eval_folder, eval_file + "_*_stats.npz"))
    print(len(stats), "len stats")

    for stat_file in stats:
      with tf.io.gfile.GFile(stat_file, "rb") as fin:
        stat = np.load(fin)
        tmp_logits = stat["logits"]
        tmp_pools = stat["pool_3"]
        if flag:
          try:
            tmp_lpips = stat["lpips"]
            tmp_psnr = stat["psnr"]
            tmp_ssim = stat["ssim"]
            tmp_mse = stat["mse"]
            all_lpips.append(tmp_lpips)
            all_psnr.append(tmp_psnr)
            all_mse.append(tmp_mse)
            all_ssim.append(tmp_ssim)
          except:
            print("did not compute distance metrics")
            flag = 0

        if not inceptionv3:
          print(tmp_logits.shape)
          all_logits.append(tmp_logits)
          print(len(all_logits))
        all_pools.append(tmp_pools)

    if not inceptionv3:
      if len(all_logits) != 1:
        all_logits = np.concatenate(all_logits, axis=0)
      else:
        all_logits = np.array(all_logits[0])

    if len(all_pools) != 1:
      all_pools = np.concatenate(all_pools, axis=0)
      if flag:
        all_lpips = np.concatenate(all_lpips, axis=0)
        all_psnr = np.concatenate(all_psnr, axis=0)
        all_ssim = np.concatenate(all_ssim, axis=0)
        all_mse = np.concatenate(all_mse, axis=0)
    else:
      all_pools = np.array(all_pools[0])
      if flag:
        all_lpips = np.array(all_lpips[0])
        all_psnr = np.array(all_psnr[0])
        all_ssim = np.array(all_ssim[0])
        all_mse = np.array(all_mse[0])

    print("logits shape: ", all_logits.shape)
    if flag:
      lpips_mean = np.mean(all_lpips)
      lpips_std = np.std(all_lpips)
      psnr_mean = np.mean(all_psnr)
      psnr_std = np.std(all_psnr)
      ssim_mean = np.mean(all_ssim)
      ssim_std = np.std(all_ssim)
      mse_mean = np.mean(all_mse)
      mse_std = np.std(all_mse)
      # Find metrics for the subset of images that sampled stably, stable as defined by an mse
      # within a theoretical limit (image has support [0., 1.] so max mse is 1.0)
      idx = np.argwhere(all_mse < 1.0).flatten()  # mse is scalar, so flatten is okay
      fraction_stable = len(idx) / jnp.shape(all_mse)[0]
      all_stable_lpips = all_lpips[idx]
      all_stable_mse = all_mse[idx]
      all_stable_ssim = all_ssim[idx]
      all_stable_psnr = all_psnr[idx]
      stable_lpips_mean = np.mean(all_stable_lpips)
      stable_lpips_std = np.std(all_stable_lpips)
      stable_psnr_mean = np.mean(all_stable_psnr)
      stable_psnr_std = np.std(all_stable_psnr)
      stable_ssim_mean = np.mean(all_stable_ssim)
      stable_ssim_std = np.std(all_stable_ssim)
      stable_mse_mean = np.mean(all_stable_mse)
      stable_mse_std = np.std(all_stable_mse)

    # Compute FID/KID/IS on all samples together.
    if not inceptionv3:
      inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
      if flag: stable_inception_score = tfgan.eval.classifier_score_from_logits(all_logits[idx])
    else:
      inception_score = -1
      if flag: stable_inception_score = -1

    fid = tfgan.eval.frechet_classifier_distance_from_activations(
      data_pools, all_pools)
    if flag: stable_fid = tfgan.eval.frechet_classifier_distance_from_activations(
      data_pools, all_pools[idx])
    # Hack to get tfgan KID work for eager execution.
    tf_data_pools = tf.convert_to_tensor(data_pools)
    tf_all_pools = tf.convert_to_tensor(all_pools)
    if flag: tf_all_stable_pools = tf.convert_to_tensor(all_pools[idx])
    kid = tfgan.eval.kernel_classifier_distance_from_activations(
      tf_data_pools, tf_all_pools).numpy()
    if flag: stable_kid = tfgan.eval.kernel_classifier_distance_from_activations(
      tf_data_pools, tf_all_stable_pools).numpy()
    del tf_data_pools, tf_all_pools
    if flag: del tf_all_stable_pools

    if flag:
      logging.info("cs_method-{} - stable: {}, \
                   IS: {:6e}, FID: {:6e}, KID: {:6e}, \
                   SIS: {:6e}, SFID: {:6e}, SKID: {:6e}, \
                   LPIPS: {:6e}+/-{:3e}, PSNR: {:6e}+/-{:3e}, SSIM: {:6e}+/-{:3e}, MSE: {:6e}+/-{:3e}, \
                   SLPIPS: {:6e}+/-{:3e}, SPSNR: {:6e}+/-{:3e}, SSSIM: {:6e}+/-{:3e}, SMSE: {:6e}+/-{:3e}".format(
          cs_method, fraction_stable,
          inception_score, fid, kid,
          stable_inception_score, stable_fid, stable_kid,
          lpips_mean, lpips_std, psnr_mean, psnr_std, ssim_mean, ssim_std, mse_mean, mse_std,
          stable_lpips_mean, stable_lpips_std, stable_psnr_mean, stable_psnr_std, stable_ssim_mean, stable_ssim_std, stable_mse_mean, stable_mse_std,
          ))
      np.savez_compressed(
        eval_file + "_reports.npz",
        IS=inception_score, fid=fid, kid=kid,
        stable_IS=stable_inception_score, stable_fid=stable_fid, stable_kid=stable_kid,
        lpips_mean=lpips_mean, lpips_std=lpips_std,
        psnr_mean=psnr_mean, psnr_std=psnr_std,
        ssim_mean=ssim_mean, ssim_std=ssim_std,
        mse_mean=mse_mean, mse_std=mse_std,
        stable_lpips_mean=stable_lpips_mean, stable_lpips_std=stable_lpips_std,
        stable_psnr=stable_psnr_mean, stable_psnr_std=stable_psnr_std,
        stable_ssim=stable_ssim_mean, stable_ssim_std=stable_ssim_std,
        stable_mse=stable_mse_mean, stable_mse_std=stable_mse_std
        )
    else:
      logging.info("cs_method-{} - inception_score: {:6e}, FID: {:6e}, KID: {:6e}".format(
          cs_method, inception_score, fid, kid))
      np.savez_compressed(
        eval_file + "_reports.npz", IS=inception_score, fid=fid, kid=kid,
        )


def deblur(config, workdir, eval_folder="eval"):
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.eval.pmap else 1

  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = random.PRNGKey(config.seed + 1)

  # Initialize model
  rng, model_rng = random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config, num_devices)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args
  checkpoint_dir = workdir
  cs_methods, sde = get_sde(config)

  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = random.fold_in(rng, jax.host_id())
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

  num_sampling_rounds = 2
  for i in range(num_sampling_rounds):
    x = get_eval_sample(scaler, inverse_scaler, config, eval_folder, num_devices)
    # x = get_prior_sample(rng, score_fn, epsilon_fn, sde, inverse_scaler, sampling_shape, config, eval_folder)
    _, y, observation_map, _ = get_blur_observation(
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
    np.savez(eval_folder + "/{}_{}_ground_observed_{}.npz".format(
      config.sampling.noise_std, config.data.dataset, i),
      x=jnp.array(x), y=y, noise_std=config.sampling.noise_std)
    print(observation_map)
    assert 0
    adjoint_observation_map = None

    H = None
    cs_method = config.sampling.cs_method

    for cs_method in cs_methods:
      config.sampling.cs_method = cs_method
      if cs_method in ddim_methods:
        sampler = get_cs_sampler(config, sde, epsilon_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=False)
      else:
        sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=False)

      rng, sample_rng = random.split(rng, 2)
      if config.eval.pmap:
        # sampler = jax.pmap(sampler, axis_name='batch')
        rng, *sample_rng = random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
      else:
        rng, sample_rng = random.split(rng, 2)

      q_samples, _ = sampler(sample_rng)
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
  rng = random.PRNGKey(config.seed + 1)

  # Initialize model
  rng, model_rng = random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config, num_devices)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  checkpoint_dir = workdir
  cs_methods, sde = get_sde(config)
  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = random.fold_in(rng, jax.host_id())
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
  shape=(
    config.eval.batch_size, config.data.image_size//4, config.data.image_size//4, config.data.num_channels)
  method='nearest'  # 'bicubic'
  num_sampling_rounds = 3

  x = get_asset_sample(config)
  _, y, *_ = get_superresolution_observation(
    rng, x, config,
    shape=shape,
    method=method)
  for i in range(num_sampling_rounds):
    # x = get_eval_sample(scaler, config, num_devices)
    # x = get_prior_sample(rng, score_fn, epsilon_fn, sde, inverse_scaler, sampling_shape, config, eval_folder)
    # _, y, *_ = get_superresolution_observation(
    #   rng, x, config,
    #   shape=shape,
    #   method=method)
    plot_samples(
      inverse_scaler(x.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_ground_{}_{}".format(
        config.sampling.noise_std, config.data.dataset, i))
    plot_samples(
      inverse_scaler(y.copy()),
      image_size=shape[1],
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_observed_{}_{}".format(
        config.sampling.noise_std, config.data.dataset, i))

    np.savez(eval_folder + "/{}_{}_ground_observed_{}.npz".format(
      config.sampling.noise_std, config.data.dataset, i),
      x=x, y=y, noise_std=config.sampling.noise_std)

    def observation_map(x):
      x = x.reshape(sampling_shape[1:])
      y = jax.image.resize(x, shape[1:], method)
      return y.flatten()

    def adjoint_observation_map(y):
      y = y.reshape(shape[1:])
      x = jax.image.resize(y, sampling_shape[1:], method)
      return x.flatten()

    H = None
    cs_method = config.sampling.cs_method

    for cs_method in cs_methods:
      config.sampling.cs_method = cs_method
      if cs_method in ddim_methods:
        sampler = get_cs_sampler(config, sde, epsilon_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=config.sampling.stack_samples)
      else:
        sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=config.sampling.stack_samples)

      rng, sample_rng = random.split(rng, 2)
      if config.eval.pmap:
        # sampler = jax.pmap(sampler, axis_name='batch')
        rng, *sample_rng = random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
      else:
        rng, sample_rng = random.split(rng, 2)

      time_prev = time.time()
      q_samples, _ = sampler(sample_rng)
      sample_time = time.time() - time_prev
      print("{}: {}s".format(cs_method, sample_time))
      if config.sampling.stack_samples:
        q_samples = q_samples.reshape(
          (config.solver.num_outer_steps, config.eval.batch_size,) + sampling_shape[1:])
        frames = 100

        fig = plt.figure(figsize=[1,1])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.set_frame_on(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img = image_grid(
          q_samples[-1],
          config.data.image_size, config.data.num_channels)
        im = ax.imshow(img, interpolation=None)
        fig.tight_layout()
        def animate(i, ax):
          ax.clear()
          idx = 1000 - int((i + 1) * config.solver.num_outer_steps / frames)
          img = image_grid(
            q_samples[idx],
            config.data.image_size, config.data.num_channels)
          ax.imshow(img, interpolation=None)

        # Plot animation of the trained score over time
        plot_animation(fig, ax,
                       animate, frames,
                       fname=eval_folder + "/{}_{}_{}_{}".format(config.data.dataset, config.sampling.noise_std, config.sampling.cs_method.lower(), i),
                       bitrate=1000,
                       dpi=300,
                       )
        plot_samples(
          q_samples[0],
          image_size=config.data.image_size,
          num_channels=config.data.num_channels,
          fname=eval_folder + "/{}_{}_{}_{}".format(config.data.dataset, config.sampling.noise_std, config.sampling.cs_method.lower(), i))

      else:
        q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])
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

  rng = random.PRNGKey(config.seed + 1)

  # Initialize model
  rng, model_rng = random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config, num_devices)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args
  checkpoint_dir = workdir
  cs_methods, sde = get_sde(config)

  # Create data normalizer and its inverse
  # scaler = datasets.get_data_scaler(config)
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
  rng = random.fold_in(rng, jax.host_id())
  num_sampling_rounds = 2

  x = get_asset_sample(config)
  _, y, mask, num_obs = get_inpainting_observation(rng, x, config, mask_name='square')
  for i in range(num_sampling_rounds):
    # x = get_eval_sample(scaler, config, num_devices)
    # x = get_prior_sample(rng, score_fn, epsilon_fn, sde, inverse_scaler, sampling_shape, config, eval_folder)
    # _, y, mask, num_obs = get_inpainting_observation(rng, x, config, mask_name='half')
    plot_samples(
      inverse_scaler(x.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_ground_{}_{}".format(
        config.sampling.noise_std, config.data.dataset, i))
    plot_samples(
      inverse_scaler(y.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_observed_{}_{}".format(
        config.sampling.noise_std, config.data.dataset, i))
    np.savez(eval_folder + "/{}_{}_ground_observed_{}.npz".format(
      config.sampling.noise_std, config.data.dataset, i),
      x=x, y=y, noise_std=config.sampling.noise_std)

    if 'plus' not in config.sampling.cs_method:
      logging.warning(
        "Using full H matrix H.shape={} which may be too large to fit in memory ".format(
          (num_obs, config.data.image_size**2 * config.data.num_channels)))
      idx_obs = np.nonzero(mask)[0]
      H = jnp.zeros((num_obs, config.data.image_size**2 * config.data.num_channels))
      ogrid = np.arange(num_obs, dtype=int)
      H = H.at[ogrid, idx_obs].set(1.0)
      y = batch_matmul_A(H, y)

      def observation_map(x):
          x = x.flatten()
          return H @ x
      adjoint_observation_map = None
    else:
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

      rng, sample_rng = random.split(rng, 2)
      if config.eval.pmap:
        sampler = jax.pmap(sampler, axis_name='batch')
        rng, *sample_rng = random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
      else:
        rng, sample_rng = random.split(rng, 2)

      time_prev = time.time()
      q_samples, _ = sampler(sample_rng)
      sample_time = time.time() - time_prev
      print("{}: {}s".format(cs_method, sample_time))
      q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])
      plot_samples(
        q_samples,
        image_size=config.data.image_size,
        num_channels=config.data.num_channels,
        fname=eval_folder + "/{}_{}_{}_{}".format(config.sampling.noise_std, config.data.dataset, config.sampling.cs_method.lower(), i))


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
  num_devices =  int(jax.local_device_count()) if config.eval.pmap else 1

  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = random.PRNGKey(config.seed + 1)

  # Create data normalizer and its inverse
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  rng, model_rng = random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config, num_devices)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  checkpoint_dir = workdir
  _, sde = get_sde(config)

  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = random.fold_in(rng, jax.host_id())
  ckpt = config.eval.begin_ckpt

  # Wait if the target checkpoint doesn't exist yet
  ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))

  if not tf.io.gfile.exists(ckpt_filename):
    raise FileNotFoundError("{} does not exist".format(ckpt_filename))

  state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)

  # score_fn is vmap'd
  score_fn = mutils.get_score_fn(
    sde, score_model, state.params_ema, state.model_state, train=False, continuous=True)

  sampler = get_sampler(
    (4, config.data.image_size, config.data.image_size, config.data.num_channels),
    EulerMaruyama(sde.reverse(score_fn), num_steps=config.model.num_scales))
  q_samples, num_function_evaluations = sampler(rng)
  print("num_function_evaluations", num_function_evaluations)
  q_samples = inverse_scaler(q_samples)
  plot_samples(
    q_samples,
    image_size=config.data.image_size,
    num_channels=config.data.num_channels,
    fname="{} samples".format(config.data.dataset))


def evaluate_inpainting(config,
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

  rng = random.PRNGKey(config.seed + 1)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  rng, model_rng = random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config, num_devices)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  checkpoint_dir = workdir
  cs_methods, sde = get_sde(config)

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
  rng = random.fold_in(rng, jax.host_id())
  num_sampling_rounds = 2

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = get_inception_model(inceptionv3=inceptionv3)
  # Load pre-computed dataset statistics.
  data_stats = load_dataset_stats(config)
  data_pools = data_stats["pool_3"]
  for i in range(num_sampling_rounds):
    x = get_eval_sample(scaler, config, num_devices)
    # x = get_prior_sample(rng, score_fn, epsilon_fn, sde, inverse_scaler, sampling_shape, config, eval_folder)
    # x = get_asset_sample(config)
    x_flat, y, mask, num_obs = get_inpainting_observation(rng, x, config, mask_name='square')
    plot_samples(
      inverse_scaler(x_flat.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_{}_ground_{}".format(
        config.sampling.noise_std, config.data.dataset, i))
    plot_samples(
      inverse_scaler(y.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_{}_observed_{}".format(
        config.sampling.noise_std, config.data.dataset, i))
    np.savez(eval_folder + "/{}_{}_ground_observed_{}.npz".format(
      config.sampling.noise_std, config.data.dataset, i),
      x=x, y=y, noise_std=config.sampling.noise_std)

    if 'plus' not in config.sampling.cs_method:
      logging.warning(
        "Using full H matrix H.shape={} which may be too large to fit in memory ".format(
          (num_obs, config.data.image_size**2 * config.data.num_channels)))
      idx_obs = np.nonzero(mask)[0]
      H = jnp.zeros((num_obs, config.data.image_size**2 * config.data.num_channels))
      ogrid = np.arange(num_obs, dtype=int)
      H = H.at[ogrid, idx_obs].set(1.0)
      y = H @ y

      def observation_map(x):
          x = x.flatten()
          return H @ x
      adjoint_observation_map = None
    else:
      y = y
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

      rng, sample_rng = random.split(rng, 2)
      if config.eval.pmap:
        sampler = jax.pmap(sampler, axis_name='batch')
        rng, *sample_rng = random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
      else:
        rng, sample_rng = random.split(rng, 2)

      time_prev = time.time()
      q_samples, _ = sampler(sample_rng)
      sample_time = time.time() - time_prev
      print("{}: {}s".format(cs_method, sample_time))
      q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])
      eval_file = "{}_{}_{}_{}".format(
        config.sampling.noise_std, config.data.dataset, config.sampling.cs_method.lower(), i)
      eval_path = eval_folder + eval_file
      np.savez(eval_path + ".npz", x=q_samples, y=y, noise_std=config.sampling.noise_std)
      plot_samples(
        q_samples,
        image_size=config.data.image_size,
        num_channels=config.data.num_channels,
        fname=eval_path)
      compute_metrics_inner(config, cs_method, eval_path, x, q_samples, data_pools, inception_model,
                            compute_lpips=True, save=True)

  compute_metrics(config, cs_methods, eval_folder)


def evaluate_super_resolution(config,
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

  rng = random.PRNGKey(config.seed + 1)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  rng, model_rng = random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config, num_devices)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args
  checkpoint_dir = workdir
  cs_methods, sde = get_sde(config)

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
  rng = random.fold_in(rng, jax.host_id())

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = get_inception_model(inceptionv3=inceptionv3)
  # Load pre-computed dataset statistics.
  data_stats = load_dataset_stats(config)
  data_pools = data_stats["pool_3"]

  shape=(config.eval.batch_size, config.data.image_size//4, config.data.image_size//4, config.data.num_channels)
  method='bicubic'  # 'bicubic'
  num_sampling_rounds = 6

  for i in range(num_sampling_rounds):
    x = get_eval_sample(scaler, config, num_devices)
    # x = get_prior_sample(rng, score_fn, epsilon_fn, sde, inverse_scaler, sampling_shape, config, eval_folder)
    # x = get_asset_sample(config)
    x_flat, y, *_ = get_superresolution_observation(
      rng, x, config,
      shape=shape,
      method=method)
    plot_samples(
      inverse_scaler(x_flat.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_{}_ground_{}".format(
        config.sampling.noise_std, config.data.dataset, i))
    plot_samples(
      inverse_scaler(y.copy()),
      image_size=shape[1],
      num_channels=config.data.num_channels,
      fname=eval_folder + "/_{}_{}_observed_{}".format(
        config.sampling.noise_std, config.data.dataset, i))
    np.savez(eval_folder + "/{}_{}_ground_observed_{}.npz".format(
      config.sampling.noise_std, config.data.dataset, i),
      x=x, y=y, noise_std=config.sampling.noise_std)

    def observation_map(x):
      x = x.reshape(sampling_shape[1:])
      y = jax.image.resize(x, shape[1:], method)
      return y.flatten()

    def adjoint_observation_map(y):
      y = y.reshape(shape[1:])
      x = jax.image.resize(y, sampling_shape[1:], method)
      return x.flatten()

    H = None

    for cs_method in cs_methods:
      config.sampling.cs_method = cs_method
      if cs_method in ddim_methods:
        sampler = get_cs_sampler(config, sde, epsilon_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=False)
      else:
        sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
          y, H, observation_map, adjoint_observation_map, stack_samples=False)

      rng, sample_rng = random.split(rng, 2)
      if config.eval.pmap:
        sampler = jax.pmap(sampler, axis_name='batch')
        rng, *sample_rng = random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
      else:
        rng, sample_rng = random.split(rng, 2)

      time_prev = time.time()
      q_samples, _ = sampler(sample_rng)
      sample_time = time.time() - time_prev
      print("{}: {}s".format(cs_method, sample_time))
      q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])
      eval_file = "{}_{}_{}_{}".format(
        config.sampling.noise_std, config.data.dataset, config.sampling.cs_method.lower(), i)
      eval_path = eval_folder + eval_file
      np.savez(eval_path + ".npz", x=q_samples, y=y, noise_std=config.sampling.noise_std)
      plot_samples(
        q_samples,
        image_size=config.data.image_size,
        num_channels=config.data.num_channels,
        fname=eval_path)
      compute_metrics_inner(config, cs_method, eval_path, x, q_samples, data_pools, inception_model,
                            compute_lpips=True, save=True)

  compute_metrics(config, cs_methods, eval_folder)


def evaluate_from_file(config,
                   workdir,
                   eval_folder="eval"):
  cs_methods, _ = get_sde(config)
  compute_metrics(config, cs_methods, eval_folder)


def dps_search_inpainting(
    config,
    workdir,
    eval_folder="eval"):
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.eval.pmap else 1

  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = random.PRNGKey(config.seed + 1)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  rng, model_rng = random.split(rng)
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
      # cs_method = 'chung2022scalar'
      cs_method = 'DPSDDPM'
    else:
      # VP/DDM methods with mask
      # cs_method = 'chung2022scalarplus'
      cs_method = 'DPSDDPMplus'
  elif config.training.sde.lower() == 'vesde':
    sde = VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    if 'plus' not in config.sampling.cs_method:
      # VE/SMLD Methods with matrix H
      # cs_method = 'chung2022scalar'
      cs_method = 'DPSSMLD'
    else:
      # cs_method = 'chung2022scalarplus'
      cs_method = 'DPSSMLDplus'
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
  rng = random.fold_in(rng, jax.host_id())

  num_sampling_rounds = 10

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = get_inception_model(inceptionv3=inceptionv3)
  # Load pre-computed dataset statistics.
  data_stats = load_dataset_stats(config)
  data_pools = data_stats["pool_3"]

  dps_hyperparameters = jnp.logspace(-1.5, 0.4, num=num_sampling_rounds, base=10.0)
  psnr_means = []
  psnr_stds = []
  lpips_means = []
  lpips_stds = []
  ssim_means = []
  ssim_stds = []
  mse_means = []
  mse_stds = []

  for scale in dps_hyperparameters:
    # round to 3 sig fig
    scale = float(f'{float(f"{scale:.3g}"):g}')
    config.solver.dps_scale_hyperparameter = scale
    x = get_eval_sample(scaler, config, num_devices)
    # x = get_prior_sample(rng, score_fn, epsilon_fn, sde, inverse_scaler, sampling_shape, config, eval_folder)
    # x = get_asset_sample(config)

    x_flat, y, mask, num_obs = get_inpainting_observation(rng, x, config, mask_name='square')
    plot_samples(
      inverse_scaler(x_flat.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/search_{}_{}_ground_{}".format(
        config.sampling.noise_std, config.data.dataset, scale))
    plot_samples(
      inverse_scaler(y.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/search_{}_{}_observed_{}".format(
        config.sampling.noise_std, config.data.dataset, scale))
    np.savez(eval_folder + "/search_{}_{}_ground_observed_{}.npz".format(
      config.sampling.noise_std, config.data.dataset, scale),
      x=x, y=y, noise_std=config.sampling.noise_std)

    if 'plus' not in config.sampling.cs_method and mask:
      logging.warning(
        "Using full H matrix H.shape={} which may be too large to fit in memory ".format(
          (num_obs, config.data.image_size**2 * config.data.num_channels)))
      idx_obs = np.nonzero(mask)[0]
      H = jnp.zeros((num_obs, config.data.image_size**2 * config.data.num_channels))
      ogrid = np.arange(num_obs, dtype=int)
      H = H.at[ogrid, idx_obs].set(1.0)
      y = H @ y

      def observation_map(x):
          x = x.flatten()
          return H @ x
      adjoint_observation_map = None
    else:
      def observation_map(x):
          x = x.flatten()
          return mask * x
      adjoint_observation_map = None
      H = None

    config.sampling.cs_method = cs_method
    if cs_method in ddim_methods:
      sampler = get_cs_sampler(config, sde, epsilon_fn, sampling_shape, inverse_scaler,
        y, H, observation_map, adjoint_observation_map, stack_samples=False)
    else:
      sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
        y, H, observation_map, adjoint_observation_map, stack_samples=False)

    rng, sample_rng = random.split(rng, 2)
    if config.eval.pmap:
      sampler = jax.pmap(sampler, axis_name='batch')
      rng, *sample_rng = random.split(rng, jax.local_device_count() + 1)
      sample_rng = jnp.asarray(sample_rng)
    else:
      rng, sample_rng = random.split(rng, 2)

    time_prev = time.time()
    q_samples, _ = sampler(sample_rng)
    sample_time = time.time() - time_prev
    print("{}: {}s".format(cs_method, sample_time))
    q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])
    plot_samples(
      q_samples,
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/search_{}_{}_{}_{}".format(config.sampling.noise_std, config.data.dataset, config.sampling.cs_method.lower(), scale))
    # Save samples
    eval_file = "search_{}_{}_{}_{}".format(
      config.sampling.noise_std, config.data.dataset, config.sampling.cs_method.lower(), scale)
    eval_path = eval_folder + eval_file
    np.savez(eval_path + ".npz", x=q_samples, y=y, noise_std=config.sampling.noise_std)
    (psnr_mean, psnr_std), (lpips_mean, lpips_std), (mse_mean, mse_std), (ssim_mean, ssim_std) = compute_metrics_inner(
      config, cs_method, eval_path, x, q_samples, data_pools, inception_model,
      compute_lpips=True, save=False)
    psnr_means.append(psnr_mean)
    psnr_stds.append(psnr_std)
    lpips_means.append(lpips_mean)
    lpips_stds.append(lpips_std)
    ssim_means.append(ssim_mean)
    ssim_stds.append(ssim_std)
    mse_means.append(mse_mean)
    mse_stds.append(mse_std)

  psnr_means = np.array(psnr_means)
  psnr_stds = np.array(psnr_stds)
  lpips_means = np.array(lpips_means)
  lpips_stds = np.array(lpips_stds)
  ssim_means = np.array(ssim_means)
  ssim_stds = np.array(ssim_stds)
  mse_means = np.array(mse_means)
  mse_stds = np.array(mse_stds)

  # plot hyperparameter search
  plot((dps_hyperparameters, psnr_means), psnr_means, psnr_stds, xlabel='dps_scale', ylabel='psnr', fname=eval_folder + "dps_psnr.png")
  plot((dps_hyperparameters, lpips_means), lpips_means, lpips_stds, xlabel='dps_scale', ylabel='lpips', fname=eval_folder + "dps_lpips.png")
  plot((dps_hyperparameters, mse_means), mse_means, mse_stds, xlabel='dps_scale', ylabel='mse', fname=eval_folder + "dps_mse.png")
  plot((dps_hyperparameters, ssim_means), ssim_means, ssim_stds, xlabel='dps_scale', ylabel='ssim', fname=eval_folder + "dps_ssim.png")


def dps_search_super_resolution(config,
             workdir,
             eval_folder="eval"):
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.eval.pmap else 1

  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = random.PRNGKey(config.seed + 1)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  rng, model_rng = random.split(rng)
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
      # cs_method = 'chung2022scalar'
      cs_method = 'DPSDDPM'
    else:
      # VP/DDM methods with mask
      # cs_method = 'chung2022scalarplus'
      cs_method = 'DPSDDPMplus'
  elif config.training.sde.lower() == 'vesde':
    sde = VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    if 'plus' not in config.sampling.cs_method:
      # VE/SMLD Methods with matrix H
      # cs_method = 'chung2022scalar'
      cs_method = 'DPSSMLD'
    else:
      # cs_method = 'chung2022scalarplus'
      cs_method = 'DPSSMLDplus'
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
  rng = random.fold_in(rng, jax.host_id())

  num_sampling_rounds = 10

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = get_inception_model(inceptionv3=inceptionv3)
  # Load pre-computed dataset statistics.
  data_stats = load_dataset_stats(config)
  data_pools = data_stats["pool_3"]

  dps_hyperparameters = jnp.logspace(-1.5, 0.4, num=num_sampling_rounds, base=10.0)
  psnr_means = []
  psnr_stds = []
  lpips_means = []
  lpips_stds = []
  ssim_means = []
  ssim_stds = []
  mse_means = []
  mse_stds = []

  shape = (config.eval.batch_size, config.data.image_size//4, config.data.image_size//4, config.data.num_channels)
  method = 'bicubic'
  for scale in dps_hyperparameters:
    # round to 3 sig fig
    scale = float(f'{float(f"{scale:.3g}"):g}')
    config.solver.dps_scale_hyperparameter = scale
    x = get_eval_sample(scaler, config, num_devices)
    # x = get_prior_sample(rng, score_fn, epsilon_fn, sde, inverse_scaler, sampling_shape, config, eval_folder)
    # x = get_asset_sample(config)

    x_flat, y, mask, num_obs = get_superresolution_observation(
      rng, x, config,
      shape=shape,
      method=method)
    plot_samples(
      inverse_scaler(x_flat.copy()),
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/search_{}_{}_ground_{}".format(
        config.sampling.noise_std, config.data.dataset, scale))
    plot_samples(
      inverse_scaler(y.copy()),
      image_size=shape[1],
      num_channels=config.data.num_channels,
      fname=eval_folder + "/search_{}_{}_observed_{}".format(
        config.sampling.noise_std, config.data.dataset, scale))
    np.savez(eval_folder + "/search_{}_{}_ground_observed_{}.npz".format(
      config.sampling.noise_std, config.data.dataset, scale),
      x=x, y=y, noise_std=config.sampling.noise_std)

    def observation_map(x):
      x = x.reshape(sampling_shape[1:])
      y = jax.image.resize(x, shape[1:], method)
      return y.flatten()

    def adjoint_observation_map(y):
      y = y.reshape(shape[1:])
      x = jax.image.resize(y, sampling_shape[1:], method)
      return x.flatten()

    H = None

    config.sampling.cs_method = cs_method
    if cs_method in ddim_methods:
      sampler = get_cs_sampler(config, sde, epsilon_fn, sampling_shape, inverse_scaler,
        y, H, observation_map, adjoint_observation_map, stack_samples=False)
    else:
      sampler = get_cs_sampler(config, sde, score_fn, sampling_shape, inverse_scaler,
        y, H, observation_map, adjoint_observation_map, stack_samples=False)

    rng, sample_rng = random.split(rng, 2)
    if config.eval.pmap:
      sampler = jax.pmap(sampler, axis_name='batch')
      rng, *sample_rng = random.split(rng, jax.local_device_count() + 1)
      sample_rng = jnp.asarray(sample_rng)
    else:
      rng, sample_rng = random.split(rng, 2)

    time_prev = time.time()
    q_samples, _ = sampler(sample_rng)
    sample_time = time.time() - time_prev
    print("{}: {}s".format(cs_method, sample_time))
    q_samples = q_samples.reshape((config.eval.batch_size,) + sampling_shape[1:])
    plot_samples(
      q_samples,
      image_size=config.data.image_size,
      num_channels=config.data.num_channels,
      fname=eval_folder + "/search_{}_{}_{}_{}".format(config.sampling.noise_std, config.data.dataset, config.sampling.cs_method.lower(), scale))
    # Save samples
    eval_file = "search_{}_{}_{}_{}".format(
      config.sampling.noise_std, config.data.dataset, config.sampling.cs_method.lower(), scale)
    eval_path = eval_folder + eval_file
    np.savez(eval_path + ".npz", x=q_samples, y=y, noise_std=config.sampling.noise_std)
    (psnr_mean, psnr_std), (lpips_mean, lpips_std), (mse_mean, mse_std), (ssim_mean, ssim_std) = compute_metrics_inner(
      config, cs_method, eval_path, x, q_samples, data_pools, inception_model,
      compute_lpips=True, save=False)
    psnr_means.append(psnr_mean)
    psnr_stds.append(psnr_std)
    lpips_means.append(lpips_mean)
    lpips_stds.append(lpips_std)
    ssim_means.append(ssim_mean)
    ssim_stds.append(ssim_std)
    mse_means.append(mse_mean)
    mse_stds.append(mse_std)

  psnr_means = np.array(psnr_means)
  psnr_stds = np.array(psnr_stds)
  lpips_means = np.array(lpips_means)
  lpips_stds = np.array(lpips_stds)
  ssim_means = np.array(ssim_means)
  ssim_stds = np.array(ssim_stds)
  mse_means = np.array(mse_means)
  mse_stds = np.array(mse_stds)

  # plot hyperparameter search
  plot((dps_hyperparameters, psnr_means), psnr_means, psnr_stds, xlabel='dps_scale', ylabel='psnr', fname=eval_folder + "dps_psnr.png")
  plot((dps_hyperparameters, lpips_means), lpips_means, lpips_stds, xlabel='dps_scale', ylabel='lpips', fname=eval_folder + "dps_lpips.png")
  plot((dps_hyperparameters, mse_means), mse_means, mse_stds, xlabel='dps_scale', ylabel='mse', fname=eval_folder + "dps_mse.png")
  plot((dps_hyperparameters, ssim_means), ssim_means, ssim_stds, xlabel='dps_scale', ylabel='ssim', fname=eval_folder + "dps_ssim.png")
