# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on CIFAR-10 with VE SDE."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True
  training.n_iters = 950001

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.fourier_scale = 16
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 8
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.0
  model.conv_size = 3

  # optim
  config.seed = 2023

  sampling.cs_method = 'tmpd2023bvjpplus'
  # sampling.cs_method = 'tmpd2023avjp'

  sampling.stack_samples = True
  sampling.noise_std = 0.01
  sampling.denoise = True
  sampling.inverse_scaler = None
  evaluate = config.eval
  evaluate.begin_ckpt = 12
  evaluate.end_ckpt = 12
  evaluate.batch_size = 1
  evaluate.pmap = False
  solver = config.solver
  solver.outer_solver = 'eulermaruyama'
  # solver.outer_solver = 'DDIMVE'
  # solver.outer_solver = 'SMLD'
  solver.num_outer_steps = model.num_scales
  solver.eta = 1.0  # DDIM hyperparameter
  # https://arxiv.org/pdf/2209.14687.pdf#page=20&zoom=100,144,757

  # inpainting half
  # solver.dps_scale_hyperparameter = 0.8  # for noise_std=0.01
  # solver.dps_scale_hyperparameter = 0.8  # for noise_std=0.05
  # solver.dps_scale_hyperparameter = 0.3  # for noise_std=0.1

  # inpainting square
  solver.dps_scale_hyperparameter = 0.5  # for noise_std=0.01
  # solver.dps_scale_hyperparameter = 0.3  # for noise_std=0.05
  # solver.dps_scale_hyperparameter = 0.05 # for noise_std=0.1

  # superresolution 2nearest
  # solver.dps_scale_hyperparameter = 0.8 # for noise_std=0.01
  # solver.dps_scale_hyperparameter = 0.5 # for noise_std=0.05
  # solver.dps_scale_hyperparameter = 0.2  # for noise_std=0.1

  # superresolution 4bicubic
  # solver.dps_scale_hyperparameter = 0.3 # for noise_std=0.01
  # solver.dps_scale_hyperparameter = 0.15 # for noise_std=0.05
  # solver.dps_scale_hyperparameter = 0.15  # for noise_std=0.1

  return config
