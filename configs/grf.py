"""Config for `grf.py`."""

from configs.default_cs_configs import get_default_configs


def get_config():
    config = get_default_configs()
    training = config.training
    model = config.model
    eval = config.eval
    sampling = config.sampling
    data = config.data
    solver = config.solver

    sampling.cs_method = 'TMPD2023ajacfwd'

    eval.pmap = True
    data.image_size = 32
    data.num_channels = 1
    eval.batch_size = 1500

    training.sde = 'vpsde'
    training.num_epochs = 4000
    training.batch_size = 16

    sampling.noise_std = 0.1
    sampling.denoise = True  # work out what denoise_override is
    sampling.innovation = True  # this will probably be superceded
    sampling.inverse_scaler = None

    model.beta_min = 0.01
    model.beta_max = 25.

    solver.num_outer_steps = 1000
    solver.outer_solver = 'EulerMaruyama'
    solver.inner_solver = None

    # optim
    config.seed = 2023

    return config
