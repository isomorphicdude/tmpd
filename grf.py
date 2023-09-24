"""
Two dimensional Gaussian Random Field for profiling purposes.
"""
# Uncomment to enable double precision
# from jax.config import config as jax_config
# jax_config.update("jax_enable_x64", True)

from absl import app, flags
from ml_collections.config_flags import config_flags

import jax
from jax import jit, vmap
import jax.random as random
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffusionjax.plot import plot_heatmap
from diffusionjax.solvers import EulerMaruyama
import diffusionjax.sde as sde_lib
from diffusionjax.utils import get_sampler

from mlkernels import Matern52
import lab as B
import numpy as np
from probit.approximators import LaplaceGP as GP
from probit.utilities import log_gaussian_likelihood
import scipy
import logging
import time

from source.plot import (
    Distance2,
    Wasserstein2,
    plot,
    plot, image_grid, plot_samples,
    plot_samples_1D
)
from source.samplers import get_cs_sampler


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "./configs/grf.py", "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", "./workdir/", "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])
logger = logging.getLogger(__name__)


x_max = 5.0
epsilon = 1e-4

# For plotting
FG_ALPHA = 0.3


def sample_image_rgb(rng, num_samples, image_size, kernel, num_channels):
    """Samples from a GMRF."""
    x = np.linspace(-x_max, x_max, image_size)
    y = np.linspace(-x_max, x_max, image_size)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(image_size**2, 1)
    yy = yy.reshape(image_size**2, 1)
    z = np.hstack((xx, yy))
    C = B.dense(kernel(z))  + epsilon * B.eye(image_size**2)
    u = random.multivariate_normal(rng, mean=jnp.zeros(xx.shape[0]), cov=C, shape=(num_samples, num_channels))
    u = u.transpose((0, 2, 1))
    return u, C, x, z


def main(argv):
    config = FLAGS.config
    jax.default_device = jax.devices()[0]
    # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
    # ... devices must be all the same model for pmap to work
    num_devices =  int(jax.local_device_count()) if config.eval.pmap else 1
    logging.info("num_devices={}, pmap={}".format(num_devices, config.eval.pmap))

    # Setup SDE
    if config.training.sde.lower()=='vpsde':
        sde = sde_lib.VP(beta_min=config.model.beta_min, beta_max=config.model.beta_max)
    elif config.training.sde.lower()=='vesde':
        sde = sde_lib.VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    else:
        raise NotImplementedError(f"SDE {config.training.SDE} unknown.")

    rng = random.PRNGKey(2023)
    samples, C, x, X = sample_image_rgb(
        rng, num_samples=config.eval.batch_size//num_devices,
        image_size=config.data.image_size, kernel=Matern52(),
        num_channels=config.data.num_channels)  # (num_samples, image_size**2, num_channels)

    # sde = VP(beta_min=1e-3, beta_max=1e1)
    # tmp_solver = EulerMaruyama(sde)

    if 0:
        # Reshape image data
        samples = samples.reshape(-1, config.data.image_size, config.data.image_size, config.data.num_channels)
        plot_samples(samples[:64], image_size=config.data.image_size, num_channels=config.data.num_channels, fname="samples")
        plot_samples_1D(samples[:64], config.data.image_size, "samples_1D", alpha=FG_ALPHA)

    def nabla_log_pt(x, t):
        """
        Args:
            x: One location in $\mathbb{R}^{image_size**2}$
            t: time
        Returns:
            The true log density.
            .. math::
                p_{t}(x)
        """
        x_shape = x.shape
        v_t = sde.variance(t)
        m_t = sde.mean_coeff(t)
        x = x.flatten()
        score = -jnp.linalg.solve(m_t**2 * C + v_t * jnp.eye(x.shape[0]), x)
        return score.reshape(x_shape)

    true_score = jit(vmap(nabla_log_pt, in_axes=(0, 0), out_axes=(0)))

    if 0:  # Prior sampling
        p_samples = random.multivariate_normal(rng, mean=jnp.zeros(config.data.image_size**2),
            cov=C, shape=(config.eval.batch_size,))
        C_emp = jnp.cov(p_samples[:, :].T)
        m_emp = jnp.mean(p_samples[:, :].T, axis=1)
        corr_emp = jnp.corrcoef(p_samples[:, :].T)
        plot_heatmap(samples=p_samples[:, [0, 1]], area_min=-3, area_max=3, fname="target_prior_heatmap")

        p_samples = p_samples.reshape(config.eval.batch_size, config.data.image_size, config.data.image_size)
        p_samples = jnp.expand_dims(p_samples, axis=3)
        plot_samples(p_samples[:64], image_size=config.data.image_size, num_channels=config.data.num_channels, fname="samples_prior")
        plot_samples_1D(p_samples, image_size=config.data.image_size, fname="analytic_prior_samples_1D", alpha=FG_ALPHA)
        delta_t_cov = jnp.linalg.norm(C - C_emp) / config.data.image_size
        delta_t_var = jnp.linalg.norm(jnp.diag(C) - jnp.diag(C_emp)) / config.data.image_size
        delta_t_mean = jnp.linalg.norm(m_emp) / config.data.image_size
        delta_t_corr = jnp.linalg.norm(C - corr_emp) / config.data.image_size
        logging.info("analytic_prior delta_mean={}, delta_var={}, delta_cov={}".format(
            delta_t_mean, delta_t_var, delta_t_cov))

        # Running the reverse SDE with the true score
        solver = EulerMaruyama(sde.reverse(true_score), num_steps=config.solver.num_outer_steps)
        plot_beta_schedule(sde, tmp_solver)
        plot_temperature_schedule(sde, tmp_solver)
        sampler= get_sampler((config.eval.batch_size//num_devices, config.data.image_size, config.data.image_size, config.data.num_channels), solver)
        if config.eval.pmap:
            sampler = jax.pmap(sampler, axis_name='batch')
            rng, *sample_rng = random.split(rng, 1 + num_devices)
            sample_rng = jnp.asarray(sample_rng)
        else:
            rng, sample_rng = random.split(rng, 1 + num_devices)
        q_samples, nfe = sampler(sample_rng)
        q_samples = q_samples.reshape(config.eval.batch_size, config.data.image_size**2)

        C_emp = jnp.cov(q_samples[:, :].T)
        m_emp = jnp.mean(q_samples[:, :].T, axis=1)
        corr_emp = jnp.corrcoef(q_samples[:, :].T)
        delta_corr = jnp.linalg.norm(C - corr_emp) / config.data.image_size
        delta_cov = jnp.linalg.norm(C - C_emp) / config.data.image_size
        delta_mean = jnp.linalg.norm(m_emp) / config.data.image_size
        plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-3, area_max=3, fname="diffusion_prior_heatmap")

        q_samples = q_samples.reshape(config.eval.batch_size, config.data.image_size, config.data.image_size)
        q_samples = np.expand_dims(q_samples, axis=3)
        plot_samples(p_samples[:64], image_size=config.data.image_size, num_channels=config.data.num_channels, fname="diffusion_prior_samples")
        plot_samples_1D(q_samples, image_size=config.data.image_size, fname="diffusion_prior_samples_1D", alpha=FG_ALPHA)
        delta_cov = jnp.linalg.norm(C - C_emp) / config.data.image_size
        delta_var = jnp.linalg.norm(jnp.diag(C) - jnp.diag(C_emp)) / config.data.image_size
        delta_mean = jnp.linalg.norm(m_emp) / config.data.image_size
        logging.info("diffusion_prior delta_mean={}, delta_var={}, delta_cov={}".format(
            delta_mean, delta_var, delta_cov))

        # logging.info(delta_t_corr)  # a value of 0.05 (for 512 samples) are indistinguisable from
        # true samples due to emprical covariance error
        # but it is possible to get a value as los as 0.005 from many more true samples
        # logging.info(delta_corr)  # a value of 0.1 are good samples

        # # Running the reverse SDE with the true score
        # # Get the outer loop of a numerical solver, also known as "predictor"
        # outer_solver = EulerMaruyama(sde.reverse(true_score), num_steps=config.solver.num_outer_steps)

        # sampler = get_sampler((config.eval.batch_size//num_devices, config.data.image_size, config.data.image_size, config.data.num_channels), outer_solver, inner_solver, denoise=True)
        # if config.eval.pmap:
        #     rng, *sample_rng = random.split(rng, 1 + num_devices)
        #     sample_rng = jnp.asarray(sample_rng)
        # else:
        #     rng, sample_rng = random.split(rng, 1 + num_devices)
        # q_samples = sampler(sample_rng)
        # q_samples = q_samples.reshape(config.eval.batch_size, config.data.image_size**2)

        # C_emp = jnp.cov(q_samples[:, :].T)
        # m_emp = jnp.mean(q_samples[:, :].T, axis=1)
        # corr_emp = jnp.corrcoef(q_samples[:, :].T)
        # delta_corr = jnp.linalg.norm(C - corr_emp) / config.data.image_size
        # delta_cov = jnp.linalg.norm(C - C_emp) / config.data.image_size
        # delta_var = jnp.linalg.norm(jnp.diag(C) - jnp.diag(C_emp)) / config.data.image_size
        # delta_mean = jnp.linalg.norm(m_emp) / config.data.image_size
        # logging.info("prior_PC_analytic delta_mean={}, delta_var={}, delta_cov={}".format(
        #     delta_mean, delta_var, delta_cov))
        # plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-3, area_max=3, fname="heatmap PC analytic score")

        # q_samples = q_samples.reshape(config.eval.batch_size, config.data.image_size, config.data.image_size)
        # q_samples = np.expand_dims(q_samples, axis=3)
        # plot_samples(p_samples[:64], image_size=config.data.image_size, num_channels=config.data.num_channels, fname="samples prior PC")
        # plot_samples_1D(q_samples, image_size=config.data.image_size, fname="samples prior PC 1D", alpha=FG_ALPHA)

        # logging.info(delta_t_corr)  # a value of 0.05 (for 512 samples) are indistinguisable from
        # # true samples due to emprical covariance error
        # # but it is possible to get a value as los as 0.005 from many more true samples
        # logging.info(delta_corr)  # a value of 0.1 are good samples

    index_lo = 0
    index_hi = -1

    num_obs = int(config.data.image_size**2/64)
    idx_obs = random.choice(rng, config.data.image_size**2, shape=(num_obs,), replace=False)
    H = jnp.zeros((num_obs, config.data.image_size**2))
    ogrid = np.arange(num_obs, dtype=int)
    H = H.at[ogrid, idx_obs].set(1.0)
    y = random.normal(rng, idx_obs.shape) * jnp.sqrt(1.0 + config.sampling.noise_std)
    y_data = y.copy()
    X_data = X[idx_obs, :]

    def observation_map(x):
        return H @ x

    def adjoint_observation_map(y):
        return H.T @ y

    if 'plus' in config.sampling.cs_method:
        mask = True
    else:
        mask = None

    if mask is not None:
        mask = jnp.zeros((config.data.image_size * config.data.image_size * config.data.num_channels,))
        mask = mask.at[idx_obs].set(1.0)
        y = jnp.zeros((config.data.image_size * config.data.image_size * config.data.num_channels,))
        y = y.at[idx_obs].set(y_data)

    def prior(prior_parameters):
        lengthscale, signal_variance = prior_parameters
        return signal_variance * Matern52().stretch(lengthscale)

    gaussian_process = GP(data=(X_data, y_data), prior=prior, log_likelihood=log_gaussian_likelihood)
    likelihood_parameters = (config.sampling.noise_std,)
    prior_parameters = (1.0, 1.0)
    parameters = (prior_parameters, likelihood_parameters)
    weight, precision = gaussian_process.approximate_posterior(parameters)
    predictive_mean, predictive_variance = gaussian_process.predict(
        X, parameters, weight, precision)
    predictive_covariance  = gaussian_process.predict_covariance(
        X, parameters, weight, precision)

    plot((X_data[:, 0], y_data), (X[:config.data.image_size, 0], predictive_mean[:config.data.image_size], predictive_variance[:config.data.image_size]), predictive_mean[:config.data.image_size], predictive_variance[:config.data.image_size],
         fname="analytic_1D.png")
    plot_samples(jnp.expand_dims(predictive_mean.reshape(-1, 1), axis=0), image_size=config.data.image_size, num_channels=config.data.num_channels, fname="analytic_target_mean")
    plot_samples(jnp.expand_dims(predictive_variance.reshape(-1, 1), axis=0), image_size=config.data.image_size, num_channels=config.data.num_channels, fname="analytic_target_variance")

    stack_samples = False
    # logspace_max = 4
    # logspace_min = 1
    # batch_sizes = jnp.array([480, 960, 1920, 3840])
    # batch_sizes = jnp.array([4, 16, 25, 36, 49, 64, 100, 121, 144])
    batch_sizes = jnp.array([9, 21, 45, 93, 189, 375, 753, 1500])
    # batch_sizes = jnp.array([16, 25, 36])
    # batch_sizes = jnp.array([config.eval.batch_size])
    num_repeats = 3

    ds_target = np.zeros((batch_sizes.size, num_repeats))
    ds_diffusion = np.zeros((batch_sizes.size, num_repeats))
    ws_target = np.zeros((batch_sizes.size, num_repeats))
    ws_diffusion = np.zeros((batch_sizes.size, num_repeats))
    times_target = np.zeros((batch_sizes.size, num_repeats))
    times_diffusion = np.zeros((batch_sizes.size, num_repeats))
    for j, batch_size in enumerate(batch_sizes):
        logging.info("\nbatch_size={}".format(batch_size))
        sampling_shape = (batch_size//num_devices, config.data.image_size, config.data.image_size, config.data.num_channels)
        sampler = get_cs_sampler(
            config, sde, true_score, sampling_shape,
            config.sampling.inverse_scaler,
            y, num_obs, H,
            observation_map, adjoint_observation_map,
            stack_samples=False)
        if config.eval.pmap:
            sampler = jax.pmap(sampler, axis_name='batch')

        for r in range(num_repeats):
            rng, sample_rng = random.split(rng, 2)
            time0 = time.time()
            p_samples = random.multivariate_normal(sample_rng, mean=predictive_mean,
                cov=predictive_covariance, shape=(batch_size,))
            time_delta = time.time() - time0
            logging.info('target_time={}'.format(time_delta))
            times_target[j, r] = time_delta

            if batch_size > 20 or jnp.isfinite(jnp.cov(p_samples[:, :].T)).all():
                plot_heatmap(samples=p_samples[:, [0, 1]], area_min=-3, area_max=3, fname="analytic_heatmap")
                C_emp = jnp.cov(p_samples[:, :].T)
                m_emp = jnp.mean(p_samples[:, :].T, axis=1)
                delta_cov = jnp.linalg.norm(C - C_emp) / config.data.image_size
                delta_var = jnp.linalg.norm(jnp.diag(C) - jnp.diag(C_emp)) / config.data.image_size
                delta_mean = jnp.linalg.norm(predictive_mean - m_emp) / config.data.image_size
                logging.info("target samples delta_mean={}, delta_var={}, delta_cov={}".format(
                    delta_mean, delta_var, delta_cov))
                plot((X_data[:, 0], y_data), (X[:config.data.image_size, 0], predictive_mean[:config.data.image_size], predictive_variance[:config.data.image_size]), m_emp[:config.data.image_size], jnp.diag(C_emp)[:config.data.image_size],
                    fname="target_empirical_1D.png")
                plot_samples(jnp.expand_dims(m_emp.reshape(-1, 1), axis=0), image_size=config.data.image_size, num_channels=1, fname="target_empirical_mean")
                plot_samples(jnp.expand_dims(jnp.diag(C_emp).reshape(-1, 1), axis=0), image_size=config.data.image_size, num_channels=1, fname="target_empirical_variance")
                tw2 = time.time()
                w2 = Wasserstein2(m_emp, C_emp, predictive_mean, predictive_covariance)  # This may take a while
                tw2delta = time.time() - tw2
                ws_target[j, r] = w2
                td2 = time.time()
                d2 = Distance2(m_emp, C_emp, predictive_mean, predictive_covariance)
                td2delta = time.time() - td2
                ds_target[j, r] = d2
                logging.info('wasserstein [{}] {} seconds'.format(w2, tw2delta))
                logging.info('distance chol [{}] {} seconds'.format(d2, td2delta))

            p_samples = p_samples.reshape((batch_size,) + sampling_shape[1:])
            plot_samples_1D(p_samples, image_size=config.data.image_size, fname="target_samples_1D", alpha=FG_ALPHA)
            plot_samples(p_samples[:int(batch_size**0.5)**2,], image_size=config.data.image_size, num_channels=config.data.num_channels, fname="target_samples")

            if 1:
                if config.eval.pmap:
                    rng, *sample_rng = random.split(rng, 1 + num_devices)
                    sample_rng = jnp.asarray(sample_rng)
                else:
                    rng, sample_rng = random.split(rng, 1 + num_devices)
                time0 = time.time()
                samples, nfe = sampler(sample_rng)
                time_delta = time.time() - time0
                logging.info('diffusion_time={}'.format(time_delta))
                times_diffusion[j, r] = time_delta
                if stack_samples:
                    samples = samples[-1]
                    samples = samples.reshape(batch_size, config.data.image_size**2 * config.data.num_channels)
                else:
                    samples = samples.reshape(batch_size, config.data.image_size**2 * config.data.num_channels)

                C_emp = jnp.cov(samples[:, :].T)
                if jnp.isfinite(C_emp).all():
                    m_emp = jnp.mean(samples[:, :].T, axis=1)
                    delta_cov = jnp.linalg.norm(predictive_covariance - C_emp) / config.data.image_size
                    delta_var = jnp.linalg.norm(jnp.diag(predictive_covariance) - jnp.diag(C_emp)) / config.data.image_size
                    delta_mean = jnp.linalg.norm(predictive_mean - m_emp) / config.data.image_size
                    logging.info("diffusion samples delta_mean={}, delta_var={}, delta_cov={}".format(
                        delta_mean, delta_var, delta_cov))
                    plot((X_data[:, 0], y_data), (X[:config.data.image_size, 0], predictive_mean[:config.data.image_size], predictive_variance[:config.data.image_size]), m_emp[:config.data.image_size], jnp.diag(C_emp)[:config.data.image_size],
                        fname="diffusion_empirical_1D.png")
                    plot_samples(jnp.expand_dims(m_emp.reshape(-1, 1), axis=0), image_size=config.data.image_size, num_channels=config.data.num_channels, fname="diffusion_empirical_mean")
                    plot_samples(jnp.expand_dims(jnp.diag(C_emp).reshape(-1, 1), axis=0), image_size=config.data.image_size, num_channels=config.data.num_channels, fname="diffusion_empirical_variance")
                    tw2 = time.time()
                    # This may take a while
                    w2 = Wasserstein2(m_emp, C_emp, predictive_mean, predictive_covariance)
                    tw2delta = time.time() - tw2
                    ws_diffusion[j, r] = w2
                    td2 = time.time()
                    d2 = Distance2(m_emp, C_emp, predictive_mean, predictive_covariance)
                    td2delta = time.time() - td2
                    ds_diffusion[j, r] = d2
                    logging.info('wasserstein [{}] {} seconds'.format(w2, tw2delta))
                    logging.info('distance chol [{}] {} seconds'.format(d2, td2delta))

                samples = samples.reshape(-1, config.data.image_size, config.data.image_size, config.data.num_channels)
                plot_samples_1D(samples, image_size=config.data.image_size, fname="diffusion_samples_1D", alpha=FG_ALPHA)
                plot_samples(samples[:int(batch_size**0.5)**2], image_size=config.data.image_size, num_channels=config.data.num_channels, fname="diffusion_samples")

        np.savez('./example_{}.npz'.format(config.sampling.cs_method.lower()),
                batch_sizes=batch_sizes,
                times_target=times_target, times_diffusion=times_diffusion,
                ws_diffusion=ws_diffusion,
                ws_target=ws_target,
                ds_diffusion=ds_diffusion,
                ds_target=ds_target)

    logging.info('diffusion_times {}'.format(jnp.mean(times_diffusion, axis=1)))

    if batch_sizes.size > 1:
        times_target_mean = jnp.mean(times_target, axis=1)
        times_diffusion_mean = jnp.mean(times_diffusion, axis=1)
        ws_target_mean = jnp.mean(ws_target, axis=1)
        ws_diffusion_mean = jnp.mean(ws_diffusion, axis=1)
        root_straight_line = jnp.sqrt(batch_sizes)
        root_straight_line = times_diffusion_mean[0] * root_straight_line / root_straight_line[0]
        straight_line = batch_sizes
        straight_line = times_diffusion_mean[0] * straight_line / straight_line[0]
        fig, ax = plt.subplots()
        ax.errorbar(batch_sizes, times_target_mean, jnp.std(times_target, axis=1), label='target sampling')
        ax.plot(batch_sizes, root_straight_line, label=r'$N^{0.5}$')
        ax.plot(batch_sizes, straight_line, label=r'$N$')
        ax.errorbar(batch_sizes, times_diffusion_mean, jnp.std(times_diffusion, axis=1), label='diffusion sampling')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        fig.savefig("profile_time.png")
        plt.close()

        square_straight_line = 1. / batch_sizes**2
        square_straight_line = times_diffusion_mean[-1] * square_straight_line / square_straight_line[-1]
        fig, ax = plt.subplots()
        ax.errorbar(times_target_mean, ws_target_mean, jnp.std(ws_target, axis=1), label='target sampling')
        # ax.plot(batch_sizes, square_straight_line, label=r'$N^{-2}$')
        ax.errorbar(times_diffusion_mean, ws_diffusion_mean, jnp.std(ws_diffusion, axis=1), label='diffusion sampling')
        ax.set_xscale('log')
        ax.set_xlabel('error')
        ax.set_yscale('log')
        ax.set_ylabel('time')
        ax.legend()
        fig.savefig("profile_time_error.png")
        plt.close()

        root_straight_line = 1. / jnp.sqrt(batch_sizes)
        root_straight_line = ws_target_mean[0] * root_straight_line / root_straight_line[0]
        straight_line = 1. / batch_sizes
        straight_line = ws_target_mean[0] * straight_line / straight_line[0]
        fig, ax = plt.subplots()
        ax.errorbar(batch_sizes, ws_target_mean, jnp.std(ws_target, axis=1), label='target sampling')
        ax.plot(batch_sizes, root_straight_line, label=r'$N^{-0.5}$')
        ax.plot(batch_sizes, straight_line, label=r'$N^{-1}$')
        ax.errorbar(batch_sizes, ws_diffusion_mean, jnp.std(ws_diffusion, axis=1), label='diffusion sampling')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        fig.savefig("profile_error_wasserstein2.png")
        plt.close()

        if not np.all(np.isnan(ds_diffusion)):
            ds_target_mean = jnp.mean(ds_target, axis=1)
            ds_diffusion_mean = jnp.mean(ds_diffusion, axis=1)
            root_straight_line = 1. / jnp.sqrt(batch_sizes)
            root_straight_line = ds_target_mean[0] * root_straight_line / root_straight_line[0]
            straight_line = 1. / batch_sizes
            straight_line = ds_target_mean[0] * straight_line / straight_line[0]
            fig, ax = plt.subplots()
            ax.errorbar(batch_sizes, ds_target_mean, jnp.std(ds_target, axis=1), label='target sampling')
            ax.plot(batch_sizes, root_straight_line, label=r'$N^{-0.5}$')
            ax.plot(batch_sizes, straight_line, label=r'$N^{-1}$')
            ax.errorbar(batch_sizes, ds_diffusion_mean, jnp.std(ds_diffusion, axis=1), label='diffusion sampling')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            fig.savefig("profile_error_distance2.png")
            plt.close()


if __name__ == "__main__":
    app.run(main)
