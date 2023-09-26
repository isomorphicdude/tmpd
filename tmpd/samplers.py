"""Samplers."""
from jax import jit, vmap
from diffusionjax.utils import get_sampler
from tmpd.inverse_problems import (
    get_dps,
    get_dps_plus,
    get_diffusion_posterior_sampling,
    get_diffusion_posterior_sampling_plus,
    get_linear_inverse_guidance,
    get_linear_inverse_guidance_plus,
    get_jacrev_approximate_posterior,
    get_jacfwd_approximate_posterior,
    get_vjp_approximate_posterior,
    get_jvp_approximate_posterior,
    get_vjp_approximate_posterior_plus,
    get_diag_approximate_posterior)
from diffusionjax.solvers import EulerMaruyama
from tmpd.solvers import (
    PKF,
    DPSDDPM, DPSDDPMplus,
    KPDDPM, KPSMLD,
    KPDDPMplus, KPSMLDplus,
    DPSSMLD, DPSSMLDplus,
    PiGDMVE, PiGDMVP, PiGDMVPplus, PiGDMVEplus,
    KGDMVE, KGDMVP, KGDMVEplus, KGDMVPplus)


# https://arxiv.org/pdf/2209.14687.pdf#page=20&zoom=100,144,757
dps_scale_hyperparameter = 0.4


def get_cs_sampler(config, sde, model, sampling_shape, inverse_scaler, y, num_y, H, observation_map, adjoint_observation_map, stack_samples=False):
    """Create a sampling function

    Args:
        config: A `ml_collections.ConfigDict` object that contains all configuration information.
        sde: A valid SDE class (the forward sde).
        score:
        shape: The shape of array, x. (num_samples,) + x_shape, where x_shape is the shape
            of the object being sampled from, for example, an image may have
            x_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
        inverse_scaler: The inverse data normalizer function.
        y: the data
        H: an observation matrix.
        operator_map:
        adjoint_operator_map: TODO generalize like this?

    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """
    if config.sampling.cs_method.lower()=='projectionkalmanfilter':
        score = model
        outer_solver = PKF(num_y, y, config.sampling.noise_std, sampling_shape, sde.reverse(score), observation_map, num_steps=config.solver.num_outer_steps)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='chung2022scalar':
        score = model
        scale = config.solver.num_outer_steps * 1.
        posterior_score = jit(vmap(get_dps(
            scale, sde, score, sampling_shape[1:], y, config.sampling.noise_std, H), in_axes=(0, 0), out_axes=(0)))
        sampler = get_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='chung2022scalarplus':
        score = model
        scale = config.solver.num_outer_steps * 1.
        posterior_score = jit(vmap(get_dps_plus(
            scale, sde, score, sampling_shape[1:], y, config.sampling.noise_std, observation_map), in_axes=(0, 0), out_axes=(0)))
        sampler = get_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='chung2022':
        score = model
        posterior_score = jit(vmap(get_diffusion_posterior_sampling(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, observation_map), in_axes=(0, 0), out_axes=(0)))
        sampler = get_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='chung2022plus':
        score = model
        posterior_score = jit(vmap(get_diffusion_posterior_sampling_plus(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, observation_map),
            in_axes=(0, 0), out_axes=(0)))
        sampler = get_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='song2023':
        score = model
        posterior_score = jit(vmap(get_linear_inverse_guidance(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, observation_map, H @ H.T),
            in_axes=(0, 0), out_axes=(0)))
        sampler = get_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='song2023plus':
        score = model
        posterior_score = jit(vmap(get_linear_inverse_guidance_plus(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, observation_map), in_axes=(0, 0), out_axes=(0)))
        sampler = get_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023ajvp':
        score = model
        posterior_score = jit(vmap(get_jvp_approximate_posterior(
                sde, score, sampling_shape[1:], y, config.sampling.noise_std, H),
                in_axes=(0, 0), out_axes=(0)))
        sampler = get_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023avjp':
        score = model
        posterior_score = jit(vmap(get_vjp_approximate_posterior(
                sde, score, sampling_shape[1:], y, config.sampling.noise_std, H),
                in_axes=(0, 0), out_axes=(0)))
        sampler = get_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023ajacfwd':
        score = model
        # NOTE Using full jacobian will be slower in cases with d_y \approx d_x ?
        # TODO: can replace with https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobian-matrix-and-matrix-jacobian-products if faster
        posterior_score = jit(vmap(get_jacfwd_approximate_posterior(
                sde, score, sampling_shape[1:], y, config.sampling.noise_std, observation_map),
                in_axes=(0, 0), out_axes=(0)))
        sampler = get_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023ajacrev':
        score = model
        # NOTE Using full jacobian will be slower in cases with d_y \approx d_x ?
        # TODO: can replace with https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#jacobian-matrix-and-matrix-jacobian-products if faster
        posterior_score = jit(vmap(get_jacrev_approximate_posterior(
                sde, score, sampling_shape[1:], y, config.sampling.noise_std, observation_map),
                in_axes=(0, 0), out_axes=(0)))
        sampler = get_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023ajacrevplus':
        score = model
        posterior_score = jit(vmap(get_jacrev_approximate_posterior(
                sde, score, sampling_shape[1:], y, config.sampling.noise_std, observation_map),
                in_axes=(0, 0), out_axes=(0)))
        sampler = get_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023b':  # This vmaps across calculating N_y vjps, so is O(num_samples * num_y * prod(shape)) in memory
        score = model
        posterior_score = jit(vmap(get_diag_approximate_posterior(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, observation_map), in_axes=(0, 0), out_axes=(0)))
        sampler = get_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023bvjpplus':
        score = model
        posterior_score = jit(vmap(get_vjp_approximate_posterior_plus(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, observation_map), in_axes=(0, 0), out_axes=(0)))
        sampler = get_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpsddpmplus':
        score = model
        # Reproduce DPS (Chung et al. 2022) paper for VP SDE
        # https://arxiv.org/pdf/2209.14687.pdf#page=20&zoom=100,144,757
        # https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/condition_methods.py#L86
        # https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/condition_methods.py#L2[…]C47
        outer_solver = DPSDDPMplus(dps_scale_hyperparameter, y, observation_map, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpsddpm':
        score = model
        # Reproduce DPS (Chung et al. 2022) paper for VP SDE
        outer_solver = DPSDDPM(dps_scale_hyperparameter, y, observation_map, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpssmldplus':
        # Reproduce DPS (Chung et al. 2022) paper for VE SDE
        score = model
        outer_solver = DPSSMLDplus(dps_scale_hyperparameter, y, observation_map, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpssmld':
        # Reproduce DPS (Chung et al. 2022) paper for VE SDE
        score = model
        outer_solver = DPSSMLD(dps_scale_hyperparameter, y, observation_map, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpddpm':
        score = model
        outer_solver = KPDDPM(y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpsmld':
        score = model
        outer_solver = KPSMLD(y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpddpmplus':
        score = model
        outer_solver = KPDDPMplus(y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpsmldplus':
        score = model
        outer_solver = KPSMLDplus(y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='pigdmvp':
        # Reproduce PiGDM (Chung et al. 2022) paper for VP SDE
        outer_solver = PiGDMVP(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='pigdmve':
        # Reproduce PiGDM (Chung et al. 2022) paper for VE SDE
        outer_solver = PiGDMVE(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='pigdmvpplus':
        # Reproduce PiGDM (Chung et al. 2022) paper for VP SDE
        outer_solver = PiGDMVPplus(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='pigdmveplus':
        # Reproduce PiGDM (Chung et al. 2022) paper for VE SDE
        outer_solver = PiGDMVEplus(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kgdmvp':
        outer_solver = KGDMVP(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kgdmve':
        outer_solver = KGDMVE(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kgdmvpplus':
        outer_solver = KGDMVPplus(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kgdmveplus':
        outer_solver = KGDMVEplus(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='wc4dvar':
        # Get prior samples via a method
        raise NotImplementedError("TODO")
    else:
        raise ValueError("`config.sampling.cs_method` not recognized, got {}".format(config.sampling.cs_method))
    return sampler
