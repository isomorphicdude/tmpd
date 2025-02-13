"""Samplers."""
from diffusionjax.utils import get_sampler
from diffusionjax.inverse_problems import (
    get_dps,
    get_diffusion_posterior_sampling,
    get_pseudo_inverse_guidance,
    get_jacrev_guidance,
    get_jacfwd_guidance,
    get_vjp_guidance_mask,
    get_vjp_guidance,
    get_diag_jacrev_guidance,
    get_diag_vjp_guidance,
    get_diag_jacfwd_guidance)
from diffusionjax.solvers import EulerMaruyama
from tmpd.solvers import (
    MPGD,
    DPSDDPM, DPSDDPMplus,
    KPDDPM, KPSMLD,
    KPSMLDdiag, KPDDPMdiag,
    KPDDPMplus, KPSMLDplus,
    DPSSMLD, DPSSMLDplus,
    PiGDMVE, PiGDMVP, PiGDMVPplus, PiGDMVEplus,
    KGDMVE, KGDMVP, KGDMVEplus, KGDMVPplus)


def get_cs_sampler(config, sde, model, sampling_shape, inverse_scaler, y, H, observation_map, adjoint_observation_map, stack_samples=False):
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
    if config.sampling.cs_method.lower()=='chung2022scalar' or config.sampling.cs_method.lower()=='chung2022scalarplus':
        scale = config.solver.num_outer_steps * config.solver.dps_scale_hyperparameter
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_dps, observation_map, y, scale)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='chung2022' or config.sampling.cs_method.lower()=='chung2022plus':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_diffusion_posterior_sampling, observation_map, y, config.sampling.noise_std)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='song2023':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_pseudo_inverse_guidance, observation_map, y, config.sampling.noise_std, H @ H.T)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='song2023plus':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_pseudo_inverse_guidance, observation_map, y, config.sampling.noise_std)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023avjp':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                get_vjp_guidance, H, y, config.sampling.noise_std, sampling_shape)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023ajacfwd':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_jacfwd_guidance, observation_map, y, config.sampling.noise_std, sampling_shape)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023ajacrev' or config.sampling.cs_method.lower()=='tmpd2023ajacrevplus':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_jacrev_guidance, observation_map, y, config.sampling.noise_std, sampling_shape)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023b':  # This vmaps across calculating N_y vjps, so is O(num_samples * num_y * prod(shape)) in memory
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_diag_jacrev_guidance, observation_map, y, config.sampling.noise_std, sampling_shape)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023bjacfwd':  # This vmaps across calculating N_y vjps, so is O(num_samples * num_y * prod(shape)) in memory
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_diag_jacfwd_guidance, observation_map, y, config.sampling.noise_std, sampling_shape)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023bvjp':  # This vmaps across calculating N_y vjps, so is O(num_samples * num_y * prod(shape)) in memory
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_diag_vjp_guidance, H, y, config.sampling.noise_std, sampling_shape)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023bvjpplus':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_vjp_guidance_mask, observation_map, y, config.sampling.noise_std)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='dpsddpmplus':
        score = model
        # Reproduce DPS (Chung et al. 2022) paper for VP SDE
        # https://arxiv.org/pdf/2209.14687.pdf#page=20&zoom=100,144,757
        # https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/condition_methods.py#L86
        # https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/condition_methods.py#L2[…]C47
        outer_solver = DPSDDPMplus(config.solver.dps_scale_hyperparameter, y, observation_map, score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='mpgd':
        # Reproduce MPGD (et al. 2023) paper for VP SDE
        outer_solver = MPGD(config.solver.mpgd_scale_hyperparameter, y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpsddpm':
        score = model
        # Reproduce DPS (Chung et al. 2022) paper for VP SDE
        outer_solver = DPSDDPM(config.solver.dps_scale_hyperparameter, y, observation_map, score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpssmldplus':
        # Reproduce DPS (Chung et al. 2022) paper for VE SDE
        score = model
        outer_solver = DPSSMLDplus(config.solver.dps_scale_hyperparameter, y, observation_map, score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpssmld':
        # Reproduce DPS (Chung et al. 2022) paper for VE SDE
        score = model
        outer_solver = DPSSMLD(config.solver.dps_scale_hyperparameter, y, observation_map, score, num_steps=config.solver.num_outer_steps,
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
    elif config.sampling.cs_method.lower()=='kpsmlddiag':
        score = model
        outer_solver = KPSMLDdiag(y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpddpmdiag':
        score = model
        outer_solver = KPDDPMdiag(y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
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
        outer_solver = PiGDMVP(y, observation_map, config.sampling.noise_std, sampling_shape[:1], model, num_steps=config.solver.num_outer_steps,
                               dt=config.solver.dt, epsilon=config.solver.epsilon,
                               data_variance=1.0,
                               beta_min=config.model.beta_min,
                               beta_max=config.model.beta_max)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='pigdmve':
        # Reproduce PiGDM (Chung et al. 2022) paper for VE SDE
        outer_solver = PiGDMVE(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                               dt=config.solver.dt, epsilon=config.solver.epsilon,
                               data_variance=1.0,
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
