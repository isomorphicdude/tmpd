"""Samplers."""
from jax import jit, vmap
from diffusionjax.utils import get_sampler as get_markov_sampler
from grfjax.inverse_problems import (
    get_dps,
    get_dps_plus,
    get_diffusion_posterior_sampling,
    get_diffusion_posterior_sampling_plus,
    get_linear_inverse_guidance,
    get_linear_inverse_guidance_plus,
    get_jac_approximate_posterior,
    get_jacrev_approximate_posterior,
    get_jacfwd_approximate_posterior,
    get_vjp_approximate_posterior,
    get_vjp_approximate_posterior_plus,
    get_jvp_approximate_posterior,
    get_jvp_approximate_posterior_plus,
    get_experimental_diag_approximate_posterior,
    get_experimental_diag_approximate_posterior_plus,
    get_diag_approximate_posterior)
from grfjax.ensemble_kalman import (
    ProjectionChol,
    AdjustmentChol,
    StochasticChol,
    StochasticSVD)
from grfjax.ensemble_kalman import get_ensemble_sampler
from diffusionjax.solvers import EulerMaruyama
from grfjax.solvers import (
    PKF,
    DPKF,
    DPSDDPM, DPSDDPMplus,
    KPDDPM, KPSMLD,
    KPDDPMplus, KPSMLDplus,
    DPSSMLD, DPSSMLDplus,
    PiGDMVE, PiGDMVP, PiGDMVPplus, PiGDMVEplus,
    KGDMVE, KGDMVP, KGDMVEplus, KGDMVPplus)


def get_cs_sampler(config, sde, model, sampling_shape, inverse_scaler, y, H, mask, observation_map, adjoint_observation_map, stack_samples=False):
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
        H: operator
        mask: another form of the operator TODO fix this
        operator_map, adjoint_operator_map: TODO generalize like this?

    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    if config.sampling.cs_method.lower()=='stochasticsvd':  # Currently scales with O(J^{3})
        score = model
        kalman_filter = StochasticSVD(H.shape[0], sampling_shape, sde.reverse(score), observation_map, innovation=config.sampling.innovation, num_steps=config.solver.num_outer_steps)
        sampler = get_ensemble_sampler(sampling_shape, kalman_filter, y, config.sampling.noise_std, denoise=config.sampling.denoise, stack_samples=stack_samples, inverse_scaler=inverse_scaler)
    elif config.sampling.cs_method.lower()=='stochasticchol':  # Currently scales approximately same as other methods
        score = model
        kalman_filter = StochasticChol(H.shape[0], sampling_shape, sde.reverse(score), observation_map, innovation=config.sampling.innovation, num_steps=config.solver.num_outer_steps)
        sampler = get_ensemble_sampler(sampling_shape, kalman_filter, y, config.sampling.noise_std, denoise=config.sampling.denoise, stack_samples=stack_samples, inverse_scaler=inverse_scaler)
    elif config.sampling.cs_method.lower()=='adjustmentsvd':  # TODO: unstable and incorrect for num_y < num_ensemble
        return ValueError()
    elif config.sampling.cs_method.lower()=='adjustmentchol':
        score = model
        kalman_filter = AdjustmentChol(H.shape[0], sampling_shape, sde.reverse(score), observation_map, innovation=config.sampling.innovation, num_steps=config.solver.num_outer_steps)
        sampler = get_ensemble_sampler(sampling_shape, kalman_filter, y, config.sampling.noise_std, denoise=config.sampling.denoise, stack_samples=stack_samples, inverse_scaler=inverse_scaler)
    elif config.sampling.cs_method.lower()=='TransformSVD':  # TODO: stable but incorrect for num_y < num_ensemble
        return ValueError()
    elif config.sampling.cs_method.lower()=='TranformChol':  # TODO: is a derivation possible for num_y < num_ensemble?
        return ValueError()
    elif config.sampling.cs_method.lower()=='projectionsvd':  # Currently scales approximately same as other methods
        score = model
        kalman_filter = ProjectionChol(H.shape[0], sampling_shape, sde.reverse(score), observation_map, innovation=config.sampling.innovation, num_steps=config.solver.num_outer_steps)
        sampler = get_ensemble_sampler(sampling_shape, kalman_filter, y, config.sampling.noise_std, denoise=config.sampling.denoise, stack_samples=stack_samples, inverse_scaler=inverse_scaler)
    elif config.sampling.cs_method.lower()=='projectionchol':  # Currently scales approximately same as other methods
        score = model
        kalman_filter = ProjectionChol(H.shape[0], sampling_shape, sde.reverse(score), observation_map, innovation=config.sampling.innovation, num_steps=config.solver.num_outer_steps)
        sampler = get_ensemble_sampler(sampling_shape, kalman_filter, y, config.sampling.noise_std, denoise=config.sampling.denoise, stack_samples=stack_samples, inverse_scaler=inverse_scaler)
    elif config.sampling.cs_method.lower()=='projectionkalmanfilter':
        score = model
        projection_filter = PKF(H.shape[0], sampling_shape, sde.reverse(score), observation_map, H, num_steps=config.solver.num_outer_steps)
        sampler = get_ensemble_sampler(sampling_shape, projection_filter, y, config.sampling.noise_std, denoise=config.sampling.denoise, stack_samples=stack_samples, inverse_scaler=inverse_scaler)
    elif config.sampling.cs_method.lower()=='approxprojectionkalmanfilter':
        score = model
        projection_filter = DPKF(H.shape[0], sampling_shape, sde.reverse(score), observation_map, H, num_steps=config.solver.num_outer_steps)
        sampler = get_ensemble_sampler(sampling_shape, projection_filter, y, config.sampling.noise_std, denoise=config.sampling.denoise, stack_samples=stack_samples, inverse_scaler=inverse_scaler)
    elif config.sampling.cs_method.lower()=='chung2022scalar':
        score = model
        scale = config.solver.num_outer_steps * 1.
        posterior_score = jit(vmap(get_dps(
            scale, sde, score, sampling_shape[1:], y, config.sampling.noise_std, H), in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='chung2022scalarplus':
        score = model
        scale = config.solver.num_outer_steps * 1.
        posterior_score = jit(vmap(get_dps_plus(
            scale, sde, score, sampling_shape[1:], y, config.sampling.noise_std, mask), in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='chung2022':
        score = model
        posterior_score = jit(vmap(get_diffusion_posterior_sampling(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, H), in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='chung2022plus':
        score = model
        posterior_score = jit(vmap(get_diffusion_posterior_sampling_plus(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, mask),
            in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='song2023':
        score = model
        posterior_score = jit(vmap(get_linear_inverse_guidance(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, H),
            in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='song2023plus':
        score = model
        posterior_score = jit(vmap(get_linear_inverse_guidance_plus(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, mask), in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023avjp':  # This vmaps across calculating full jacobian, so is O(num_samples * prod(shape)**2) in memory, which is prohibitive
        score = model
        posterior_score = jit(vmap(get_vjp_approximate_posterior(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, H),
            in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023ajvp':  # This vmaps across calculating full jacobian, so is O(num_samples * prod(shape)**2) in memory, which is prohibitive
        score = model
        posterior_score = jit(vmap(get_jvp_approximate_posterior(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, H),
            in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023ajacfwd':
        score = model
        # NOTE Using full jacobian will be slower in cases with d_y \approx d_x ?
        posterior_score = jit(vmap(get_jacfwd_approximate_posterior(
                sde, score, sampling_shape[1:], y, config.sampling.noise_std, H),
                in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023ajacrev':
        score = model
        # NOTE Using full jacobian will be slower in cases with d_y \approx d_x ?
        posterior_score = jit(vmap(get_jacrev_approximate_posterior(
                sde, score, sampling_shape[1:], y, config.sampling.noise_std, H, mask),
                in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023ajac':  # This vmaps across calculating full jacobian, so is O(num_samples * prod(shape)**2) in memory, which is prohibitive
        score = model
        posterior_score = jit(vmap(get_jac_approximate_posterior(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, H), in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023b':  # This vmaps across calculating N_y vjps, so is O(num_samples * num_y * prod(shape)) in memory
        score = model
        posterior_score = jit(vmap(get_diag_approximate_posterior(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, H), in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    # elif config.sampling.cs_method.lower()=='boys2023bplus':  # This vmaps across calculating N_y vjps, so is O(num_samples * num_y * prod(shape)) in memory
    #     posterior_score = jit(vmap(get_diag_approximate_posterior_plus(
    #         sde, score, sampling_shape[1:], y, config.sampling.noise_std, H, mask), in_axes=(0, 0), out_axes=(0)))
    #     sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023bvjpplus':
        score = model
        posterior_score = jit(vmap(get_vjp_approximate_posterior_plus(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, mask), in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023bjvpplus':
        score = model
        posterior_score = jit(vmap(get_jvp_approximate_posterior_plus(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, mask), in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023c':
        score = model
        # This method is experimental, it involves taking the gradient of a scalar product between mask and mmse estimate of the denoise sample
        posterior_score = jit(vmap(get_experimental_diag_approximate_posterior(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, H), in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='boys2023cplus':
        score = model
        # This method is experimental, it involves taking the gradient of a scalar product between mask and mmse estimate of the denoise sample
        posterior_score = jit(vmap(get_experimental_diag_approximate_posterior_plus(
            sde, score, sampling_shape[1:], y, config.sampling.noise_std, mask), in_axes=(0, 0), out_axes=(0)))
        sampler = get_markov_sampler(sampling_shape, EulerMaruyama(sde.reverse(posterior_score)), inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpsddpmplus':
        score = model
        scale = 1.0
        # Designed to replicate dps paper as closely as possible
        outer_solver = DPSDDPMplus(scale, y, mask, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpsddpm':
        score = model
        scale = 1.0
        # Designed to replicate dps paper as closely as possible
        outer_solver = DPSDDPM(scale, y, H, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpssmldplus':
        score = model
        scale = 1.0
        outer_solver = DPSSMLDplus(scale, y, mask, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpssmld':
        score = model
        scale = 1.0
        outer_solver = DPSSMLD(scale, y, H, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpddpm':
        score = model
        outer_solver = KPDDPM(y, H, config.sampling.noise_std, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpsmld':
        score = model
        outer_solver = KPSMLD(y, H, config.sampling.noise_std, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpddpmplus':
        score = model
        outer_solver = KPDDPMplus(y, mask, config.sampling.noise_std, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpsmldplus':
        score = model
        outer_solver = KPSMLDplus(y, mask, config.sampling.noise_std, sampling_shape[1:], score, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='pigdmvp':
        outer_solver = PiGDMVP(y, H, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='pigdmve':
        outer_solver = PiGDMVE(y, H, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='pigdmvpplus':
        outer_solver = PiGDMVPplus(y, mask, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='pigdmveplus':
        outer_solver = PiGDMVEplus(y, mask, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kgdmvp':
        outer_solver = KGDMVP(y, H, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kgdmve':
        outer_solver = KGDMVE(y, H, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kgdmvpplus':
        outer_solver = KGDMVPplus(y, mask, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kgdmveplus':
        outer_solver = KGDMVEplus(y, mask, config.sampling.noise_std, sampling_shape[1:], model, num_steps=config.solver.num_outer_steps,
                            dt=config.solver.dt, epsilon=config.solver.epsilon,
                            sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max)
        sampler = get_markov_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='wc4dvar':
        # Get prior samples via a method
        raise NotImplementedError("TODO")
    else:
        raise ValueError("`config.sampling.cs_method` not recognized, got {}".format(config.sampling.cs_method))
    return sampler
