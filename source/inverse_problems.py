"""Inverse problems."""
from jax import vmap, grad, jacfwd, vjp, jacrev, jvp, jit, value_and_grad
import jax.numpy as jnp
from jax.lax import scan
import jax.random as random
from diffusionjax.utils import batch_mul, get_score
import numpy as np


def get_estimate_h_x_0(sde, score, shape, H):
    """Get an MMSE estimate of x_0, pushed through forward linear map.
    """
    def estimate_x_0(x, t):
        """The MMSE estimate for x_0|x_t,
        which is it's expectation as given by Tweedie's formula."""
        m_t = sde.mean_coeff(t)
        v_t = sde.variance(t)
        x = x.reshape(shape)
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        s = score(x, t)
        s = s.flatten()
        x = x.flatten()
        return H @ (x + v_t * s) / m_t, s

    # def estimate_x_0_(x, t):
    #     """The MMSE estimate for x_0|x_t,
    #     which is it's expectation as given by Tweedie's formula."""
    #     m_t = sde.sde.mean_coeff(t)
    #     v_t = sde.sde.variance(t)
    #     x = x.flatten()
    #     s = score(x, t)
    #     return H @ (x + v_t * s) / m_t, s

    return estimate_x_0


def get_estimate_mask_x_0(sde, score, shape, mask):
    """Get an MMSE estimate of x_0, pushed through forward linear map.
    """
    def estimate_h_x_0(x, t):
        m_t = sde.mean_coeff(t)
        v_t = sde.variance(t)
        x = x.reshape(shape)
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        s = score(x, t)
        s = s.flatten()
        x = x.flatten()
        return mask * (x + v_t * s) / m_t, s

    return estimate_h_x_0


def get_ratio(sde):
    if type(sde).__name__=='VE':
        def ratio(t):
            return sde.variance(t)
    elif type(sde).__name__=='VP':
        def ratio(t):
            return sde.variance(t) / sde.mean_coeff(t)
    else:
        raise ValueError("Did not recognise forward SDE (got {}, expected VE or VP)".format(type(sde).__name__))
    return ratio


def get_model_variance(sde):
    if type(sde).__name__=='VE':
        def model_variance(t):
            # return sde.variance(t) / (1 + sde.variance(t))  # This was unstable for Song et al. 2023
            return sde.variance(t)  # used this instead, as it is stable and a free hyperparameter in Song's method
    elif type(sde).__name__=='VP':
        def model_variance(t):
            return sde.variance(t)
    else:
        raise ValueError("Did not recognise SDE (got {}, expected VE or VP)".format(type(sde).__name__))
    return model_variance


def get_batch_estimate_x_0(sde, score, shape, flag=0):
    """Get an MMSE estimate of x_0 over a batch. This cannot be
    vector jacobian product, because it will return a batch, and
    would require a vjp for each batch member. So how to calculate Kalman gain?
    Maybe not possible to batch, but would rather pmap across GPU?

    # TODO: vjp can be evaluated against an array of any shape?
    """
    # TODO: problem is that score is already vmapped, does this cause any inefficiencies?
    def estimate_x_0(x, t):
        """The MMSE estimate for x_0|x_t,
        which is it's expectation as given by Tweedie's formula."""
        m_t = sde.mean_coeff(t)
        v_t = sde.variance(t)
        x = x.reshape(shape)
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        s = score(x, t)
        s = s.flatten()
        x = x.flatten()
        return (x + v_t * s) / m_t, s

    # def estimate_x_0_(x, t):
    #     """The MMSE estimate for x_0|x_t,
    #     which is it's expectation as given by Tweedie's formula."""
    #     m_t = sde.mean_coeff(t)
    #     v_t = sde.variance(t)
    #     x = x.flatten()
    #     s = score(x, t)
    #     return (x + v_t * s) / m_t, s

    return estimate_x_0


def get_scalar_estimate_H_x_0(H, sde, score, shape):
    # TODO: SS, it's probably wrong and not very general?
    def estimate_x_0(x, t):
        """The MMSE estimate for x_0|x_t,
        which is it's expectation as given by Tweedie's formula."""
        m_t = sde.mean_coeff(t)
        v_t = sde.variance(t)
        x = x.reshape(shape)
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        s = score(x, t)
        s = s.flatten()
        x = x.flatten()
        return jnp.sum(H @ (x + v_t * s) / m_t)

    return estimate_x_0


def get_scalar_estimate_mask_x_0(mask, sde, score, shape):
    # TODO: SS, it's probably wrong or not very general?
    def estimate_x_0(x, t):
        """The MMSE estimate for x_0|x_t,
        which is it's expectation as given by Tweedie's formula."""
        m_t = sde.mean_coeff(t)
        v_t = sde.variance(t)
        x = x.reshape(shape)
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        s = score(x, t)
        s = s.flatten()
        x = x.flatten()
        return jnp.dot((x + v_t * s) / m_t, mask)

    return estimate_x_0


def get_estimate_x_0(sde, score, shape):
    """Get an MMSE estimate of x_0
    """
    # TODO: problem is that score is already vmapped, does this cause any inefficiencies?
    # TODO: this can be a method within rsde class? no because it needs to reshape. So it could be
    # a method within the sampler class. This is probably the cleanest solution for now
    # since if it was in the sampler class, then sampler needs to access sde and is not general to
    # markov chains.

    def estimate_x_0(x, t):
        """The MMSE estimate for x_0|x_t,
        which is it's expectation as given by Tweedie's formula."""
        m_t = sde.mean_coeff(t)
        v_t = sde.variance(t)
        x = x.reshape(shape)
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        s = score(x, t)
        s = s.flatten()
        x = x.flatten()
        return (x + v_t * s) / m_t, s

    # def estimate_x_0(x, t):
    #     """The MMSE estimate for x_0|x_t,
    #     which is it's expectation as given by Tweedie's formula."""
    #     m_t = sde.sde.mean_coeff(t)
    #     v_t = sde.sde.variance(t)
    #     x = x.flatten()
    #     s = score(x, t)
    #     return (x + v_t * s) / m_t, s

    return estimate_x_0


def get_estimate_x_n(solver, rng, t):
    return lambda x, t: solver.update(rng, x, t)


def get_dps(
        scale, sde, score, shape, y, noise_std, H):
    """
    `Diffusion Posterior Sampling for general noisy inverse problems'
    implemented with a vmap grad.
    """

    def get_l2_norm(y, estimate_x_0, H):
        assert jnp.shape(y)[0] == jnp.shape(H)[0]
        def likelihood_score_approx(x, t):
            # y is given as a d_y length vector
            x_0, s = estimate_x_0(x, t)
            innovation = y - H @ x_0
            return jnp.linalg.norm(innovation), s
        return likelihood_score_approx

    estimate_x_0 = get_estimate_x_0(sde, score, shape)
    l2_norm = get_l2_norm(y, estimate_x_0, H)
    likelihood_score = grad(l2_norm, has_aux=True)

    def approx_posterior_score(x, t):
        x = x.flatten()
        ls, s = likelihood_score(x, t)
        posterior_score = s - scale * ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_dps_plus(
        scale, sde, score, shape, y, noise_std, mask):
    """
    TODO: Unstable(?) and doesn't work on FFHQ_noise_std=0.001
    TODO: Unstable and doesn't work - weird oversmoothing of images
    `Diffusion Posterior Sampling for general noisy inverse problems'
    implemented with a vmap grad.
    """

    num_steps = 1000.
    def get_l2_norm(y, estimate_x_0, mask):
        # y = y.flatten()
        # mask = mask.flatten()
        assert jnp.shape(y) == jnp.shape(mask)
        def likelihood_score_approx(x, t):
            # y is given as a d_x length vector
            x_0, s = estimate_x_0(x, t)
            innovation = (y - x_0) * mask
            return jnp.linalg.norm(innovation), s
        return likelihood_score_approx

    estimate_x_0 = get_estimate_x_0(sde, score, shape)
    l2_norm = get_l2_norm(y, estimate_x_0, mask)
    likelihood_score = grad(l2_norm, has_aux=True)

    def approx_posterior_score(x, t):
        x = x.flatten()
        ls, s = likelihood_score(x, t)
        posterior_score = s - scale * ls  # multiply by dt = 1. / num_steps, should get back to DPS
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_diffusion_posterior_sampling_plus(
        sde, score, shape, y, noise_std, mask):
    """
    TODO: Unstable on FFHQ_noise_std=0.001
    `Diffusion Posterior Sampling for general noisy inverse problems'
    implemented with a single vjp.
    NOTE: This is not how they implemented their method, their method is get_dps_plus.
    TODO: Generalizable for non-linear forward map?
    """

    def get_likelihood_score(y, mask, noise_std):
        y = y.flatten()
        mask = mask.flatten()
        # y is given as a full length vector
        assert jnp.shape(y) == jnp.shape(mask)
        def likelihood_score_approx(x_0):  # No linear solve required
            innovation = (y - x_0) * mask
            Cyy = noise_std
            return innovation / Cyy
        return likelihood_score_approx

    estimate_x_0 = get_estimate_x_0(sde, score, shape)
    likelihood_score = get_likelihood_score(y, mask, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        x_0, vjp_estimate_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        ls = likelihood_score(x_0)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_diffusion_posterior_sampling(
        sde, score, shape, y, noise_std, H):
    """
    `Diffusion Posterior Sampling for general noisy inverse problems'
    implemented with a single vjp.
    TODO: Generalizable for non-linear forward map?
    """

    def get_likelihood_score(y, H, noise_std):
        # y is given as just observations, H a linear operator
        assert jnp.shape(y)[0] == jnp.shape(H)[0]
        def likelihood_score_approx(x_0):
            innovation = y - H @ x_0
            Cyy = noise_std**2
            f = innovation / Cyy
            return H.T @ f
        return likelihood_score_approx

    estimate_x_0 = get_estimate_x_0(sde, score, shape)
    likelihood_score = get_likelihood_score(y, H, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        x_0, vjp_estimate_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        ls = likelihood_score(x_0)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_pseudo_inverse_guidance(
        sde, score, shape, y, h, h_dagger):
    """
    Pseudo-inverse guidance score for non-linear forward map h, with pseudo-inverse h_dagger.
    :arg shape:
    :arg y:
    :arg h:
    :arg h_dagger:
    :arg mask:
    :arg likelihood_score:
    :arg estimate_x_0:
    """
    def ratio(t):
        return sde.variance(t) / sde.mean_coeff(t)

    def get_likelihood_score():
        y_hat = h(jnp.zeros(shape))
        assert jnp.shape(y) == jnp.shape(y_hat)
        def likelihood_score_approx(x_0, t):
            innovation = h_dagger(y) - h_dagger(h(x_0))
            return innovation

        return likelihood_score_approx

    estimate_x_0 = get_estimate_x_0(sde, score, shape)
    likelihood_score = get_likelihood_score()

    def approx_posterior_score(x, t):
        x = x.flatten()
        x_0, vjp_estimate_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        ls = likelihood_score(x_0, t)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls / ratio(t)
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_linear_inverse_guidance_plus(
        sde, score, shape, y, noise_std, mask):
    """
    Pseudo-Inverse guidance score for a forward map that can be
    represented by a mask.
    """
    model_variance = get_model_variance(sde)

    def get_likelihood_score(y, mask, noise_std):
        y = y.flatten()
        mask = mask.flatten()
        assert jnp.shape(y) == jnp.shape(mask)
        def likelihood_score(x_0, t):
            innovation = (y - x_0) * mask
            Cyy = model_variance(t) + noise_std**2
            return innovation / Cyy
        return likelihood_score

    estimate_x_0 = get_estimate_x_0(sde, score, shape)
    likelihood_score = get_likelihood_score(y, mask, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        x_0, vjp_estimate_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        ls = likelihood_score(x_0, t)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_linear_inverse_guidance(
        sde, score, shape, y, noise_std, H):
    """Pseudo-Inverse guidance score for linear forward map H.
    """

    model_variance = get_model_variance(sde)

    def get_likelihood_score(y, H, noise_std):
        # y is given as just observations, H a linear operator
        assert jnp.shape(y)[0] == jnp.shape(H)[0]
        def likelihood_score_approx(x_0, t):
            innovation = y - H @ x_0
            Cyy = model_variance(t) * H @ H.T + noise_std**2 * jnp.eye(y.shape[0])
            f = jnp.linalg.solve(Cyy, innovation)
            return H.T @ f
        return likelihood_score_approx

    estimate_x_0 = get_estimate_x_0(sde, score, shape)
    likelihood_score = get_likelihood_score(y, H, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        x_0, vjp_estimate_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        ls = likelihood_score(x_0, t)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_experimental_diag_approximate_posterior(
        sde, score, shape, y, noise_std, H):
    """Use a diagonal approximation to the variance inside the likelihood,
    This produces similar results when the covariance is approximately diagonal
    """
    ratio = get_ratio(sde)
    mask_estimate_x_0 = get_scalar_estimate_H_x_0(H, sde, score, shape)
    estimate_x_0 = get_estimate_x_0(sde, score, shape)

    def get_likelihood_score(y, H, noise_std):
        # use a diagonal approximation to the variance inside the likelihood
        # Not correct unless Sigma_d_r_t is diagonal
        assert jnp.shape(y)[0] == jnp.shape(H)[0]
        def likelihood_score(grad_mask_estimate_x_0, x_0, t):
            # TODO this doesn't seem to work?
            # It actually projects onto the data
            # this is due to it being the wrong calculation
            # Sigma_diag = grad_mask_estimate_x_0 * ratio(t)  # TODO: I think this should be postmultiplied by mask, like below
            Sigma_diag = H @ grad_mask_estimate_x_0 * ratio(t)
            Cyy = Sigma_diag + noise_std**2
            innovation = y - H @ x_0
            f = innovation / Cyy
            return H.T @ f  # remember to take vjp_estimate_x_0(f)[0]
        return likelihood_score

    likelihood_score = get_likelihood_score(y, H, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        # Take the gradient of the scalar product between mask and MMSE estimate wrt to its inputs, x
        # this will result in a vector which can be used to weight the error covariance
        grad_mask_estimate_x_0 = grad(lambda x: mask_estimate_x_0(x, t))(x)
        x_0, vjp_estimate_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        ls = likelihood_score(grad_mask_estimate_x_0, x_0, t)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_experimental_diag_approximate_posterior_plus(
        sde, score, shape, y, noise_std, mask):
    """
    TODO: Unstable on FFHQ_noise_std=0.001
    Use a diagonal approximation to the variance inside the likelihood,
    This produces similar results when the covariance is approximately diagonal
    """
    ratio = get_ratio(sde)
    mask_estimate_x_0 = get_scalar_estimate_mask_x_0(mask, sde, score, shape)
    estimate_x_0 = get_estimate_x_0(sde, score, shape)

    def get_likelihood_score(y, mask, noise_std):
        # posterior_score 1 - Using multivariate Gaussian, with exploring calculation of vjp with H or mask
        y = y.flatten()
        mask = mask.flatten()
        assert jnp.shape(y) == jnp.shape(mask)
        # use a diagonal approximation to the variance inside the likelihood
        # Not correct unless Sigma_d_r_t is diagonal
        def likelihood_score(grad_mask_estimate_x_0, x_0, t):
            # TODO this doesn't seem to work?
            # It actually projects onto the data
            # this is due to it being the wrong calculation
            # Sigma_diag = grad_mask_estimate_x_0 * ratio(t)  # TODO: I think this should be postmultiplied by mask, like below
            Sigma_diag = grad_mask_estimate_x_0 * ratio(t) * mask
            Cyy = Sigma_diag + noise_std**2
            innovation = y - mask * x_0
            f = innovation / Cyy
            return f  # remember to take vjp_estimate_x_0(f)[0]
        return likelihood_score

    likelihood_score = get_likelihood_score(y, mask, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        # Take the gradient of the scalar product between mask and MMSE estimate wrt to its inputs, x
        # this will result in a vector which can be used to weight the error covariance
        grad_mask_estimate_x_0 = grad(lambda x: mask_estimate_x_0(x, t))(x)
        x_0, vjp_estimate_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        ls = likelihood_score(grad_mask_estimate_x_0, x_0, t)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_diag_approximate_posterior(sde, score, shape, y, noise_std, H):
    """Use a diagonal approximation to the variance inside the likelihood,
    This produces similar results when the covariance is approximately diagonal
    """
    ratio = get_ratio(sde)

    def get_likelihood_score(y, H, noise_std):
        # posterior_score 1 - Using multivariate Gaussian, with exploring calculation of vjp with H or mask
        assert jnp.shape(y)[0] == jnp.shape(H)[0]
        def likelihood_score(vjp_estimate_x_0, x_0, t):
            # You can compute the diagonal of the hessian by using vmap of grad
            vec_vjp_fn = vmap(vjp_estimate_x_0)
            H_grad_x_hat = vec_vjp_fn(H)[0]
            Cyy = ratio(t) * H @ H_grad_x_hat.T + noise_std**2 * jnp.eye(y.shape[0])
            Cyy = jnp.diag(Cyy)
            innovation = y - H @ x_0
            f = innovation / Cyy
            f = H.T @ f
            return f  # remember to take vjp_estimate_x_0(f)[0]
        return likelihood_score

    estimate_x_0 = get_estimate_x_0(sde, score, shape)
    likelihood_score = get_likelihood_score(y, H, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        x_0, vjp_estimate_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        ls = likelihood_score(vjp_estimate_x_0, x_0, t)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_jac_approximate_posterior(sde, score, shape, y, noise_std, H):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using the full Jacobian, computed using reverse mode.
    In early tests, this was around two times slower
    than vjp for a sparse observation inpainting operator (d_x=64, d_y=2).
    vjp should have scaling with d_x.
    """
    ratio = get_ratio(sde)

    def get_likelihood_score(y, mask, noise_std):
        # posterior_score_0 - Using multivariate Gaussian, with exploring calculation of a full Jacobian, not desireable in general
        # TODO: see if I can replace if statement with an observation_map function?
        assert jnp.shape(y)[0] == jnp.shape(H)[0]
        def likelihood_score(_estimate_x_0, x, x_0, t):
            inv_hessian = jacrev(_estimate_x_0)  #TODO: factor this line out!
            x = x.flatten()
            Sigma_d_r_t = inv_hessian(x)
            Sigma = Sigma_d_r_t * ratio(t)
            Cyy = H @ Sigma @ H.T + noise_std**2 * jnp.eye(y.shape[0])
            innovation = y - H @ x_0
            f = jnp.linalg.solve(Cyy, innovation)
            f = H.T @ f
            return f
        return likelihood_score

    estimate_x_0 = get_estimate_x_0(sde, score, shape)
    likelihood_score = get_likelihood_score(y, H, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        _estimate_x_0 = lambda _x: estimate_x_0(_x, t)
        __estimate_x_0 = lambda _x: _estimate_x_0(_x)[0]
        x_0, vjp_estimate_x_0, s = vjp(
            _estimate_x_0, x, has_aux=True)
        ls = likelihood_score(__estimate_x_0, x, x_0, t)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_jac_approximate_posterior_plus(sde, score, shape, y, noise_std, mask):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using the full Jacobian, computed using reverse mode.
    In early tests, this was around two times slower
    than vjp for a sparse observation inpainting operator (d_x=64, d_y=2).
    vjp should have scaling with d_x.
    """
    ratio = get_ratio(sde)

    def get_likelihood_score(y, H, mask, noise_std):
        # posterior_score_0 - Using multivariate Gaussian, with exploring calculation of a full Jacobian, not desireable in general
        # TODO: see if I can replace if statement with an observation_map function?
        assert jnp.shape(y) == jnp.shape(mask)
        def likelihood_score(_estimate_x_0, x, x_0, t):  # No linear solve required
            inv_hessian = jacrev(_estimate_x_0)  # TODO: has_aux not implemented in old version of JAX?
            x = x.flatten()
            Sigma_d_r_t = inv_hessian(x)
            Sigma = Sigma_d_r_t * ratio(t)
            Cyy = Sigma + noise_std**2 * jnp.eye(y.shape[0])
            innovation = y - mask * x_0
            f = jnp.linalg.solve(Cyy, innovation)
            return mask * f
        return likelihood_score

    estimate_x_0 = get_estimate_x_0(sde, score, shape)
    likelihood_score = get_likelihood_score(y, mask, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        _estimate_x_0 = lambda _x: estimate_x_0(_x, t)
        __estimate_x_0 = lambda _x: _estimate_x_0(_x)[0]
        x_0, vjp_estimate_x_0, s = vjp(
            _estimate_x_0, x, has_aux=True)
        ls = likelihood_score(__estimate_x_0, x, x_0, t)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_vjp_approximate_posterior_plus(
        sde, score, shape, y, noise_std, mask):
    """
    TODO: Unstable on FFHQ_noise_std=0.001
    Uses diagonal of second moment approximation of the covariance of x_0|x_t.

    Computes only two vjps.
    """
    ratio = get_ratio(sde)
    estimate_mask_x_0 = get_estimate_mask_x_0(sde, score, shape, mask)

    def get_likelihood_score(y, noise_std):
        def likelihood_score(vjp_h_x_0, h_x_0, t):
            diag = vjp_h_x_0(jnp.ones(y.shape[0]))[0]
            Cyy = ratio(t) * diag + noise_std**2
            innovation = y - h_x_0
            return innovation / Cyy
        return likelihood_score

    likelihood_score = get_likelihood_score(y, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, vjp_h_x_0, s = vjp(
            lambda x: estimate_mask_x_0(x, t), x, has_aux=True)
        ls = likelihood_score(vjp_h_x_0, h_x_0, t)
        ls = vjp_h_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_jvp_approximate_posterior_plus(
        sde, score, shape, y, noise_std, mask):
    """
    TODO: Unstable on FFHQ_noise_std=0.001
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using one jvp and one vjp.
    """
    mask = jnp.float32(mask)
    ratio = get_ratio(sde)
    estimate_h_x_0 = get_estimate_mask_x_0(sde, score, shape, mask)

    def get_likelihood_score(y, mask, noise_std):
        def likelihood_score(jvp_h_x_0, h_x_0, t):
            Cyy = ratio(t) * jvp_h_x_0 + noise_std**2
            innovation = y - h_x_0
            return innovation / Cyy
        return likelihood_score

    likelihood_score = get_likelihood_score(y, mask, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, jvp_h_x_0 = jvp(lambda x: estimate_h_x_0(x, t)[0], (x,), (mask,))
        ls = likelihood_score(jvp_h_x_0, h_x_0, t)
        print(ls.shape)
        # x_0, vjp_x_0, s = vjp(
        #     lambda x: estimate_x_0(x, t), x, has_aux=True)
        h_x_0, vjp_h_x_0, s = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        ls = vjp_h_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_vjp_approximate_posterior(
        sde, score, shape, y, noise_std, H):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using vjps where possible. TODO: profile vs jacfwd.
    """

    ratio = get_ratio(sde)

    def get_likelihood_score(y, H, noise_std):
        assert jnp.shape(y)[0] == jnp.shape(H)[0]
        def likelihood_score(vjp_estimate_x_0, x_0, t):
            vec_vjp_fn = vmap(vjp_estimate_x_0)
            H_grad_x_hat = vec_vjp_fn(H)[0]
            Cyy = ratio(t) * H @ H_grad_x_hat.T + noise_std**2 * jnp.eye(y.shape[0])
            innovation = y - H @ x_0
            f = jnp.linalg.solve(Cyy, innovation)
            f = H.T @ f
            return f
        return likelihood_score

    likelihood_score = get_likelihood_score(y, H, noise_std)
    estimate_x_0 = get_estimate_x_0(sde, score, shape)

    def approx_posterior_score(x, t):
        x = x.flatten()
        x_0, vjp_estimate_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        ls = likelihood_score(vjp_estimate_x_0, x_0, t)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_jvp_approximate_posterior(
        sde, score, shape, y, noise_std, H):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using d_y jvps and one vjp.
    """
    ratio = get_ratio(sde)
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, H)

    def get_likelihood_score(y, H, noise_std):
        assert jnp.shape(y)[0] == jnp.shape(H)[0]
        def likelihood_score(jvp_h_x_0, t):
            h_x_0, h_grad_h_x_0 = jvp_h_x_0(H)
            Cyy = ratio(t) * h_grad_h_x_0 + noise_std**2 * jnp.eye(y.shape[0])
            innovation = y - h_x_0[0]
            f = jnp.linalg.solve(Cyy, innovation)
            return f
        return likelihood_score

    likelihood_score = get_likelihood_score(y, H, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        jvp_h_x_0 = vmap(lambda h: jvp(lambda x: estimate_h_x_0(x, t)[0], (x,), (h,)))
        ls = likelihood_score(jvp_h_x_0, t)
        x_0, vjp_h_x_0, s = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        ls = vjp_h_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_jacfwd_approximate_posterior(
        sde, score, shape, y, noise_std, H):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using the full Jacobian. In early tests, this was around two times slower
    than vjp for a sparse observation inpainting operator (d_x=64, d_y=2).
    VJP should have scaling with d_x.
    """
    ratio = get_ratio(sde)

    def estimate_h_x_0(x, t):
        x_0, s = estimate_x_0(x, t)
        h_x_0 = H @ x_0
        return h_x_0, (s, h_x_0)

    def get_likelihood_score(y, H, noise_std):
        # posterior_score_0 - Using multivariate Gaussian, with exploring calculation of a full Jacobian, not desireable in general
        # TODO: see if I can replace if statement with an observation_map function?
        assert jnp.shape(y)[0] == jnp.shape(H)[0]
        def likelihood_score(H_grad_x_0, h_x_0, t):
            H_Sigma = H_grad_x_0 * ratio(t)
            Cyy = H_Sigma @ H.T + noise_std**2 * jnp.eye(y.shape[0])
            innovation = y - h_x_0
            f = jnp.linalg.solve(Cyy, innovation)
            f = H_grad_x_0.T @ f
            return f
        return likelihood_score

    estimate_x_0 = get_estimate_x_0(sde, score, shape)
    likelihood_score = get_likelihood_score(y, H, noise_std)

    def approx_posterior_score(x, t):
        _estimate_h_x_0 = lambda _x: estimate_h_x_0(_x, t)
        inv_hessian = jacfwd(_estimate_h_x_0, has_aux=True)
        H_grad_x_0, (s, h_x_0) = inv_hessian(x.flatten())
        ls = likelihood_score(H_grad_x_0, h_x_0, t)
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_jacrev_approximate_posterior(
        sde, score, shape, y, noise_std, H):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using the full Jacobian. In early tests, this was around two times slower
    than vjp for a sparse observation inpainting operator (d_x=64, d_y=2).
    VJP should have scaling with d_x.
    """
    ratio = get_ratio(sde)

    def estimate_h_x_0(x, t):
        x_0, s = estimate_x_0(x, t)
        h_x_0 = H @ x_0
        return h_x_0, (s, h_x_0)

    def get_likelihood_score(y, H, noise_std):
        # posterior_score_0 - Using multivariate Gaussian, with exploring calculation of a full Jacobian, not desireable in general
        # TODO: see if I can replace if statement with an observation_map function?
        assert jnp.shape(y)[0] == jnp.shape(H)[0]
        def likelihood_score(H_grad_x_0, h_x_0, t):
            H_Sigma = H_grad_x_0 * ratio(t)
            Cyy = H_Sigma @ H.T + noise_std**2 * jnp.eye(y.shape[0])
            innovation = y - h_x_0
            f = jnp.linalg.solve(Cyy, innovation)
            f = H_grad_x_0.T @ f
            return f
        return likelihood_score

    estimate_x_0 = get_estimate_x_0(sde, score, shape)
    likelihood_score = get_likelihood_score(y, H, noise_std)

    def approx_posterior_score(x, t):
        _estimate_h_x_0 = lambda _x: estimate_h_x_0(_x, t)
        inv_hessian = jacrev(_estimate_h_x_0, has_aux=True)
        H_grad_x_0, (s, h_x_0) = inv_hessian(x.flatten())
        ls = likelihood_score(H_grad_x_0, h_x_0, t)
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_approximate_posterior(sde, score, log_likelihood, grad_log_likelihood=None):
    if grad_log_likelihood is None:  # Try JAX grad
        grad_log_likelihood = grad(log_likelihood)
    likelihood_score = jit(vmap(
        grad_log_likelihood, in_axes=(0, 0), out_axes=(0)))

    # TODO: decide whether it will be preferable to use un vmapped prior here
    # which would change the way that the model is written
    return lambda x, t: likelihood_score(x, t) + score(x, t)

