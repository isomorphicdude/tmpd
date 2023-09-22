"""Inverse problems, but only defined through forward and adjoint operator."""
from jax import vmap, grad, jacfwd, vjp
import jax.numpy as jnp
from jax.lax import scan
import jax.random as random
from diffusionjax.utils import batch_mul, get_score
import numpy as np
# TODO: get rid of H and replace with observation map, if H_grad_x_hat_t is not correct or not necessary?
# TODO: work out if it is more efficient to work with a vmap'd score or not vmap'd score,
# would make more sense to plug in an inverse score, but if it is more efficient to evaluate the score function
# in a vector way, that is better. All depends on how the vjp/jvp can be calculated.


def get_batch_estimate_x_0(score, sde, shape, flag=0):
    """Get an MMSE estimate of x_0 over a batch. This cannot be
    vector jacobian product, because it will return a batch, and
    would require a vjp for each batch member. So how to calculate Kalman gain?
    Maybe not possible to batch, but would rather pmap across GPU?

    # TODO: vjp can be evaluated against an array of any shape?
    """

    if flag==0:
        # problem is that score is already vmapped
        def estimate_x_0(x, t):
            """The MMSE estimate for x_0|x_t,
            which is it's expectation as given by Tweedie's formula."""
            m_t = sde.mean_coeff(t)
            v_t = sde.variance(t)
            _x = x.reshape(shape)
            _x = jnp.expand_dims(_x, axis=0)
            _t = jnp.expand_dims(t, axis=0)
            _s = score(_x, _t)
            s = _s.flatten()
            return (x + v_t * s) / m_t, s

        def estimate_x_0_(x, t):
            """The MMSE estimate for x_0|x_t,
            which is it's expectation as given by Tweedie's formula."""
            m_t = sde.mean_coeff(t)
            v_t = sde.variance(t)
            x = x.flatten()
            s = score(x, t)
            return (x + v_t * s) / m_t, s

    elif flag==1:  # Due to be superceded, and is to test a correction to the literature
        def estimate_x_0(x, t):
            """The MMSE estimate for x_0|x_t,
            which is it's expectation as given by Tweedie's formula."""
            v_t = sde.variance(t)
            _x = x.reshape(shape)
            _x = jnp.expand_dims(x, axis=0)
            _t = jnp.expand_dims(t, axis=0)
            _s = score(_x, _t)
            s = _s.flatten()
            return x + v_t * s, s

        def estimate_x_0_(x, t):
            """The MMSE estimate for x_0|x_t,
            which is it's expectation as given by Tweedie's formula."""
            v_t = sde.variance(t)
            x = x.flatten()
            s = score(x, t)
            return x + v_t * s, s


def get_estimate_x_0(sde, score, shape, flag=0):
    """Get an MMSE estimate of x_0
    # TODO: vjp can be evaluated against an array of any shape?
    """
    if flag==0:
        # TODO: problem is that score is already vmapped, does this cause any inefficiencies?
        def estimate_x_0(x, t):
            """The MMSE estimate for x_0|x_t,
            which is it's expectation as given by Tweedie's formula."""
            m_t = sde.mean_coeff(t)
            v_t = sde.variance(t)
            _x = x.reshape(shape)
            _x = jnp.expand_dims(_x, axis=0)
            _t = jnp.expand_dims(t, axis=0)
            _s = score(_x, _t)
            s = _s.flatten()
            return (x + v_t * s) / m_t, s

        def estimate_x_0_(x, t):
            """The MMSE estimate for x_0|x_t,
            which is it's expectation as given by Tweedie's formula."""
            m_t = sde.mean_coeff(t)
            v_t = sde.variance(t)
            x = x.flatten()
            s = score(x, t)
            return (x + v_t * s) / m_t, s

    elif flag==1:  # Due to be superceded, and is to test a correction to the literature
        def estimate_x_0(x, t):
            """The MMSE estimate for x_0|x_t,
            which is it's expectation as given by Tweedie's formula."""
            v_t = sde.variance(t)
            _x = x.reshape(shape)
            _x = jnp.expand_dims(x, axis=0)
            _t = jnp.expand_dims(t, axis=0)
            _s = score(_x, _t)
            s = _s.flatten()
            return x + v_t * s, s

        def estimate_x_0_(x, t):
            """The MMSE estimate for x_0|x_t,
            which is it's expectation as given by Tweedie's formula."""
            v_t = sde.variance(t)
            x = x.flatten()
            s = score(x, t)
            return x + v_t * s, s

    return estimate_x_0


def get_estimate_x_n(solver, rng, t):
    return lambda x, t: solver.update(rng, x, t)


def get_diffusion_posterior_sampling_score(
        sde, shape, y, noise_std, forward_operator, adjoint_operator,
        likelihood_score=None, estimate_x_0=None):
    """
    TODO: Easily generalizable for non-linear forward map?
    Pseudo-inverse guidance score for linear forward map H.
    :arg shape:
    :arg y:
    :arg likelihood_score:
    :arg estimate_x_0:
    """
    def ratio(t):
        return sde.variance(t) / sde.mean_coeff(t)

    def get_likelihood_score(y, forward_operator, adjoint_operator, noise_std):
        def likelihood_score_approx(x_0, t):
            innovation = y - forward_operator(x_0)
            Cyy = noise_std**2
            f = innovation / Cyy
            return adjoint_operator(f)
        return likelihood_score_approx

    if estimate_x_0 is None:
        estimate_x_0 = get_estimate_x_0(score, sde, shape)
    if likelihood_score is None:  # likelihood score is not provided
        likelihood_score = get_likelihood_score(y, forward_operator, adjoint_operator, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        x_0, vjp_estimate_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        ls = likelihood_score(x_0, t)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_linear_inverse_guidance_score(
        sde, shape, y, noise_std, forward_operator, adjoint_operator,
        likelihood_score=None, estimate_x_0=None):
    """
    TODO: the precision in parallel with the observation precision is
    different depending on the type (VP or VE) of SDE.

    Currently this is only for VP SDE
    Pseudo-Inverse guidance score for linear forward map H.
    :arg shape:
    :arg y:
    :arg H:
    :arg mask:
    :arg likelihood_score:
    :arg estimate_x_0:
    """
    def ratio(t):
        # if sde is VE:
        # return sde.variance(t) / sde.mean_coeff(t)
        # if sde is VP:
        return sde.variance(t)

    def get_likelihood_score(
            y, forward_operator, adjoint_operator, noise_std):
        # y is given as just observations, H a linear operator
        assert jnp.shape(y)[0] == jnp.shape(H)[0]
        def likelihood_score_approx(x_0, t):
            innovation = y - H @ x_0
            Cyy = ratio(t) * H @ H.T + noise_std**2 * jnp.eye(y.shape[0])
            f = jnp.linalg.solve(Cyy, innovation)
            return H.T @ f
        return likelihood_score_approx

    if estimate_x_0 is None:
        estimate_x_0 = get_estimate_x_0(score, sde, shape)
    if likelihood_score is None:  # likelihood score is not provided
        likelihood_score = get_likelihood_score(y, H, mask, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        x_0, vjp_estimate_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        ls = likelihood_score(x_0, t)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_experimental_diag_approximate_posterior_score(sde, shape, y, noise_std,
        forward_operator, adjoint_operator,
        likelihood_score=None, estimate_x_0=None, mask_estimate_x_0=None):
    """Use a diagonal approximation to the variance inside the likelihood,
    This produces similar results when the covariance is approximately diagonal
    """
    def ratio(t):
        return sde.variance(t) / sde.mean_coeff(t)


def get_diag_approximate_posterior_score(sde, shape, y, noise_std,
        forward_operator, adjoint_operator,
        likelihood_score=None, estimate_x_0=None):
    """Use a diagonal approximation to the variance inside the likelihood,
    This produces similar results when the covariance is approximately diagonal
    """
    def ratio(t):
        return sde.variance(t) / sde.mean_coeff(t)

    def get_likelihood_score(y, forward_operator, noise_std):
        # posterior_score 1 - Using multivariate Gaussian, with exploring calculation of vjp with H or mask
        if mask is not None:
            y = y.flatten()
            mask = mask.flatten()
            assert jnp.shape(y) == jnp.shape(mask)
            # use a diagonal approximation to the variance inside the likelihood
            # Not correct unless Sigma_d_r_t is diagonal
            def likelihood_score(vjp_x_0, x_0, t):
                # TODO this doesn't seem to work?
                # It actually projects onto the data
                # this is due to it being the wrong calculation
                vec_vjp_fn = vmap(vjp_estimate_x_0)
                H_grad_x_hat = vec_vjp_fn(H)[0]
                Sigma_d_r_t_diag = jnp.diag(H.T @ H_grad_x_hat)
                # Sigma_d_r_t_diag = jnp.diag(H @ H_grad_x_hat.T)
                Sigma_diag = Sigma_d_r_t_diag * ratio(t)
                Cyy = Sigma_diag + noise_std**2
                innovation = y - mask * x_0
                f = innovation / Cyy
                return f  # remember to take vjp_estimate_x_0(f)[0]
        else:
            assert jnp.shape(y)[0] == jnp.shape(H)[0]
            def likelihood_score(vjp_h_x_0, x_0, t):
                # You can compute the diagonal of the hessian by using vmap of grad
                vec_vjp_fn = vmap(vjp_h_x_0)
                h_grad_x_hat_h_T = vec_vjp_fn(H)[0]
                Cyy = ratio(t) * H @ H_grad_x_hat.T + noise_std**2 * jnp.eye(y.shape[0])
                Cyy = jnp.diag(Cyy)
                innovation = y - H @ x_0
                f = innovation / Cyy
                f = H.T @ f
                return f  # remember to take vjp_estimate_x_0(f)[0]
        return likelihood_score

    if estimate_x_0 is None:
        estimate_x_0 = get_estimate_x_0(score, sde, shape)
    if likelihood_score is None:  # likelihood score is not provided
        likelihood_score = get_likelihood_score(y, H, mask, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        x_0, vjp_estimate_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        ls = likelihood_score(vjp_estimate_x_0, x_0, t)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_jac_approximate_posterior_score(sde, shape, y, noise_std,
        forward_map, adjoint_map, likelihood_score=None, estimate_x_0=None):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using the full Jacobian. TODO: profile vs vjp
    This one is just a guess using forward and adjoint map.
    """
    def ratio(t):
        return sde.variance(t) / sde.mean_coeff(t)

    def get_likelihood_score(y, vec_forward_map, adjoint_map, noise_std):
        # posterior_score_0 - Using multivariate Gaussian, with exploring calculation of a full Jacobian, not desireable in general
        # TODO: see if I can replace if statement with an observation_map function?
        assert jnp.shape(y)[0] == jnp.shape(H)[0]
        def likelihood_score(_estimate_x_0, x, x_0, t):
            inv_hessian = jacfwd(_estimate_x_0, has_aux=True)
            x = x.flatten()
            Sigma_d_r_t, score = inv_hessian(x)
            Sigma = Sigma_d_r_t * ratio(t)
            Cyy = vec_forward_map(vec_forward_map(Sigma).T).T + noise_std**2 * jnp.eye(y.shape[0])
            innovation = y - H @ x_0
            f = jnp.linalg.solve(Cyy, innovation)
            f = adjoint_map(f)
            return f
        return likelihood_score

    if estimate_x_0 is None:
        estimate_x_0 = get_estimate_x_0(score, sde, shape)
    if likelihood_score is None:  # likelihood score is not provided
        vec_forward_map = vmap(forward_map)
        likelihood_score = get_likelihood_score(y, vec_forward_map, adjoint_map, noise_std)

    def approx_posterior_score(x, t):
        x = x.flatten()
        _estimate_x_0 = lambda _x: estimate_x_0(_x, t)
        x_0, vjp_estimate_x_0, s = vjp(
            _estimate_x_0, x, has_aux=True)
        ls = likelihood_score(_estimate_x_0, x, x_0, t)
        ls = vjp_estimate_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score

