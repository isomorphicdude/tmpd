"""Inverse problems."""
import jax.numpy as jnp
from jax import vmap, grad, jacfwd, vjp, jacrev, jvp, jit


# def get_estimate_h_x_0(sde, score, shape, observation_map):
#     """Get an MMSE estimate of x_0, pushed through observation_map.
#     Args:
#         observation_map: forward, assumed linear, map.
#     """
#     def estimate_x_0(x, t):
#         """The MMSE estimate for x_0|x_t,
#         which is it's expetion as given by Tweedie's formula."""
#         x = x.reshape(shape)
#         x = jnp.expand_dims(x, axis=0)
#         t = jnp.expand_dims(t, axis=0)
#         m_t = sde.mean_coeff(t)
#         v_t = sde.variance(t)
#         s = score(x, t)
#         s = s.flatten()
#         x = x.flatten()
#         return observation_map((x + v_t * s) / m_t), s

#     return estimate_x_0


# def get_estimate_x_0(sde, score, shape):
#     """Get an MMSE estimate of x_0
#     """
#     def estimate_x_0(x, t):
#         """The MMSE estimate for x_0|x_t,
#         which is it's expectation as given by Tweedie's formula."""
#         x = x.reshape(shape)
#         x = jnp.expand_dims(x, axis=0)
#         t = jnp.expand_dims(t, axis=0)
#         m_t = sde.mean_coeff(t)
#         v_t = sde.variance(t)
#         s = score(x, t)
#         s = s.flatten()
#         x = x.flatten()
#         return (x + v_t * s) / m_t, s

#     return estimate_x_0


def get_dps(
        scale, sde, score, shape, y, noise_std, observation_map):
    """
    `Diffusion Posterior Sampling for general noisy inverse problems'
    implemented with a vmap grad.
    """
    def get_l2_norm(y, estimate_h_x_0):
        def likelihood_score_approx(x, t):
            h_x_0, s = estimate_h_x_0(x, t)
            innovation = y - h_x_0
            return jnp.linalg.norm(innovation), s
        return likelihood_score_approx

    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape)
    l2_norm = get_l2_norm(y, estimate_h_x_0)
    likelihood_score = grad(l2_norm, has_aux=True)
    def approx_posterior_score(x, t):
        x = x.flatten()
        ls, s = likelihood_score(x, t)
        posterior_score = s - scale * ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_dps_plus(
        scale, sde, score, shape, y, noise_std, observation_map):
    """
    `Diffusion Posterior Sampling for general noisy inverse problems'
    implemented with a vmap grad.
    """
    def get_l2_norm(y, estimate_h_x_0):
        def l2_norm(x, t):
            # y is given as a d_x length vector
            h_x_0, s = estimate_h_x_0(x, t)
            innovation = y - h_x_0
            return jnp.linalg.norm(innovation), s
        return l2_norm

    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    l2_norm = get_l2_norm(y, estimate_h_x_0)
    likelihood_score = grad(l2_norm, has_aux=True)
    def approx_posterior_score(x, t):
        x = x.flatten()
        ls, s = likelihood_score(x, t)
        posterior_score = s - scale * ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_diffusion_posterior_sampling_plus(
        sde, score, shape, y, noise_std, observation_map):
    """
    `Diffusion Posterior Sampling for general noisy inverse problems'
    implemented with a single vjp.
    NOTE: This is not how Chung et al. 2022, https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/guided_diffusion/condition_methods.py
    implemented their method, their method is :meth:get_dps_plus.
    Whereas this method uses their approximation in Eq. 11 https://arxiv.org/pdf/2209.14687.pdf#page=20&zoom=100,144,757
    to directly calculate the score.
    """
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, vjp_estimate_h_x_0, s = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        innovation = y - h_x_0
        C_yy = noise_std**2
        ls = innovation / C_yy
        ls = vjp_estimate_h_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_diffusion_posterior_sampling(
        sde, score, shape, y, noise_std, observation_map):
    """
    `Diffusion Posterior Sampling for general noisy inverse problems'
    implemented with a single vjp.
    Assumes linear observation_map
    """
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, vjp_estimate_h_x_0, s = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        innovation = y - h_x_0
        C_yy = noise_std**2
        f = innovation / C_yy
        ls = vjp_estimate_h_x_0(f)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_linear_inverse_guidance_plus(
        sde, score, shape, y, noise_std, observation_map):
    """
    Pseudo-Inverse guidance score for an observation_map that can be
    represented by a lambda x: mask * x
    """
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, vjp_estimate_h_x_0, s = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        innovation = y - h_x_0
        C_yy = sde.r2(t, data_variance=1.) + noise_std**2
        f = innovation / C_yy
        ls = vjp_estimate_h_x_0(f)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_linear_inverse_guidance(
        sde, observation_map, shape, y, noise_std, HHT):
        # sde, score, shape, y, noise_std, observation_map, HHT):
    """Pseudo-Inverse guidance score for linear observation_map.
    Args:
        HHT: H @ H.T which has shape (d_y, d_y)
    """
    estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
    batch_linalg_solve = vmap(lambda a, b: jnp.linalg.solve(a, b), in_axes=(None, 0))
    # estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        print(x.shape)
        print(t.shape)
        # x = x.flatten()
        h_x_0, vjp_estimate_h_x_0, (s, x_0) = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        print(h_x_0.shape)
        print(s.shape)
        print(x_0.shape)
        innovation = y - h_x_0
        print("hi")
        print(HHT.shape)
        print(sde.r2(t[0], 1.).shape)
        print("i")
        C_yy = sde.r2(t[0], data_variance=1.) * HHT + noise_std**2 * jnp.eye(y.shape[0])
        print("innovation", innovation.shape)
        print("cyy", C_yy.shape)
        # Will need to be a batch operation: More memory efficient this way as well, only use once C_yy matrix
        # f = jnp.linalg.solve(C_yy, innovation)
        f = batch_linalg_solve(C_yy, innovation)
        print('f', f.shape)
        ls = vjp_estimate_h_x_0(f)[0]
        print('ls', ls.shape)
        print('s', s.shape)
        guided_score = s + ls
        print("shape", shape)
        return guided_score
        assert 0
        return posterior_score.reshape(shape)

    # return vmap(approx_posterior_score)
    return approx_posterior_score


def get_diag_approximate_posterior(sde, score, shape, y, noise_std, observation_map):
    """Use a diagonal approximation to the variance inside the likelihood,
    This produces similar results when the covariance is approximately diagonal
    """
    batch_observation_map = vmap(observation_map)
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, s = estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: estimate_h_x_0(_x, t)[0])(x)
        H_grad_H_x_0 = batch_observation_map(grad_H_x_0)
        C_yy = sde.ratio(t) * jnp.diag(H_grad_H_x_0) + noise_std**2
        innovation = y - h_x_0
        f = innovation / C_yy
        ls = grad_H_x_0.T @ f
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_diag_vjp_approximate_posterior(sde, score, shape, y, noise_std, H):
    """Use a diagonal approximation to the variance inside the likelihood,
    This produces similar results when the covariance is approximately diagonal
    """
    observation_map = lambda x: H @ x
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, vjp_estimate_h_x_0, s = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        diag_vjp = vmap(lambda h: jnp.dot(vjp_estimate_h_x_0(h)[0], h))
        diag_H_grad_H_x_0 = diag_vjp(H.T)
        C_yy = sde.ratio(t) * diag_H_grad_H_x_0 + noise_std**2
        innovation = y - h_x_0
        f = innovation / C_yy
        ls = vjp_estimate_h_x_0(f)
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_diag_jacfwd_approximate_posterior(sde, score, shape, y, noise_std, observation_map):
    """Use a diagonal approximation to the variance inside the likelihood,
    This produces similar results when the covariance is approximately diagonal
    """
    batch_observation_map = vmap(observation_map)
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, s = estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        H_grad_x_0 = jacfwd(lambda _x: estimate_h_x_0(_x, t)[0])(x)
        H_grad_H_x_0 = batch_observation_map(H_grad_x_0)
        C_yy = sde.ratio(t) * jnp.diag(H_grad_H_x_0) + noise_std**2
        f = (y - h_x_0) / C_yy
        ls = H_grad_x_0.T @ f
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score



def get_jvp_approximate_posterior(
        sde, score, shape, y, noise_std, H):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using vjps where possible.
    """
    observation_map = lambda x: H @ x
    batch_observation_map = vmap(observation_map)
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, s = estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        H_grad_x_0 = jacfwd(lambda _x: estimate_h_x_0(_x, t)[0])(x)
        H_grad_H_x_0 = batch_observation_map(H_grad_x_0)
        C_yy = sde.ratio(t) * H_grad_H_x_0 + noise_std**2 * jnp.eye(y.shape[0])
        innovation = y - h_x_0
        f = jnp.linalg.solve(C_yy, innovation)
        ls = H_grad_x_0.T @ f
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_vjp_approximate_posterior(
        sde, score, shape, y, noise_std, H):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using vjps where possible.
    """
    estimate_x_0 = get_estimate_h_x_0(sde, score, shape, lambda x: x)
    def approx_posterior_score(x, t):
        x = x.flatten()
        x_0, vjp_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        vec_vjp_x_0 = vmap(vjp_x_0)
        H_grad_x_0 = vec_vjp_x_0(H)[0]
        C_yy = sde.ratio(t) * H @ H_grad_x_0.T + noise_std**2 * jnp.eye(y.shape[0])
        innovation = y - H @ x_0
        f = jnp.linalg.solve(C_yy, innovation)
        ls = vjp_x_0(H.T @ f)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_vjp_approximate_posterior_plus(
        sde, score, shape, y, noise_std, observation_map):
    """
    Uses diagonal of second moment approximation of the covariance of x_0|x_t.

    Computes only two vjps.
    """
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, vjp_h_x_0, s = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        diag = observation_map(vjp_h_x_0(observation_map(jnp.ones(x.shape[0])))[0])
        C_yy = sde.ratio(t) * diag + noise_std**2
        innovation = y - h_x_0
        ls = innovation / C_yy
        ls = vjp_h_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_jacrev_approximate_posterior(
        sde, score, shape, y, noise_std, observation_map):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using vjps where possible.
    """
    batch_observation_map = vmap(observation_map)
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, s = estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: estimate_h_x_0(_x, t)[0])(x)
        H_grad_H_x_0 = batch_observation_map(grad_H_x_0)
        C_yy = sde.ratio(t) * H_grad_H_x_0 + noise_std**2 * jnp.eye(y.shape[0])
        innovation = y - h_x_0
        f = jnp.linalg.solve(C_yy, innovation)
        ls = grad_H_x_0.T @ f
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_jacfwd_approximate_posterior(
        estimate_h_x_0, shape, y, noise_std, observation_map):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using d_y jvps.
    """
    batch_observation_map = vmap(observation_map)
    # estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    # estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
    def approx_posterior_score(x, t):
        # x = x.flatten()
        h_x_0, (s, x_0) = estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        assert 0
        H_grad_x_0 = jacfwd(lambda _x: estimate_h_x_0(_x, t)[0])(x)
        H_grad_H_x_0 = batch_observation_map(H_grad_x_0)
        C_yy = sde.ratio(t) * H_grad_H_x_0 + noise_std**2 * jnp.eye(y.shape[0])
        innovation = y - h_x_0
        f = jnp.linalg.solve(C_yy, innovation)
        ls = H_grad_x_0.T @ f
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score
