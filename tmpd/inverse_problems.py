"""Inverse problems."""
import jax.numpy as jnp
from jax import vmap, grad, jacfwd, vjp, jacrev, jit
from diffusionjax.utils import batch_mul, batch_matmul, batch_linalg_solve_A, batch_linalg_solve


def get_linear_inverse_guidance(
        sde, observation_map, y, noise_std, HHT):
    """
    Pseudo-Inverse guidance score for an observation_map that can be
    represented by a lambda x: mask * x
    Args:
        HHT: H @ H.T which has shape (d_y, d_y) or ().
    """
    estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
    def guidance_score(x, t):
        h_x_0, vjp_estimate_h_x_0, (s, _) = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        innovation = y - h_x_0
        if HHT.shape == (y.shape[0], y.shape[0]):
            C_yy = sde.r2(t[0], data_variance=1.) * HHT + noise_std**2 * jnp.eye(y.shape[0])
            f = batch_linalg_solve_A(C_yy, innovation)
        else:
            C_yy = sde.r2(t[0], data_variance=1.) + noise_std**2
            f = innovation / C_yy
        ls = vjp_estimate_h_x_0(f)[0]
        gs = s + ls
        return gs

    return guidance_score


def get_vjp_guidance(
        sde, H, y, noise_std, shape):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using vjps where possible.
    """
    estimate_x_0 = sde.get_estimate_x_0(lambda x: x, shape=(-1))

    def guidance_score(x, t):
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        x_0, vjp_x_0, (s, _) = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        vec_vjp_x_0 = vmap(vjp_x_0)
        H_grad_x_0 = vec_vjp_x_0(H)[0]
        H_grad_x_0 = jnp.squeeze(vec_vjp_x_0(H)[0], axis=(1, -1)).reshape(H.shape)
        C_yy = sde.ratio(t) * H @ H_grad_x_0.T + noise_std**2 * jnp.eye(y.shape[0])
        innovation = y - H @ x_0
        f = jnp.linalg.solve(C_yy, innovation)
        f = H.T @ f
        ls = vjp_x_0(f)[0]
        gs = s + ls
        return jnp.squeeze(gs, axis=0)

    return vmap(guidance_score)


def get_vjp_guidance_plus(
        sde, observation_map, y, noise_std):
    """
    Uses diagonal of second moment approximation of the covariance of x_0|x_t.

    Computes only two vjps.
    """
    estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
    batch_observation_map = vmap(observation_map)
    def guidance_score(x, t):
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        h_x_0, vjp_h_x_0, (s, _) = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        diag = vjp_h_x_0(batch_observation_map(jnp.ones(x.shape)))[0]
        diag = batch_observation_map(vjp_h_x_0(batch_observation_map(jnp.ones(x.shape)))[0])
        C_yy = sde.ratio(t[0]) * diag + noise_std**2
        innovation = y - h_x_0
        ls = innovation / C_yy
        ls = vjp_h_x_0(ls)[0]
        gs = s + ls
        return jnp.squeeze(gs, axis=0)

    return vmap(guidance_score)


def get_jacrev_guidance(
        sde, observation_map, y, noise_std, shape):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using vjps where possible.
    """
    batch_observation_map = vmap(observation_map)
    estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
    def guidance_score(x, t):
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        h_x_0, (s, _) = estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: estimate_h_x_0(_x, t)[0])(x)
        grad_H_x_0 = jnp.squeeze(jacrev(lambda _x: estimate_h_x_0(_x, t)[0])(x), axis=(0, 2, -1))
        H_grad_H_x_0 = batch_observation_map(grad_H_x_0)
        C_yy = sde.ratio(t[0]) * H_grad_H_x_0 + noise_std**2 * jnp.eye(y.shape[0])
        innovation = y - h_x_0.reshape(y.shape[0])
        innovation = innovation.reshape(-1, 1)
        f = jnp.linalg.solve(C_yy, innovation)
        ls = jnp.transpose(grad_H_x_0, (1, 2, 0)) @ f
        gs = jnp.squeeze(s, axis=0) + ls
        return gs

    return vmap(guidance_score)


def get_jacfwd_guidance(
        sde, observation_map, y, noise_std, shape):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using d_y jvps.
    """
    batch_observation_map = vmap(observation_map)
    estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
    def guidance_score(x, t):
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        h_x_0, (s, _) = estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        H_grad_x_0 = jnp.squeeze(jacfwd(lambda _x: estimate_h_x_0(_x, t)[0])(x), axis=(0, 2, -1))
        H_grad_H_x_0 = batch_observation_map(H_grad_x_0)
        C_yy = sde.ratio(t[0]) * H_grad_H_x_0 + noise_std**2 * jnp.eye(y.shape[0])
        innovation = y - h_x_0.reshape(y.shape[0])
        innovation = innovation.reshape(-1, 1)
        f = jnp.linalg.solve(C_yy, innovation)
        ls = jnp.transpose(H_grad_x_0, (1, 2, 0)) @ f
        gs = jnp.squeeze(s, axis=0) + ls
        return gs

    return vmap(guidance_score)


def get_diag_jacrev_guidance(sde, observation_map, y, noise_std):
    """Use a diagonal approximation to the variance inside the likelihood,
    This produces similar results when the covariance is approximately diagonal
    """
    batch_observation_map = vmap(observation_map)
    estimate_h_x_0 = sde.get_estimate_x_0(observation_map)

    def guidance_score(x, t):
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        h_x_0, (s, _) = estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        h_x_0 = jnp.squeeze(h_x_0, axis=0)
        grad_H_x_0 = jacrev(lambda _x: estimate_h_x_0(_x, t)[0])(x)
        grad_H_x_0 = jnp.squeeze(grad_H_x_0, axis=(0, 2))
        H_grad_H_x_0 = batch_observation_map(grad_H_x_0)
        C_yy = sde.ratio(t) * jnp.diag(H_grad_H_x_0) + noise_std**2
        innovation = y - h_x_0
        f = innovation / C_yy
        ls = grad_H_x_0.T @ f
        ls = ls.reshape(s.shape)
        gs = s + ls
        return jnp.squeeze(gs, axis=0)

    return vmap(guidance_score)


def get_diag_vjp_guidance(sde, H, y, noise_std, shape):
    """Use a diagonal approximation to the variance inside the likelihood,
    This produces similar results when the covariance is approximately diagonal
    """

    def observation_map(x):
        x = x.flatten()
        return H @ x

    estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
    estimate_x_0 = sde.get_estimate_x_0(lambda x: x, shape=(-1))

    def guidance_score(x, t):
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        x_0, vjp_x_0, (s, _) = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        vec_vjp_x_0 = vmap(lambda h: vjp_x_0(h))
        H_grad_x_0 = vec_vjp_x_0(H)[0]
        H_grad_x_0 = H_grad_x_0.reshape(H.shape)
        diag_H_grad_H_x_0 = jnp.sum(H * H_grad_x_0, axis=-1)
        C_yy = sde.ratio(t[0]) * diag_H_grad_H_x_0 + noise_std**2
        innovation = y - H @ x_0
        f = innovation / C_yy
        ls = vjp_x_0(H.T @ f)[0]
        gs = s + ls
        return jnp.squeeze(gs, axis=0)

    return vmap(guidance_score)
