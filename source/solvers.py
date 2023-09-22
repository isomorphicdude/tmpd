"""Markov chains."""
import abc
from functools import partial
import jax
import jax.numpy as jnp
from jax import random
from jax import grad, vmap, vjp
from diffusionjax.utils import batch_mul
from diffusionjax.solvers import Solver, DDIMVP, DDIMVE, SMLD, DDPM


class KGDMVP(DDIMVP):
    """PiGDM Song et al. 2021. Markov chain using the DDIM Markov Chain or VP SDE. TODO: needs debugging"""
    def __init__(self, y, H, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.):
        super().__init__(model, eta, num_steps, dt, epsilon, beta_min, beta_max)
        self.estimate_x_0 = self.get_estimate_x_0(model, shape)
        self.batch_analysis = vmap(self.analysis)
        self.H = H
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]

    def get_estimate_x_0(self, model, shape):
        def estimate_x_0(x, t, timestep):
            m = self.sqrt_alphas_cumprod[timestep]
            sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
            x = x.reshape(shape)
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            epsilon = model(x, t)
            epsilon = epsilon.flatten()
            x = x.flatten()
            x_0 = (x - sqrt_1m_alpha * epsilon) / m
            return x_0, epsilon
        return estimate_x_0

    def analysis(self, x, t, timestep, ratio, v, m):
        x = x.flatten()
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t, timestep)
        x_0, vjp_estimate_x_0, epsilon = vjp(
            _estimate_x_0, x, has_aux=True)
        batch_vjp_estimate_x_0 = vmap(lambda x: vjp_estimate_x_0(x)[0])
        C_yy = ratio * batch_vjp_estimate_x_0(self.H) @ self.H.T + self.noise_std**2 * jnp.eye(self.num_y)
        ls = vjp_estimate_x_0(self.H.T @ jnp.linalg.solve(C_yy, self.y - self.H @ x_0))[0]  # nonsense image outputs
        return epsilon.reshape(self.shape), ls.reshape(self.shape)

    def posterior(self, x, t):
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        m = self.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
        v = sqrt_1m_alpha**2
        ratio = v / m
        alpha = m**2
        epsilon, ls = self.batch_analysis(x, t, timestep, ratio, v, m)
        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
        alpha_prev = m_prev**2
        coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
        coeff2 = jnp.sqrt(v_prev - coeff1**2)
        posterior_score = - batch_mul(1. / sqrt_1m_alpha, epsilon) + ls
        x_mean = batch_mul(m_prev / m, x) + batch_mul(sqrt_1m_alpha * (sqrt_1m_alpha * m_prev / m - coeff2), posterior_score)
        std = coeff1
        return x_mean, std


class KGDMVPplus(KGDMVP):
    """KGDMVP with a mask. TODO: needs debugging"""
    def __init__(self, y, mask, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.):
        super().__init__(y, mask, noise_std, shape, model, eta, num_steps, dt, epsilon, beta_min, beta_max)
        self.mask = mask

    def analysis(self, x, t, timestep, ratio):
        # TODO: do I need to flatten or is there a better way?
        x = x.flatten()
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t, timestep)
        x_0, vjp_estimate_x_0, epsilon = vjp(
            _estimate_x_0, x, has_aux=True)
        C_yy = ratio * vjp_estimate_x_0(self.mask)[0] + self.noise_std**2
        ls = vjp_estimate_x_0(self.mask * (self.y - x_0) / C_yy)[0]
        return epsilon.reshape(self.shape), ls.reshape(self.shape)


class KGDMVE(DDIMVE):
    def __init__(self, y, H, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
        super().__init__(model, eta, num_steps, dt, epsilon, sigma_min, sigma_max)
        self.eta = eta
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = jnp.exp(
            jnp.linspace(jnp.log(self.sigma_min),
                        jnp.log(self.sigma_max),
                        self.num_steps))
        self.H = H
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]
        self.estimate_x_0 = self.get_estimate_x_0(model, shape)
        self.batch_analysis = vmap(self.analysis)

    def get_estimate_x_0(self, model, shape):
        def estimate_x_0(x, t, timestep):
            std = self.discrete_sigmas[timestep]
            x = x.reshape(shape)
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            epsilon = model(x, t)
            epsilon = epsilon.flatten()
            x = x.flatten()
            x_0 = x - std * epsilon
            return x_0, epsilon
        return estimate_x_0

    def analysis(self, x, t, timestep, v):
        x = x.flatten()
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t, timestep)
        x_0, vjp_estimate_x_0, epsilon = vjp(
            _estimate_x_0, x, has_aux=True)
        batch_vjp_estimate_x_0 = vmap(lambda x: vjp_estimate_x_0(x)[0])
        C_yy = v * batch_vjp_estimate_x_0(self.H) @ self.H.T + self.noise_std**2 * jnp.eye(self.num_y)
        ls = vjp_estimate_x_0(self.H.T @ jnp.linalg.solve(C_yy, self.y - self.H @ x_0))[0]  # nonsense image outputs
        return epsilon.reshape(self.shape), ls.reshape(self.shape)

    def posterior(self, x, t):
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        sigma = self.discrete_sigmas[timestep]
        sigma_prev = self.discrete_sigmas_prev[timestep]
        epsilon, ls = self.batch_analysis(x, t, timestep, sigma**2)
        coeff1 = self.eta * jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
        coeff2 = jnp.sqrt(sigma_prev**2  - coeff1**2)
        std = coeff1
        posterior_score = - batch_mul(1. / sigma, epsilon) + ls
        x_mean = x + batch_mul(sigma * (sigma - coeff2), posterior_score)
        return x_mean, std


class KGDMVEplus(KGDMVE):
    def __init__(self, y, mask, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
        super().__init__(y, mask, noise_std, shape, model, eta, num_steps, dt, epsilon, sigma_min, sigma_max)
        self.mask = mask

    def analysis(self, x, t, timestep, v):
        x = x.flatten()
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t, timestep)
        x_0, vjp_estimate_x_0, epsilon = vjp(
            _estimate_x_0, x, has_aux=True)
        C_yy = v * vjp_estimate_x_0(self.mask)[0] + self.noise_std**2
        ls = vjp_estimate_x_0(self.mask * (self.y - x_0) / C_yy)[0]
        return epsilon.reshape(self.shape), ls.reshape(self.shape)


class PiGDMVP(DDIMVP):
    """PiGDM Song et al. 2021. Markov chain using the DDIM Markov Chain or VP SDE."""
    def __init__(self, y, H, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.):
        super().__init__(model, eta, num_steps, dt, epsilon, beta_min, beta_max)
        self.estimate_x_0 = self.get_estimate_x_0(model, shape)
        self.batch_analysis = vmap(self.analysis)
        self.H = H
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]

    def get_estimate_x_0(self, model, shape):
        def estimate_x_0(x, t, timestep):
            m = self.sqrt_alphas_cumprod[timestep]
            sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
            x = x.reshape(shape)
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            epsilon = model(x, t)
            epsilon = epsilon.flatten()
            x = x.flatten()
            x_0 = (x - sqrt_1m_alpha * epsilon) / m
            return x_0, epsilon
        return estimate_x_0

    def analysis(self, x, t, timestep, r):
        x = x.flatten()
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t, timestep)
        x_0, vjp_estimate_x_0, epsilon = vjp(
            _estimate_x_0, x, has_aux=True)
        batch_vjp_estimate_x_0 = vmap(lambda x: vjp_estimate_x_0(x)[0])
        C_yy = 1 + self.noise_std**2 / r
        ls = vjp_estimate_x_0(self.H.T @ (self.y - self.H @ x_0) / C_yy)[0]
        return x_0.reshape(self.shape), ls.reshape(self.shape), epsilon.reshape(self.shape)

    def posterior(self, x, t):
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        m = self.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
        v = sqrt_1m_alpha**2
        alpha = m**2
        x_0, ls, epsilon = self.batch_analysis(x, t, timestep, v)
        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
        alpha_prev = m_prev**2
        coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
        coeff2 = jnp.sqrt(v_prev - coeff1**2)
        x_mean = batch_mul(m_prev, x_0) + batch_mul(m, ls) + batch_mul(coeff2, epsilon)
        std = coeff1
        return x_mean, std


class PiGDMVPplus(PiGDMVP):
    """PiGDMVP with a mask."""
    def __init__(self, y, mask, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.):
        super().__init__(y, mask, noise_std, shape, model, eta, num_steps, dt, epsilon, beta_min, beta_max)
        self.mask = mask

    def analysis(self, x, t, timestep, v):
        x = x.flatten()
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t, timestep)
        x_0, vjp_estimate_x_0, epsilon = vjp(
            _estimate_x_0, x, has_aux=True)
        C_yy = 1. + self.noise_std**2 / v
        ls = vjp_estimate_x_0(self.mask * (self.y - x_0) / C_yy)[0]
        return x_0.reshape(self.shape), ls.reshape(self.shape), epsilon.reshape(self.shape)


class PiGDMVE(DDIMVE):
    """PiGDMVE for the SMLD Markov Chain or VE SDE."""
    def __init__(self, y, H, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
        super().__init__(model, eta, num_steps, dt, epsilon, sigma_min, sigma_max)
        self.eta = eta
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = jnp.exp(
            jnp.linspace(jnp.log(self.sigma_min),
                        jnp.log(self.sigma_max),
                        self.num_steps))
        self.H = H
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]
        self.estimate_x_0 = self.get_estimate_x_0(model, shape)
        self.batch_analysis = vmap(self.analysis)

    def get_estimate_x_0(self, model, shape):
        def estimate_x_0(x, t, timestep):
            std = self.discrete_sigmas[timestep]
            x = x.reshape(shape)
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            epsilon = model(x, t)
            epsilon = epsilon.flatten()
            x = x.flatten()
            x_0 = x - std * epsilon
            return x_0, epsilon
        return estimate_x_0

    def analysis(self, x, t, timestep, v):
        x = x.flatten()
        r = v / (v + 1.)
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t, timestep)
        x_0, vjp_estimate_x_0, epsilon = vjp(
            _estimate_x_0, x, has_aux=True)
        batch_vjp_estimate_x_0 = vmap(lambda x: vjp_estimate_x_0(x)[0])
        C_yy = 1. + self.noise_std**2 / r
        ls = vjp_estimate_x_0(self.H.T @ (self.y - self.H @ x_0) / C_yy)[0]
        return x_0.reshape(self.shape), ls.reshape(self.shape), epsilon.reshape(self.shape)

    def posterior(self, x, t):
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        sigma = self.discrete_sigmas[timestep]
        sigma_prev = self.discrete_sigmas_prev[timestep]
        x_0, ls, epsilon = self.batch_analysis(x, t, timestep, sigma**2)
        coeff1 = self.eta * jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
        coeff2 = jnp.sqrt(sigma_prev**2  - coeff1**2)
        x_mean = x_0 + batch_mul(coeff2, epsilon) + ls
        std = coeff1
        return x_mean, std


class PiGDMVEplus(PiGDMVE):
    """PiGDMVE with a mask."""
    def __init__(self, y, mask, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
        super().__init__(y, mask, noise_std, shape, model, eta, num_steps, dt, epsilon, sigma_min, sigma_max)
        self.mask = mask

    def analysis(self, x, t, timestep, v):
        x = x.flatten()
        r = v / (v + 1.)
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t, timestep)
        x_0, vjp_estimate_x_0, epsilon = vjp(
            _estimate_x_0, x, has_aux=True)
        C_yy = 1. + self.noise_std**2 / r
        ls = vjp_estimate_x_0(self.mask * (self.y - self.mask * x_0) / C_yy)[0]
        return x_0.reshape(self.shape), ls.reshape(self.shape), epsilon.reshape(self.shape)


class DPSSMLD(SMLD):
    """DPS."""
    def __init__(self, scale, y, H, shape, score, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
        super().__init__(score, num_steps, dt, epsilon, sigma_min, sigma_max)
        self.scale = scale
        self.shape = shape
        self.estimate_h_x_0 = self.get_estimate_h_x_0(score, shape, H)
        self.likelihood_score = self.get_likelihood_score(y, self.estimate_h_x_0)

    def get_estimate_h_x_0(self, score, shape, H):
        def estimate_h_x_0(x, t, timestep):
            v = self.discrete_sigmas[timestep]**2
            x = x.reshape(shape)
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            s = score(x, t)
            x = x.flatten()
            s = s.flatten()
            return H @ (x + v * s), s
        return estimate_h_x_0

    def get_likelihood_score(self, y, estimate_h_x_0):
        def l2_norm(x, t, timestep):
            x = x.flatten()
            # y is given as a d_x length vector
            h_x_0, s = estimate_h_x_0(x, t, timestep)
            innovation = y - h_x_0
            norm = jnp.linalg.norm(innovation)
            return norm, s  # l2 norm
        grad_l2_norm = grad(l2_norm, has_aux=True)
        def likelihood_score(x, t, timestep):
            return grad_l2_norm(x, t, timestep)
        return vmap(likelihood_score)

    def update(self, rng, x, t):
        """Return the update of the state and any auxilliary variables."""
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        ls, score = self.likelihood_score(x, t, timestep)
        # there must be a more efficient way than reshaping
        ls = ls.reshape((-1,) + self.shape)
        score = score.reshape((-1,) + self.shape)
        x_mean, std = self.posterior(score, x, timestep)
        x_mean -= self.scale * ls  # DPS
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean


class DPSSMLDplus(DPSSMLD):
    """
    DPSSMLD with a mask.
    NOTE: Stable and works on FFHQ
    """
    def get_estimate_h_x_0(self, score, shape, mask):
        def estimate_h_x_0(x, t, timestep):
            v = self.discrete_sigmas[timestep]**2
            x = x.reshape(shape)
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            s = score(x, t)
            s = s.flatten()
            x = x.flatten()
            return mask * (x + v * s), s
        return estimate_h_x_0

    def get_likelihood_score(self, y, estimate_h_x_0):
        def l2_norm(x, t, timestep):
            # y is given as a d_x length vector
            h_x_0, s = estimate_h_x_0(x, t, timestep)
            innovation = y - h_x_0
            norm = jnp.linalg.norm(innovation)
            return norm, s  # l2 norm
        grad_l2_norm = grad(l2_norm, has_aux=True)
        def likelihood_score(x, t, timestep):
            x = x.flatten()
            return grad_l2_norm(x, t, timestep)
        return vmap(likelihood_score)


class DPSDDPM(DDPM):
    """DPS."""
    def __init__(self, scale, y, H, shape, score, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.0):
        super().__init__(score, num_steps, dt, epsilon, beta_min, beta_max)
        self.scale = scale
        self.estimate_h_x_0 = self.get_estimate_h_x_0(score, shape, H)
        self.likelihood_score = self.get_likelihood_score(y, self.estimate_h_x_0)

    def get_estimate_h_x_0(self, score, shape, H):
        def estimate_h_x_0(x, t, timestep):
            m = self.sqrt_alphas_cumprod[timestep]
            v = self.sqrt_1m_alphas_cumprod[timestep]**2
            x = x.reshape(shape)
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            s = score(x, t)
            s = s.flatten()
            x = x.flatten()
            return H @ (x + v * s) / m, s
        return estimate_h_x_0

    def get_likelihood_score(self, y, estimate_h_x_0):
        def l2_norm(x, t, timestep):
            # y is given as a d_x length vector
            h_x_0, s = estimate_h_x_0(x, t, timestep)
            innovation = y - h_x_0
            norm = jnp.linalg.norm(innovation)
            return norm, s  # l2 norm
        grad_l2_norm = grad(l2_norm, has_aux=True)
        def likelihood_score(x, t, timestep):
            x = x.flatten()
            return grad_l2_norm(x, t, timestep)
        return vmap(likelihood_score)

    def update(self, rng, x, t):
        """Return the update of the state and any auxilliary variables."""
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        ls, score = self.likelihood_score(x, t, timestep)
        x_mean, std = self.posterior(score, x, timestep)
        x_mean -= self.scale * ls  # DPS
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean


class DPSDDPMplus(DPSDDPM):
    """DPS with a mask."""
    def get_estimate_h_x_0(self, score, shape, mask):
        def estimate_h_x_0(x, t, timestep):
            m = self.sqrt_alphas_cumprod[timestep]
            v = self.sqrt_1m_alphas_cumprod[timestep]**2
            x = x.reshape(shape)
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            s = score(x, t)
            s = s.flatten()
            x = x.flatten()
            return mask * (x + v * s) / m, s

    def get_likelihood_score(self, y, estimate_h_x_0, mask):
        assert jnp.shape(y) == jnp.shape(mask)
        def l2_norm(x, t):
            # y is given as a d_x length vector
            h_x_0, s = estimate_h_x_0(x, t)
            innovation = (y - h_x_0) * mask
            norm = jnp.linalg.norm(innovation)
            return norm, s  # l2 norm
        grad_l2_norm = grad(l2_norm, has_aux=True)
        def likelihood_score(x, t):
            x = x.flatten()
            return grad_l2_norm(x, t)
        return vmap(likelihood_score)


class KPDDPM(DDPM):
    """Kalman posterior for DDPM Ancestral sampling. TODO: needs debugging"""
    def __init__(self, y, H, noise_std, shape, score, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.0):
        super().__init__(score, num_steps, dt, epsilon, beta_min, beta_max)
        self.H = H
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]
        self.estimate_x_0 = self.get_estimate_x_0(score, shape)
        self.batch_analysis = vmap(self.analysis)

    def get_estimate_x_0(self, score, shape):
        def estimate_x_0(x, t, timestep):
            m = self.sqrt_alphas_cumprod[timestep]
            v = self.sqrt_1m_alphas_cumprod[timestep]**2
            x = x.reshape(shape)
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            s = score(x, t)
            s = s.flatten()
            x = x.flatten()
            x_0 = (x + v * s) / m
            return x_0, s
        return estimate_x_0

    def analysis(self, x, t, timestep, ratio):
        x = x.flatten()
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t, timestep)
        x_0, vjp_estimate_x_0, score = vjp(
            _estimate_x_0, x, has_aux=True)
        batch_vjp_estimate_x_0 = vmap(lambda x: vjp_estimate_x_0(x)[0])
        C_yy = batch_vjp_estimate_x_0(self.H) @ self.H.T + self.noise_std**2 / ratio * jnp.eye(self.num_y)
        ls = vjp_estimate_x_0(self.H.T @ jnp.linalg.solve(C_yy, self.y - self.H @ x_0))[0]  # nonsense image outputs
        # C_yy = 1 + self.noise_std**2 / ratio
        # ls = vjp_estimate_x_0(self.H.T @ (self.y - self.H @ x_0) / C_yy)[0]
        return (x_0 + ls).reshape(self.shape)

    def posterior(self, x, t):
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        beta = self.discrete_betas[timestep]
        m = self.sqrt_alphas_cumprod[timestep]
        v = self.sqrt_1m_alphas_cumprod[timestep]**2
        ratio = v / m
        x_dash = self.batch_analysis(x, t, timestep, ratio)
        alpha = self.alphas[timestep]
        # Kalman
        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
        x_mean = batch_mul(jnp.sqrt(alpha) * v_prev / v, x) + batch_mul(m_prev * beta / v, x_dash)
        std = jnp.sqrt(beta * v_prev / v)
        return x_mean, std

    def update(self, rng, x, t):
        """Return the update of the state and any auxilliary variables."""
        x_mean, std = self.posterior(x, t)
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean


class KPDDPMplus(KPDDPM):
    """Kalman posterior for DDPM Ancestral sampling. TODO: needs debugging"""
    def __init__(self, y, mask, noise_std, shape, score, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.0):
        super().__init__(y, mask, noise_std, shape, score, num_steps, dt, epsilon, beta_min, beta_max)
        self.mask = mask

    def analysis(self, x, t, timestep, ratio):
        x = x.flatten()
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t, timestep)
        x_0, vjp_estimate_x_0, score = vjp(
            _estimate_x_0, x, has_aux=True)
        C_yy = vjp_estimate_x_0(self.mask) + self.noise_std**2 / ratio
        ls = vjp_estimate_x_0(self.mask * (self.y - x_0) / C_yy)[0]
        # C_yy = 1 + self.noise_std**2 / ratio
        # ls = vjp_estimate_x_0(self.mask (self.y - x_0) / C_yy)[0]
        return (x_0 + ls).reshape(self.shape)


class KPSMLD(SMLD):
    """Kalman posterior for DDPM Ancestral sampling. TODO: needs debugging"""
    def __init__(self, y, H, noise_std, shape, score, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
        super().__init__(score, num_steps, dt, epsilon, sigma_min, sigma_max)
        self.H = H
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]
        self.estimate_x_0 = self.get_estimate_x_0(score, shape)
        self.batch_analysis = vmap(self.analysis)

    def get_estimate_x_0(self, score, shape):
        def estimate_x_0(x, t, timestep):
            v = self.discrete_sigmas[timestep]**2
            x = x.reshape(shape)
            x = jnp.expand_dims(x, axis=0)
            t = jnp.expand_dims(t, axis=0)
            s = score(x, t)
            s = s.flatten()
            x = x.flatten()
            x_0 = x + v * s
            return x_0, s
        return estimate_x_0

    def analysis(self, x, t, timestep, v):
        x = x.flatten()
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t, timestep)
        x_0, vjp_estimate_x_0, score = vjp(
            _estimate_x_0, x, has_aux=True)
        batch_vjp_estimate_x_0 = vmap(lambda x: vjp_estimate_x_0(x)[0])
        C_yy = batch_vjp_estimate_x_0(self.H) @ self.H.T + self.noise_std**2 / v * jnp.eye(self.num_y)
        ls = vjp_estimate_x_0(self.H.T @ jnp.linalg.solve(C_yy, self.y - self.H @ x_0))[0]  # nonsense image outputs
        # C_yy = 1. + self.noise_std**2 / v
        # ls = vjp_estimate_x_0(self.H.T @ (self.y - self.H @ x_0) / C_yy)[0]
        return (x_0 + ls).reshape(self.shape)

    def posterior(self, x, t):
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        sigma = self.discrete_sigmas[timestep]
        sigma_prev = self.discrete_sigmas_prev[timestep]
        x_0 = self.batch_analysis(x, t, timestep, sigma**2)
        x_mean = batch_mul(sigma_prev**2 / sigma**2, x) + batch_mul(1 - sigma_prev**2 / sigma**2, x_0)
        std = jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
        return x_mean, std

    def update(self, rng, x, t):
        """Return the update of the state and any auxilliary variables."""
        x_mean, std = self.posterior(x, t)
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean


class KPSMLDplus(KPSMLD):
    """
    Kalman posterior for DDPM Ancestral sampling.

    TODO: Intermittently stable on FFHQ and CelebA. Increasing noise_std makes more stable.
    TODO: needs debugging. Try taking vjp of ratio * x_0, may be more stable? Or using epsilon network?"""
    def __init__(self, y, mask, noise_std, shape, score, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
        super().__init__(y, mask, noise_std, shape, score, num_steps, dt, epsilon, sigma_min, sigma_max)
        self.mask = mask

    def analysis(self, x, t, timestep, ratio):
        x = x.flatten()
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t, timestep)
        x_0, vjp_estimate_x_0, score = vjp(
            _estimate_x_0, x, has_aux=True)
        C_yy = vjp_estimate_x_0(self.mask)[0] + self.noise_std**2 / ratio
        ls = vjp_estimate_x_0(self.mask * (self.y - x_0) / C_yy)[0]
        # C_yy = 1 + self.noise_std**2 / ratio
        # ls = vjp_estimate_x_0(self.mask * (self.y - x_0) / C_yy)[0]
        return (x_0 + ls).reshape(self.shape)
