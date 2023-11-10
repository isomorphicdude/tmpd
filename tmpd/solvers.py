"""Markov Chains."""
import jax.numpy as jnp
from jax import random, grad, vmap, vjp, jacrev
from diffusionjax.utils import batch_mul, batch_matmul, batch_linalg_solve, batch_mul_A
from diffusionjax.solvers import DDIMVP, DDIMVE, SMLD, DDPM


class KGDMVP(DDIMVP):
    """PiGDM Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""
    def __init__(self, y, observation_map, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.):
        super().__init__(model, eta, num_steps, dt, epsilon, beta_min, beta_max)
        self.estimate_h_x_0 = self.get_estimate_x_0(observation_map)
        self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(shape, observation_map)
        self.batch_analysis_vmap = vmap(self.analysis)
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]
        self.observation_map = observation_map
        self.batch_observation_map = vmap(observation_map)
        self.batch_batch_observation_map = vmap(vmap(observation_map))
        self.grad_H_x_0 = vmap(lambda x, t, timestep: jacrev(lambda _x: self.estimate_h_x_0(jnp.expand_dims(_x, axis=0), jnp.expand_dims(t, axis=0), jnp.expand_dims(timestep, axis=0))[0])(x))  # sorry...

    def analysis(self, x, t, timestep, ratio):
        x = x.flatten()
        h_x_0, (epsilon, _) = self.estimate_h_x_0_vmap(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
        H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
        Cyy = ratio * H_grad_H_x_0 + self.noise_std**2 * jnp.eye(self.num_y)
        f = jnp.linalg.solve(Cyy, self.y - h_x_0)
        ls = grad_H_x_0.T @ f
        return epsilon.reshape(self.shape[1:]), ls.reshape(self.shape[1:])

    def batch_analysis(self, x, t, timestep, ratio):
        h_x_0, (epsilon, _) = self.estimate_h_x_0(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = self.grad_H_x_0(x, t, timestep)
        grad_H_x_0 = jnp.squeeze(grad_H_x_0, axis=1)
        H_grad_H_x_0 = self.batch_batch_observation_map(grad_H_x_0)
        Cyy = batch_mul(ratio, H_grad_H_x_0) + self.noise_std**2 * jnp.eye(self.num_y)
        f = batch_linalg_solve(Cyy, self.y - h_x_0)
        ls = batch_matmul(grad_H_x_0.transpose(0, 2, 1), f)
        return epsilon, ls

    def posterior(self, x, t):
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        m = self.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
        v = sqrt_1m_alpha**2
        ratio = v / m
        alpha = m**2
        # epsilon, ls = self.batch_analysis(x, t, timestep, ratio)
        epsilon, ls = self.batch_analysis_vmap(x, t, timestep, ratio)
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
    """KGDMVP with a mask."""
    def analysis(self, x, t, timestep, ratio):
        x = x.flatten()
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0(x, t, timestep)
        h_x_0, vjp_estimate_h_x_0, (epsilon, x_0) = vjp(
            _estimate_h_x_0, x, has_aux=True)
        C_yy = ratio * self.observation_map(vjp_estimate_h_x_0(self.observation_map(jnp.ones_like(x)))[0]) + self.noise_std**2
        ls = vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
        return epsilon.reshape(self.shape), ls.reshape(self.shape)


class KGDMVE(DDIMVE):
    def __init__(self, y, observation_map, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
        super().__init__(model, eta, num_steps, dt, epsilon, sigma_min, sigma_max)
        self.eta = eta
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = jnp.exp(
            jnp.linspace(jnp.log(self.sigma_min),
                         jnp.log(self.sigma_max),
                         self.num_steps))
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]
        self.estimate_h_x_0 = self.get_estimate_x_0(shape, observation_map)
        self.batch_analysis = vmap(self.analysis)
        self.observation_map = observation_map
        self.batch_observation_map  = vmap(observation_map)

    def analysis(self, x, t, timestep, v):
        x = x.flatten()
        h_x_0, (epsilon, x_0) = self.estimate_h_x_0(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0(_x, t, timestep)[0])(x)
        H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
        Cyy = v * H_grad_H_x_0 + self.noise_std**2 * jnp.eye(self.num_y)
        f = jnp.linalg.solve(Cyy, self.y - h_x_0)
        ls = grad_H_x_0.T @ f
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
    def analysis(self, x, t, timestep, v):
        x = x.flatten()
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0(x, t, timestep)
        h_x_0, vjp_estimate_h_x_0, (epsilon, x_0) = vjp(
            _estimate_h_x_0, x, has_aux=True)
        C_yy = v * self.observation_map(vjp_estimate_h_x_0(self.observation_map(jnp.ones_like(x)))[0]) + self.noise_std**2
        ls = vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
        return epsilon.reshape(self.shape), ls.reshape(self.shape)


class PiGDMVP(DDIMVP):
    """PiGDM Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""
    def __init__(self, y, observation_map, noise_std, shape, model, data_variance=1.0, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.):
        super().__init__(model, eta, num_steps, dt, epsilon, beta_min, beta_max)
        self.estimate_h_x_0 = self.get_estimate_x_0(observation_map)
        self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(shape, observation_map)
        self.batch_analysis_vmap = vmap(self.analysis)
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.data_variance = data_variance
        self.num_y = y.shape[0]
        self.observation_map = observation_map
        self.batch_observation_map = vmap(observation_map)
        self.grad_H_x_0 = vmap(lambda x, t, timestep: jacrev(lambda _x: self.estimate_h_x_0(jnp.expand_dims(_x, axis=0), jnp.expand_dims(t, axis=0), jnp.expand_dims(timestep, axis=0))[0])(x))  # sorry...

    def analysis(self, x, t, timestep, v, alpha):
        x = x.flatten()
        h_x_0, (epsilon, _) = self.estimate_h_x_0_vmap(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
        # Value suggested for VPSDE in original PiGDM paper
        r = v * self.data_variance  / (v + alpha * self.data_variance)
        Cyy = r + self.noise_std**2
        f = (self.y - h_x_0) / Cyy
        ls = grad_H_x_0.T @ f
        return epsilon.reshape(self.shape[1:]), ls.reshape(self.shape[1:])

    def batch_analysis(self, x, t, timestep, v, alpha):
        h_x_0, (epsilon, _) = self.estimate_h_x_0(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = self.grad_H_x_0(x, t, timestep)
        grad_H_x_0 = jnp.squeeze(grad_H_x_0, axis=1)
        # Value suggested for VPSDE in original PiGDM paper
        r = v * self.data_variance  / (v + alpha * self.data_variance)
        Cyy = r + self.noise_std**2
        f = batch_mul((self.y - h_x_0), 1. / Cyy)
        ls = batch_matmul(jnp.transpose(grad_H_x_0, axes=(0, 2, 1)), f)
        return epsilon, ls

    def posterior(self, x, t):
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        m = self.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
        v = sqrt_1m_alpha**2
        alpha = m**2
        epsilon, ls = self.batch_analysis_vmap(x, t, timestep, v, alpha)
        # epsilon, ls = self.batch_analysis(x, t, timestep, v, alpha)
        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
        alpha_prev = m_prev**2
        coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
        coeff2 = jnp.sqrt(v_prev - coeff1**2)
        posterior_score = - batch_mul(1. / sqrt_1m_alpha, epsilon) + ls
        x_mean = batch_mul(m_prev / m, x) + batch_mul(sqrt_1m_alpha * (sqrt_1m_alpha * m_prev / m - coeff2), posterior_score)
        std = coeff1
        return x_mean, std


class PiGDMVPplus(PiGDMVP):
    """KGDMVP with a mask."""
    def analysis(self, x, t, timestep, v, alpha):
        x = x.flatten()
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0(x, t, timestep)
        h_x_0, vjp_estimate_h_x_0, (epsilon, _) = vjp(
            _estimate_h_x_0, x, has_aux=True)
        # Value suggested for VPSDE in original PiGDM paper
        r = variance * self.data_variance  / (variance + alpha * self.data_variance)
        C_yy = r + self.noise_std**2
        ls = vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
        return epsilon.reshape(self.shape), ls.reshape(self.shape)


class SSPiGDMVP(DDIMVP):
    """
    Note: We found this method to be unstable on all datasets.
    PiGDM Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""
    def __init__(self, y, observation_map, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.):
        super().__init__(model, eta, num_steps, dt, epsilon, beta_min, beta_max)
        self.estimate_h_x_0 = self.get_estimate_x_0(shape, observation_map)
        self.batch_analysis = vmap(self.analysis)
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]

    def analysis(self, x, t, timestep, v, alpha):
        x = x.flatten()
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0(x, t, timestep)
        h_x_0, vjp_estimate_h_x_0, (epsilon, x_0) = vjp(
            _estimate_h_x_0, x, has_aux=True)
        # Value suggested for VPSDE in original PiGDM paper
        r = variance * self.data_variance  / (variance + alpha * self.data_variance)
        C_yy = r + self.noise_std**2
        ls = r * vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
        return x_0.reshape(self.shape), ls.reshape(self.shape), epsilon.reshape(self.shape)

    def posterior(self, x, t):
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        m = self.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
        v = sqrt_1m_alpha**2
        alpha = m**2
        x_0, ls, epsilon = self.batch_analysis(x, t, timestep, v, alpha)
        m_prev = self.sqrt_alphas_cumprod_prev[timestep]
        v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
        alpha_prev = m_prev**2
        coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
        coeff2 = jnp.sqrt(v_prev - coeff1**2)
        x_mean = batch_mul(m_prev, x_0) + batch_mul(coeff2, epsilon) + batch_mul(m, ls)
        std = coeff1
        return x_mean, std


class PiGDMVE(DDIMVE):
    """PiGDMVE for the SMLD Markov Chain or VE SDE."""
    def __init__(self, y, observation_map, noise_std, shape, model, data_variance=1.0, eta=0.0, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
        super().__init__(model, eta, num_steps, dt, epsilon, sigma_min, sigma_max)
        self.eta = eta
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = jnp.exp(
            jnp.linspace(jnp.log(self.sigma_min),
                        jnp.log(self.sigma_max),
                        self.num_steps))
        self.y = y
        self.data_variance = data_variance
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]
        self.estimate_h_x_0 = self.get_estimate_x_0(shape, observation_map)
        self.batch_analysis = vmap(self.analysis)
        self.batch_observation_map = vmap(observation_map)

    def analysis(self, x, t, timestep, v):
        x = x.flatten()
        r = v * self.data_variance / (variance + self.data_variance)
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0(x, t, timestep)
        h_x_0, vjp_estimate_h_x_0, (epsilon, x_0) = vjp(
            _estimate_h_x_0, x, has_aux=True)
        C_yy = 1 + self.noise_std**2 / r
        ls = vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
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


PiGDMVEplus = PiGDMVE


class DPSSMLD(SMLD):
    """DPS."""
    def __init__(self, scale, y, observation_map, shape, score, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
        super().__init__(score, num_steps, dt, epsilon, sigma_min, sigma_max)
        self.scale = scale
        self.shape = shape
        self.estimate_h_x_0 = self.get_estimate_x_0(shape, observation_map)
        self.likelihood_score = self.get_likelihood_score(y, self.estimate_h_x_0, shape)

    def get_likelihood_score(self, y, estimate_h_x_0, shape):
        def l2_norm(x, t, timestep):
            h_x_0, (s, _) = estimate_h_x_0(x, t, timestep)
            innovation = y - h_x_0
            norm = jnp.linalg.norm(innovation)
            return norm, s.reshape(shape)
        grad_l2_norm = grad(l2_norm, has_aux=True)
        return vmap(grad_l2_norm)

    def update(self, rng, x, t):
        """Return the update of the state and any auxilliary variables."""
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        ls, score = self.likelihood_score(x, t, timestep)
        x_mean, std = self.posterior(score, x, timestep)
        x_mean -= self.scale * ls  # DPS
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean


DPSSMLDplus = DPSSMLD


class DPSDDPM(DDPM):
    """DPS."""
    def __init__(self, scale, y, observation_map, shape, score, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.0):
        super().__init__(score, num_steps, dt, epsilon, beta_min, beta_max)
        self.scale = scale
        self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(shape, observation_map)
        self.estimate_h_x_0 = self.get_estimate_x_0(observation_map)
        self.likelihood_score = self.get_likelihood_score(y)
        self.likelihood_score_vmap = self.get_likelihood_score_vmap(y, shape)

    def get_likelihood_score_vmap(self, y, shape):
        def l2_norm(x, t, timestep):
            h_x_0, (s, _) = self.estimate_h_x_0_vmap(x, t, timestep)
            norm = jnp.linalg.norm(y - h_x_0)
            return norm, s.reshape(shape[1:])
        grad_l2_norm = grad(l2_norm, has_aux=True)
        return vmap(grad_l2_norm)

    def get_likelihood_score(self, y):
        batch_norm = vmap(jnp.linalg.norm)
        def l2_norm(x, t, timestep):
            h_x_0, (s, _) = self.estimate_h_x_0(x, t, timestep)
            norm = batch_norm(y - h_x_0)
            norm = jnp.sum(norm)
            return norm, s
        grad_l2_norm = grad(l2_norm, has_aux=True)
        return grad_l2_norm

    def update(self, rng, x, t):
        """Return the update of the state and any auxilliary variables."""
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        ls, score = self.likelihood_score(x, t, timestep)
        # ls, score = self.likelihood_score_vmap(x, t, timestep)
        x_mean, std = self.posterior(score, x, timestep)
        x_mean -= self.scale * ls  # DPS
        z = random.normal(rng, x.shape)
        x = x_mean + batch_mul(std, z)
        return x, x_mean


DPSDDPMplus = DPSDDPM


class KPDDPM(DDPM):
    """Kalman posterior for DDPM Ancestral sampling."""
    def __init__(self, y, observation_map, noise_std, shape, score, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.0):
        super().__init__(score, num_steps, dt, epsilon, beta_min, beta_max)
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]
        self.estimate_h_x_0 = self.get_estimate_x_0(observation_map)
        self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(shape, observation_map)
        self.batch_analysis_vmap = vmap(self.analysis)
        self.observation_map = observation_map
        self.batch_observation_map = vmap(observation_map)
        self.batch_batch_observation_map = vmap(vmap(observation_map))
        self.grad_H_x_0 = vmap(lambda x, t, timestep: jacrev(lambda _x: self.estimate_h_x_0(jnp.expand_dims(_x, axis=0), jnp.expand_dims(t, axis=0), jnp.expand_dims(timestep, axis=0))[0])(x))  # sorry...

    def analysis(self, x, t, timestep, ratio):
        x = x.flatten()
        h_x_0, (_, x_0) = self.estimate_h_x_0_vmap(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
        H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
        Cyy = H_grad_H_x_0 + self.noise_std**2 / ratio * jnp.eye(self.num_y)
        innovation = self.y - h_x_0
        f = jnp.linalg.solve(Cyy, innovation)
        ls = grad_H_x_0.T @ f
        return (x_0 + ls).reshape(self.shape[1:])

    def batch_analysis(self, x, t, timestep, ratio):
        h_x_0, (_, x_0) = self.estimate_h_x_0(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = self.grad_H_x_0(x, t, timestep)
        grad_H_x_0 = jnp.squeeze(grad_H_x_0, axis=1)
        H_grad_H_x_0 = self.batch_batch_observation_map(grad_H_x_0)
        Cyy = H_grad_H_x_0 + batch_mul_A(jnp.eye(self.num_y), self.noise_std**2 / ratio)
        innovation = self.y - h_x_0
        f = batch_linalg_solve(Cyy, innovation)
        ls = batch_matmul(grad_H_x_0.transpose(0, 2, 1), f)
        return x_0 + ls

    def posterior(self, x, t):
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        beta = self.discrete_betas[timestep]
        m = self.sqrt_alphas_cumprod[timestep]
        v = self.sqrt_1m_alphas_cumprod[timestep]**2
        ratio = v / m
        # x_dash = self.batch_analysis(x, t, timestep, ratio)
        x_dash = self.batch_analysis_vmap(x, t, timestep, ratio)
        alpha = self.alphas[timestep]
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
    """Kalman posterior for DDPM Ancestral sampling."""
    def analysis(self, x, t, timestep, ratio):
        x = x.flatten()
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0(x, t, timestep)
        h_x_0, vjp_estimate_h_x_0, (score, x_0) = vjp(
            _estimate_h_x_0, x, has_aux=True)
        C_yy = self.observation_map(vjp_estimate_h_x_0(self.observation_map(jnp.ones_like(x)))[0]) + self.noise_std**2 / ratio
        # C_yy = vjp_estimate_h_x_0(self.observation_map(jnp.ones_like(x)))[0] + self.noise_std**2 / ratio
        ls = vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
        return (x_0 + ls).reshape(self.shape)


class KPSMLD(SMLD):
    """Kalman posterior for DDPM Ancestral sampling."""
    def __init__(self, y, observation_map, noise_std, shape, score, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
        super().__init__(score, num_steps, dt, epsilon, sigma_min, sigma_max)
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]
        self.estimate_h_x_0 = self.get_estimate_x_0(shape, observation_map)
        self.batch_analysis = vmap(self.analysis)
        self.observation_map = observation_map
        self.batch_observation_map = vmap(observation_map)

    def analysis(self, x, t, timestep, v):
        x = x.flatten()
        h_x_0, (s, x_0) = self.estimate_h_x_0(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0(_x, t, timestep)[0])(x)
        H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
        Cyy = H_grad_H_x_0 + self.noise_std**2 / v * jnp.eye(self.num_y)
        innovation = self.y - h_x_0
        f = jnp.linalg.solve(Cyy, innovation)
        ls = grad_H_x_0.T @ f
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
    """
    def analysis(self, x, t, timestep, ratio):
        x = x.flatten()
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0_vmap(x, t, timestep)
        h_x_0, vjp_estimate_h_x_0, (_, x_0) = vjp(
            _estimate_h_x_0, x, has_aux=True)
        C_yy = self.observation_map(vjp_estimate_h_x_0(self.observation_map(jnp.ones_like(x)))[0]) + self.noise_std**2 / ratio
        ls = vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
        return (x_0 + ls).reshape(self.shape)
