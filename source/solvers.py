"""Solver classes, including Markov Chains."""
import jax.numpy as jnp
from jax import random, grad, vmap, vjp, jacrev
from diffusionjax.utils import batch_mul
from diffusionjax.solvers import Solver, DDIMVP, DDIMVE, SMLD, DDPM


class PKF(Solver):
    """Not an Projection Kalman filter. Abstract class for all concrete Projection Kalman Filter solvers.
    Functions are designed for an ensemble of inputs.
    """
    def __init__(self, num_y, y, noise_std, shape, sde, observation_map, num_steps=1000):
        """Construct a Projected Moment Kalman Filter Solver.
        Args:
            shape: Shape of array, x. (num_samples,) + x_shape, where x_shape is the shape
                of the object being sampled from, for example, an image may have
                x_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
            sde: A valid SDE class.
            num_steps: number of discretization time steps.
        """
        super().__init__(num_steps)
        self.y = y
        self.noise_std = noise_std
        self.sde = sde
        self.shape = shape
        self.observation_map = observation_map
        self.batch_observation_map = vmap(observation_map)
        self.estimate_h_x_0 = self.get_estimate_h_x_0(self.sde, shape[1:], observation_map)
        self.num_y = num_y
        self.prior = self.get_prior(sde)

    def get_prior(self, sde):
        if type(sde).__name__=='RVE':
            def prior(rng, shape):
                return random.normal(rng, shape) * self.sigma_max
        elif type(sde).__name__=='RVP':
            def prior(rng, shape):
                return random.normal(rng, shape)
        else:
            raise ValueError("Did not recognise reverse SDE (got {}, expected VE or VP)".format(type(sde).__name__))
        return prior

    def get_estimate_x_0(self, sde, shape, observation_map):
        # TODO: can this be included as a method in the in sde class
        if type(sde).__name__=='RVE':
            def estimate_x_0(x, t):
                v_t = sde.variance(t)
                x = x.reshape(shape)
                x = jnp.expand_dims(x, axis=0)
                t = jnp.expand_dims(t, axis=0)
                s = sde.score(x, t)
                s = s.flatten()
                x = x.flatten()
                x_0 = x + v_t * s
                return observation_map(x_0), (s, x_0)

        elif type(sde).__name__=='RVP':
            def estimate_x_0(x, t):
                m_t = sde.mean_coeff(t)
                v_t = sde.variance(t)
                x = x.reshape(shape)
                x = jnp.expand_dims(x, axis=0)
                t = jnp.expand_dims(t, axis=0)
                s = sde.score(x, t)
                s = s.flatten()
                x = x.flatten()
                x_0 = (x + v_t * s) / m_t
                return observation_map(x_0), (s, x_0)

        else:
            raise ValueError("Did not recognise reverse SDE (got {}, expected VE or VP)".format(type(sde).__name__))
        return estimate_h_x_0

    def batch_observation_map(self, x, t):
        return vmap(lambda x: self.observation_map(x, t))(x)

    def batch_analysis(self, x, t, v):
        return vmap(lambda x, t: self.analysis(x, t, v), in_axes=(0, 0))(x, t)

    def predict(self, rng, x, t):
        x = x.reshape(self.shape)
        drift, diffusion = self.sde.sde(x, t)
        # TODO: possible reshaping needs to occur here, if score
        # applies to an image vector
        alpha = self.sde.mean_coeff(t)[0]**2
        R = alpha * self.noise_std**2
        eta = jnp.sqrt(R)
        f = drift * self.dt
        G = diffusion * jnp.sqrt(self.dt)
        noise = random.normal(rng, x.shape)
        x_hat_mean = x + f
        x_hat = x_hat_mean + batch_mul(G, noise)
        x_hat_mean = x_hat_mean.reshape(self.shape[0], -1)
        x_hat = x_hat.reshape(self.shape[0], -1)
        return x_hat, x_hat_mean

    def update(self, rng, x, t):
        r"""Return the drift and diffusion coefficients of the SDE.

        Args:

        Returns:
            x: A JAX array of the next state.
            x_mean: A JAX array of the next state without noise, for denoising.
        """
        t = t.flatten()
        v = self.sde.variance(t)[0]
        sqrt_v = jnp.sqrt(v)
        sqrt_alpha = self.sde.mean_coeff(t)[0]
        ratio = v / sqrt_alpha  # TODO: generalize across sdes
        x_hat, x_hat_mean = self.predict(rng, x, t)
        m = self.batch_analysis(x_hat, t, ratio)  # denoise x and perform kalman update
        # Missing a step here where x_0 is sampled as N(m, C), which makes Gaussian case exact probably
        x = sqrt_alpha * m + jnp.sqrt(v) * random.normal(rng, x.shape)  # renoise denoised x
        # x = sqrt_alpha * m - jnp.sqrt(v) * score # renoise denoised x
        # x = sqrt_alpha * m - v * score # renoise denoised x
        return x, m
        # Summary:
        # x = x_hat + v * score + sqrt_alpha * C_xh_hat @ jnp.linalg.solve(C_yy_hat, y - o_hat) + sqrt_v * random.normal(rng, x.shape)

    def analysis(self, x, t, ratio):
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0(x, t)
        h_x_0, s, x_0 = self.estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0(_x, t)[0])(x)
        H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
        Cyy = ratio * H_grad_H_x_0 + self.noise_std**2 * jnp.eye(self.num_y)
        innovation = self.y - h_x_0
        f = jnp.linalg.solve(Cyy, innovation)
        ls = ratio * grad_H_x_0.T @ f
        return x_0 + ls # , score


class KGDMVP(DDIMVP):
    """PiGDM Song et al. 2021. Markov chain using the DDIM Markov Chain or VP SDE. TODO: needs debugging"""
    def __init__(self, y, observation_map, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.):
        super().__init__(model, eta, num_steps, dt, epsilon, beta_min, beta_max)
        self.estimate_h_x_0 = self.get_estimate_x_0(shape, observation_map)
        self.batch_analysis = vmap(self.analysis)
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]
        self.observation_map = observation_map
        self.batch_observation_map = vmap(observation_map)

    def analysis(self, x, t, timestep, ratio):
        x = x.flatten()
        h_x_0, (epsilon, x_0) = self.estimate_h_x_0(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0(_x, t, timestep)[0])(x)
        H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
        Cyy = ratio * H_grad_H_x_0 + self.noise_std**2 * jnp.eye(self.num_y)
        f = jnp.linalg.solve(Cyy, self.y - h_x_0)
        ls = grad_H_x_0.T @ f
        return epsilon.reshape(self.shape), ls.reshape(self.shape)

    def posterior(self, x, t):
        timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
        m = self.sqrt_alphas_cumprod[timestep]
        sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
        v = sqrt_1m_alpha**2
        ratio = v / m
        alpha = m**2
        epsilon, ls = self.batch_analysis(x, t, timestep, ratio)
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
    def analysis(self, x, t, timestep, ratio):
        # TODO: do I need to flatten or is there a better way?
        x = x.flatten()
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0(x, t, timestep)
        h_x_0, vjp_estimate_h_x_0, (epsilon, x_0) = vjp(
            _estimate_h_x_0, x, has_aux=True)
        C_yy = ratio * vjp_estimate_h_x_0(self.observation_map(jnp.ones_like(x)))[0] + self.noise_std**2
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
        C_yy = v * vjp_estimate_h_x_0(self.observation_map(jnp.ones_like(x)))[0] + self.noise_std**2
        ls = vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
        return epsilon.reshape(self.shape), ls.reshape(self.shape)


class PiGDMVP(DDIMVP):
    """ TODO: this is not identical to PiGDMVP in grfjax.
    PiGDM Song et al. 2021. Markov chain using the DDIM Markov Chain or VP SDE."""
    def __init__(self, y, observation_map, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.):
        super().__init__(model, eta, num_steps, dt, epsilon, beta_min, beta_max)
        self.estimate_h_x_0 = self.get_estimate_x_0(shape, observation_map)
        self.batch_analysis = vmap(self.analysis)
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]

    def analysis(self, x, t, timestep, r):
        x = x.flatten()
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0(x, t, timestep)
        h_x_0, vjp_estimate_h_x_0, (epsilon, x_0) = vjp(
            _estimate_h_x_0, x, has_aux=True)
        C_yy = 1 + self.noise_std**2 / r
        ls = vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
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
    def analysis(self, x, t, timestep, v):
        x = x.flatten()
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0(x, t, timestep)
        h_x_0, vjp_estimate_h_x_0, (epsilon, x_0) = vjp(
            _estimate_h_x_0, x, has_aux=True)
        C_yy = 1. + self.noise_std**2 / v
        ls = vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
        return x_0.reshape(self.shape), ls.reshape(self.shape), epsilon.reshape(self.shape)


class PiGDMVE(DDIMVE):
    """PiGDMVE for the SMLD Markov Chain or VE SDE."""
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
        self.batch_observation_map = vmap(observation_map)

    def analysis(self, x, t, timestep, v):
        x = x.flatten()
        r = v / (v + 1.)
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


class PiGDMVEplus(PiGDMVE):
    """PiGDMVE with a mask."""
    def analysis(self, x, t, timestep, v):
        x = x.flatten()
        r = v / (v + 1.)
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0(x, t, timestep)
        h_x_0, vjp_estimate_x_0, (epsilon, x_0) = vjp(
            _estimate_h_x_0, x, has_aux=True)
        C_yy = 1. + self.noise_std**2 / r
        ls = vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
        return x_0.reshape(self.shape), ls.reshape(self.shape), epsilon.reshape(self.shape)


class DPSSMLD(SMLD):
    """DPS."""
    def __init__(self, scale, y, observation_map, shape, score, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
        super().__init__(score, num_steps, dt, epsilon, sigma_min, sigma_max)
        self.scale = scale
        self.shape = shape
        self.estimate_h_x_0 = self.get_estimate_x_0(shape, observation_map)
        self.likelihood_score = self.get_likelihood_score(y, self.estimate_h_x_0)

    def get_likelihood_score(self, y, estimate_h_x_0):
        def l2_norm(x, t, timestep):
            h_x_0, (s, _) = estimate_h_x_0(x, t, timestep)
            innovation = y - h_x_0
            norm = jnp.linalg.norm(innovation)
            return norm, s
        grad_l2_norm = grad(l2_norm, has_aux=True)
        def likelihood_score(x, t, timestep):
            x = x.flatten()
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


DPSSMLDplus = DPSSMLD


class DPSDDPM(DDPM):
    """DPS."""
    def __init__(self, scale, y, observation_map, shape, score, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.0):
        super().__init__(score, num_steps, dt, epsilon, beta_min, beta_max)
        self.scale = scale
        self.estimate_h_x_0 = self.get_estimate_x_0(shape, observation_map)
        self.likelihood_score = self.get_likelihood_score(y, self.estimate_h_x_0)

    def get_likelihood_score(self, y, estimate_h_x_0):
        def l2_norm(x, t, timestep):
            h_x_0, (s, _) = estimate_h_x_0(x, t, timestep)
            innovation = y - h_x_0
            norm = jnp.linalg.norm(innovation)
            return norm, s
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
    def get_likelihood_score(self, y, estimate_h_x_0):
        def l2_norm(x, t):
            # y is given as a d_x length vector
            h_x_0, (s, _) = estimate_h_x_0(x, t)
            norm = jnp.linalg.norm(y - h_x_0)
            return norm, s  # l2 norm
        grad_l2_norm = grad(l2_norm, has_aux=True)
        def likelihood_score(x, t):
            x = x.flatten()
            return grad_l2_norm(x, t)
        return vmap(likelihood_score)


class KPDDPM(DDPM):
    """Kalman posterior for DDPM Ancestral sampling. TODO: needs debugging"""
    def __init__(self, y, observation_map, noise_std, shape, score, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.0):
        super().__init__(score, num_steps, dt, epsilon, beta_min, beta_max)
        self.y = y
        self.noise_std = noise_std
        self.shape = shape
        self.num_y = y.shape[0]
        self.estimate_h_x_0 = self.get_estimate_x_0(shape, observation_map)
        self.batch_analysis = vmap(self.analysis)
        self.observation_map = observation_map
        self.batch_observation_map = vmap(observation_map)

    def analysis(self, x, t, timestep, ratio):
        x = x.flatten()
        h_x_0, (s, x_0) = self.estimate_h_x_0(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0(_x, t, timestep)[0])(x)
        H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
        Cyy = H_grad_H_x_0 + self.noise_std**2 / ratio * jnp.eye(self.num_y)
        innovation = self.y - h_x_0
        f = jnp.linalg.solve(Cyy, innovation)
        ls = grad_H_x_0.T @ f
        # posterior_score = s + ls / ratio
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
    def analysis(self, x, t, timestep, ratio):
        x = x.flatten()
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0(x, t, timestep)
        h_x_0, vjp_estimate_h_x_0, (score, x_0) = vjp(
            _estimate_h_x_0, x, has_aux=True)
        C_yy = vjp_estimate_h_x_0(self.observation_map(jnp.ones_like(x)))[0] + self.noise_std**2 / ratio
        ls = vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
        # C_yy = 1 + self.noise_std**2 / ratio
        # ls = vjp_estimate_x_0((self.y - x_0) / C_yy)[0]
        return (x_0 + ls).reshape(self.shape)


class KPSMLD(SMLD):
    """Kalman posterior for DDPM Ancestral sampling. TODO: needs debugging"""
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
        # C_yy = 1. + self.noise_std**2 / v
        # ls = vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
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
    def analysis(self, x, t, timestep, ratio):
        x = x.flatten()
        _estimate_h_x_0 = lambda x: self.estimate_h_x_0(x, t, timestep)
        h_x_0, vjp_estimate_h_x_0, (score, x_0) = vjp(
            _estimate_h_x_0, x, has_aux=True)
        C_yy = vjp_estimate_h_x_0(self.observation_map(jnp.ones_like(x)))[0] + self.noise_std**2 / ratio
        ls = vjp_estimate_h_x_0((self.y - h_x_0) / C_yy)[0]
        # C_yy = 1 + self.noise_std**2 / ratio
        # ls = vjp_estimate_x_0((self.y - h_x_0) / C_yy)[0]
        return (x_0 + ls).reshape(self.shape)
