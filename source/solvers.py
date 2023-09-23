"""Solver classes, including Markov Chains."""
import jax.numpy as jnp
from jax import random, grad, vmap, vjp
from diffusionjax.utils import batch_mul
from diffusionjax.solvers import Solver, DDIMVP, DDIMVE, SMLD, DDPM


class PKF(Solver):
    """Not an Projection Kalman filter. Abstract class for all concrete Projection Kalman Filter solvers.
    Functions are designed for an ensemble of inputs."""

    def __init__(self, num_y, shape, sde, observation_map, H, num_steps=1000):
        """Construct an Esemble Kalman Filter Solver.
        Args:
            shape: Shape of array, x. (num_samples,) + x_shape, where x_shape is the shape
                of the object being sampled from, for example, an image may have
                x_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
            sde: A valid SDE class.
            num_steps: number of discretization time steps.
        """
        super().__init__(num_steps)
        self.sde = sde
        self.shape = shape
        self.observation_map = observation_map
        self.estimate_x_0 = self.get_estimate_x_0(self.sde, shape[1:])
        self.num_y = num_y
        self.H = H

    def get_estimate_x_0(self, sde, shape):
        """Get an MMSE estimate of x_0
        """
        if type(sde).__name__=='RVE':
            def estimate_x_0(x, t):
                v_t = sde.variance(t)
                x = x.reshape(shape)
                x = jnp.expand_dims(x, axis=0)
                t = jnp.expand_dims(t, axis=0)
                s = sde.score(x, t)
                s = s.flatten()
                x = x.flatten()
                return x + v_t * s, s
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
                return (x + v_t * s) / m_t, s
        else:
            raise ValueError("Did not recognise reverse SDE (got {}, expected VE or VP)".format(type(sde).__name__))
        return estimate_x_0

    def batch_observation_map(self, x, t):
        return vmap(lambda x: self.observation_map(x, t))(x)

    def batch_analysis(self, x, t, y, noise_std, v):
        return vmap(lambda x, t: self.analysis(x, t, y, noise_std, v), in_axes=(0, 0))(x, t)

    def predict(self, rng, x, t, noise_std):
        x = x.reshape(self.shape)
        drift, diffusion = self.sde.sde(x, t)
        # TODO: possible reshaping needs to occur here, if score
        # applies to an image vector
        alpha = self.sde.mean_coeff(t)[0]**2
        R = alpha * noise_std**2
        eta = jnp.sqrt(R)
        f = drift * self.dt
        G = diffusion * jnp.sqrt(self.dt)
        noise = random.normal(rng, x.shape)
        x_hat_mean = x + f
        x_hat = x_hat_mean + batch_mul(G, noise)
        x_hat_mean = x_hat_mean.reshape(self.shape[0], -1)
        x_hat = x_hat.reshape(self.shape[0], -1)
        return x_hat, x_hat_mean

    def update(self, rng, x, t, y, noise_std):
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
        ratio = v / sqrt_alpha
        x_hat, x_hat_mean = self.predict(rng, x, t, noise_std)
        m = self.batch_analysis(x_hat, t, y, noise_std, ratio)  # denoise x and perform kalman update
        # Missing a step here where x_0 is sampled as N(m, C), which makes Gaussian case exact probably
        x = sqrt_alpha * m + jnp.sqrt(v) * random.normal(rng, x.shape)  # renoise denoised x
        # x = sqrt_alpha * m - jnp.sqrt(v) * score # renoise denoised x
        # x = sqrt_alpha * m - v * score # renoise denoised x
        return x, m
        # Summary:
        # x = x_hat + v * score + sqrt_alpha * C_xh_hat @ jnp.linalg.solve(C_yy_hat, y - o_hat) + sqrt_v * random.normal(rng, x.shape)

    def analysis(self, x_hat, t, y, noise_std, ratio):
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t)
        m_hat, vjp_estimate_x_0, score = vjp(
            _estimate_x_0, x_hat, has_aux=True)
        o_hat = self.H @ m_hat
        batch_vjp_estimate_x_0 = vmap(lambda x: vjp_estimate_x_0(x)[0])
        C_yy = ratio * batch_vjp_estimate_x_0(self.H) @ self.H.T + noise_std**2 * jnp.eye(self.num_y)
        ls = ratio * vjp_estimate_x_0(self.H.T @ jnp.linalg.solve(C_yy, y - o_hat))[0]
        return m_hat + ls # , score


class DPKF(PKF):
    """Instead of using Tweedie second moment, use an approximation thereof."""

    def __init__(self, num_y, shape, sde, observation_map, H, data_variance=1.0, num_steps=1000):
        """Construct an Esemble Kalman Filter Solver.
        Args:
            shape: Shape of array, x. (num_samples,) + x_shape, where x_shape is the shape
                of the object being sampled from, for example, an image may have
                x_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
            sde: A valid SDE class.
            num_steps: number of discretization time steps.
        """
        self.data_variance = data_variance
        super().__init__(num_y, shape, sde, observation_map, H, num_steps)

    def analysis(self, x_hat, t, y, noise_std, v):
        r"""Return the drift and diffusion coefficients of the SDE.

        Args:

        Returns:
            x: A JAX array of the next state.
            x_mean: A JAX array of the next state without noise, for denoising.
        """
        sqrt_v = jnp.sqrt(v)
        sqrt_alpha = self.sde.mean_coeff(t)
        ratio = v / sqrt_alpha
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t)
        m_hat, vjp_estimate_x_0, score = vjp(
            _estimate_x_0, x_hat, has_aux=True)
        o_hat = self.H @ m_hat
        data_variance = 130.0
        # model_variance = 1. / (sqrt_alpha**2 / v + 1. / data_variance)
        model_variance = v * self.data_variance / (sqrt_alpha**2 * self.data_variance + v)
        C_yy = model_variance * self.H @ self.H.T + noise_std**2 * jnp.eye(self.num_y)
        ls = ratio * vjp_estimate_x_0(self.H.T @ jnp.linalg.solve(C_yy, y - o_hat))[0]
        return m_hat + ratio * ls


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

class Euler():
    """Euler numerical solve of an ODE. Functions are designed for a mini-batch of inputs."""

    def __init__(self, ode):
        """
        Constructs an Euler sampler.
        Args:
            ode: A valid ODE class.
        """
        self.ode = ode

    def get_update(self):
        discretize = self.ode.discretize

        def update(x, t):
            """
            Args:
                rng: A JAX random state.
                x: A JAX array representing the current state.
                t: A JAX array representing the current step.

            Returns:
                x: A JAX array of the next state:
                x_mean: A JAX array. The next state without random noise. Useful for denoising.
            """
            f = discretize(x, t)
            x = x + f
            return x
        return update


class EulerMaruyama():
    """
    Variant on the solver from diffusionjax in that it also returns the noise
    Euler Maruyama numerical solver of an SDE. Functions are designed for a mini-batch of inputs."""

    def __init__(self, sde):
        """Constructs an Euler Maruyama sampler.
        Args:
            sde: A valid SDE class.
            score_fn: A valid score function.
        """
        self.sde = sde

    def get_update(self):
        discretize = self.sde.discretize

        def update(rng, x, w, t):
            """
            Args:
                rng: A JAX random state.
                x: A JAX array representing the current state.
                t: A JAX array representing the current step.

            Returns:
                x: A JAX array of the next state:
                x_mean: A JAX array. The next state without random noise. Useful for denoising.
            """
            f, G = discretize(x, t)
            z = random.normal(rng, x.shape)
            w = w + z * jnp.sqrt(self.sde.dt)
            x_mean = x + f
            x = x_mean + batch_mul(G, z)
            return x, x_mean, w
        return update


class VelocityVerlet():
    """Velocity verlet solve of an ODE. Functions are designed for a mini-batch of inputs."""

    def __init__(self, ode):
        """
        Constructs a Velocity Verlet sampler.
        Args:
            ode: A valid ODE class.
        """
        self.ode = ode

    def get_update(self):
        discretize = self.ode.discretize

        def update(rng, x, xd, xdd, t):
            """
            Args:
                rng: A JAX random state.
                x: A JAX array representing the current state.
                t: A JAX array representing the current step.

            Returns:
                x: A JAX array of the next state:
                x_mean: A JAX array. The next state without random noise. Useful for denoising.
            """
            f, G = discretize(x, t)
            xd1 = xd + (self.ode.dt / 2) * xdd # Half-step velocity
            xdd1 = f - self.ode.damping * xd1  # / densities, assume density is 1.0
            xd = xd1 + (self.ode.dt / 2) * xdd1  # Full-step velocity
            xdd = xdd1
            x = x + self.ode.dt * (xd + (self.ode.dt / 2) * xdd1)
            z = random.normal(rng, x.shape)
            x = x + batch_mul(G, z)
            # x = x.at[bc_types == 0].set(bc_values)  # assume no boundary conditions are applied
            return x, xd, xdd

        return update


class EulerMaruyamaUnderdamped():
    """Euler Maruyama discretization of underdamped SDE. Functions are designed for a mini-batch of inputs."""

    def __init__(self, sde):
        """
        Constructs an Euler Cromer sampler.
        Args:
            sde: A valid SDE class.
        """
        self.sde = sde

    def get_update(self):
        discretize = self.sde.discretize
        sde = self.sde.sde

        def update(rng, x, xd, t):
            """
            Args:
                rng: A JAX random state.
                x: A JAX array representing the current state.
                t: A JAX array representing the current step.

            Returns:
                x: A JAX array of the next state:
                x_mean: A JAX array. The next state without random noise. Useful for denoising.
            """
            drift, diffusion = sde(x, t)
            G = diffusion * jnp.sqrt(self.sde.dt)
            xdd = drift - self.sde.damping * xd  # / densities, assume density is 1.0
            F = xdd * self.sde.dt
            z = random.normal(rng, x.shape)
            xd = xd + F + batch_mul(G, z)
            x = x + self.sde.dt * xd
            # x = x.at[bc_types == 0].set(bc_values)  # assume no boundary conditions are applied
            return x, xd

        return update


class MatrixEulerMaruyama():
    """Euler Maruyama numerical solver of an SDE. Functions are designed for a mini-batch of inputs."""

    def __init__(self, sde):
        """Constructs an Euler Maruyama sampler.
        Args:
            sde: A valid SDE class.
        """
        self.sde = sde
        self.ts = sde.ts

    def get_update(self):
        discretize = self.sde.discretize

        def update(rng, x, t):
            """
            Args:
                rng: A JAX random state.
                x: A JAX array representing the current state.
                t: A JAX array representing the current step.

            Returns:
                x: A JAX array of the next state:
                x_mean: A JAX array. The next state without random noise. Useful for denoising.
            """
            f, G = discretize(x, t)
            z = random.normal(rng, x.shape)
            x_mean = x + f
            x = x_mean + jnp.einsum('ij, kjm -> kim', G, z)
            return x, x_mean
        return update


class ChengUnderdamped():
    """
    This algorithm computes the exact moments conditioned on the current state and samples from them exactly."""


    def __init__(self, sde):
        """
        Constructs an Euler Cromer sampler.
        Args:
            sde: A valid SDE class.
        """
        self.sde = sde

    def get_update(self):
        discretize = self.sde.discretize
        sde = self.sde.sde

        def update(rng, x, xd, t):
            """
            gamma is a damping parameter
            u is a mass parameter
            L is a Lipschitz constant, which is proportional to the maximum eigenvalue of Q,
            and acts as an inverse mass parameter.
            step size delta < 1
            number of iterations n
            initial point x0, 0
            Args:
                rng: A JAX random state.
                x: A JAX array representing the current state.
                t: A JAX array representing the current step.

            Returns:
                x: A JAX array of the next state:
                x_mean: A JAX array. The next state without random noise. Useful for denoising.
            """
            # Better can be done than EM, when using an exact integrator
            # Gibbs steps
            drift, diffusion = sde(x, t)

            const2 = - 1. / 2 * (1 - jnp.exp(-2. * self.sde.dt))  # note negative
            const1 = self.sde.dt - (1. / 4) * jnp.exp(- 4. * self.sde.dt) - 3. / 4 + jnp.exp(-2. * self.sde.dt)
            const3 = 1 - jnp.exp(-4. * self.sde.dt)
            const0 = 1. / 2 * (1 + jnp.exp(-4. * self.sde.dt) - 2. * jnp.exp(-2. * self.sde.dt))

            var_xd = 1. / self.sde.L * const3 - 1. / self.sde.L * const0**2 / const1
            var_x = 1. / self.sde.L * const1 - 1. / self.sde.L * const0**2 / const3

            x_mean = x - const2 * xd - 1. / (2. * self.sde.L) * (self.sde.dt + const2) * drift
            xd_mean = xd * jnp.exp(-2. * self.sde.dt) + 1. / self.sde.L * const2 * drift

            z_xd = random.normal(rng, x.shape)
            xd = xd_mean + const0 / const1 * (x - x_mean) + jnp.sqrt(var_xd) * z_xd

            x_mean = x - const2 * xd - 1. / (2. * self.sde.L) * (self.sde.dt + const2) * drift
            rng, step_rng = random.split(rng)
            z_x = random.normal(rng, x.shape)
            x = x_mean + const0 / const3 * (xd - xd_mean) + jnp.sqrt(var_x) * z_x
            return x, xd

        return update
