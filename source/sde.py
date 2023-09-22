"""SDE class."""
import abc
from functools import partial
import jax.numpy as jnp
import jax
import jax.random as random
from diffusionjax.sde import SDE, VP, VE
from diffusionjax.utils import batch_mul


class LangevinCorrector(SDE):
    """Underdamped Langevin SDE."""
    def __init__(self, score):
        super().__init__()
        self.score = score

    def sde(self, x, t):
        # drift = -self.score(x, t)

        if isinstance(self, VP):
            print(self)
            assert 0
            beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
            alpha = 1. - beta_t
        elif isinstance(self, VE):
            print(self)
            assert 0
            alpha = 1.

        grad = self.score(x, t)
        rng, step_rng = random.split(rng)
        noise = random.normal(step_rng, x.shape)
        grad_norm = jnp.linalg.norm(
            grad.reshape((grad.shape[0], -1)), axis=-1
        ).mean()
        grad_norm = jax.lax.pmean(grad_norm, axis_name='batch')
        noise_norm = jnp.linalg.norm(
            noise.reshape((noise.shape[0], -1)), axis=-1
        ).mean()
        noise_norm = jax.lax.pmean(noise_norm, axis_name='batch')
        step_size = (target_snr * noise_norm / grad_norm)**2 * 2 * alpha
        return grad, jnp.sqrt(step_size * 2)

        x_mean = x + batch_mul(step_size, grad)
        x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
        return rng, x, x_mean

        diffusion = jnp.ones(x.shape) * jnp.sqrt(2)
        return drift, diffusion


class GradientFlow(SDE):
    """Gradient flow ODE."""
    def __init__(self, grad_log_potential, forcing, dt=1e-4, n_steps=10000):
        super().__init__(n_steps)
        self.dt = dt
        t1 = self.dt * n_steps
        self.train_ts = jnp.linspace(0, t1, self.n_steps + 1)[:-1].reshape(-1, 1)
        self.grad_log_potential = jit(vmap(
            grad_log_potential, in_axes=(0, 0), out_axes=(0)))
        self.forcing = forcing

    def marginal_prob(self, x, t):
        return 0

    def sde(self, x, t):
        drift = -self.grad_log_potential(x, t)
        return drift

    def discretize(self, x, t):
        r"""Discretize the SDE in the form,

        .. math::
            x_{i+1} = x_{i} + f_i(x_i) + G_i z_i

        Useful for diffusion sampling and probability flow sampling.
        Defaults to Euler-Maryama discretization.

        Args:
            x: a JAX tensor of the state
            t: a JAX float of the time step

        Returns:
            f, G
        """
        drift = self.sde(x, t)
        f = (drift + self.forcing) * self.dt
        return f


class CorrelatedOU(SDE):
    # TODO this is from diffusionjax package and will be useful for infinite dimension stuff
    # grfjax should be like an infinite dimensional version of diffusionjax,
    # and with operators and stuff
    """Time rescaled Correlated Ohrnstein Uhlenbeck (OU) SDE."""
    def __init__(self, C, L, beta_min=0.001, beta_max=3, n_steps=1000):
        super().__init__(n_steps)
        self.ts = jnp.linspace(0, 1, self.n_steps + 1)[:-1].reshape(-1, 1)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.L = L
        self.C = C
        self.dt = 1. / n_steps
        # self.L = jnp.expand_dims(L, axis=2)
        # self.C = jnp.expand_dims(C, axis=2)

    def sde(self, x, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * batch_mul(beta_t, x)
        diffusion = jnp.sqrt(beta_t[0]) * self.L
        diffusion2 = beta_t[0] * self.C
        # TODO
        # 1. why do need to batch multiply over beta_t
        # 2. how to generalize other functions so that can deal with this matrix diffusion
        # problem is that need to evaluate drift with a vec_t since score(x, vec_t) is used
        # then the other use case is when having a scalar or a matrix
        # so maybe could generalize batch_mul, or would it be more efficient to rewrite the whole
        # thing, or can the whole thing be done without much change to the code
        # Sampling must be rewritten, but score matching objective may be general?
        # Since scaling with CM norm is equivalent to score_scaling
        # The easiest option is to make a whole new package, the setup is too different
        # cost of steps is now in O(N^2) for example rather than O(N)
        # diffusion = batch_mul(jnp.sqrt(beta_t), self.L)
        # diffusion2 = batch_mul(beta_t, self.C)
        return drift, diffusion, diffusion2

    def log_mean_coeff(self, t):
        return -0.5 * t * self.beta_min - 0.25 * t**2 * (self.beta_max - self.beta_min)

    def mean_coeff(self, t):
        return jnp.exp(self.log_mean_coeff(t))

    def variance(self, t):
        return 1.0 - jnp.exp(2 * self.log_mean_coeff(t))

    def marginal_prob(self, x, t):
        r"""Evaluate marginal probability for evaluating loss wrt Cameron-Martin norm.
        Parameters to determine the marginal distribution of the SDE,

        .. math::
            p_t(x)

        Args:
            x: a JAX tensor of the state
            t: JAX float of the time
        """
        m = self.mean_coeff(t)
        try:
            mean = batch_mul(m, x)
        except:
            mean = m * x
        std = batch_mul(jnp.sqrt(self.variance(t)), L)
        return mean, std

    def reverse(self, score):
        """Create the reverse-time SDE/ODE

        Args:
            score: A time-dependent score-based model that takes x and t and returns the score.
        """
        # TODO: Is there a better way of inheriting these variables from self.__class__?
        ts = self.ts
        sde = self.sde
        beta_min = self.beta_min
        beta_max = self.beta_max

        def discretize(x, t):
            r"""Discretize the SDE in the form,

            .. math::
                x_{i+1} = x_{i} + f_i(x_i) + G_i z_i

            Useful for diffusion sampling and probability flow sampling.
            Defaults to Euler-Maryama discretization.

            Args:
                x: a JAX tensor of the state
                t: a JAX float of the time step

            Returns:
                f, G
            """
            drift, diffusion, diffusion2 = self.sde(x, t)
            f = drift * self.dt
            G = diffusion * jnp.sqrt(self.dt)
            G2 = diffusion2 * self.dt
            return f, G, G2

        class RSDE(self.__class__):

            def __init__(self):
                self.ts = ts
                self.beta_min = beta_min
                self.beta_max = beta_max

            def sde(self, x, t):
                beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
                drift = -0.5 * batch_mul(beta_t, x)
                diffusion = batch_mul(jnp.sqrt(beta_t), self.L)
                diffusion2 = batch_mul(beta_t, self.C)
                score = score(x, t)
                drift = drift - diffusion2 * score
                return drift, diffusion, diffusion2

            def discretize(self, x, t):
                f, G, G2 = discretize(x, t)
                # rev_f = -f + jnp.einsum('ij, kjm -> kim', G2, score(x, t))
                rev_f = -f + batch_mul(G2, score(x, t))
                return rev_f, G

        return RSDE()
