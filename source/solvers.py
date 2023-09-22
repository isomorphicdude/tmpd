"""Solver classes."""
import jax.numpy as jnp
from jax.lax import scan, pmean
import jax.random as random
from diffusionjax.utils import batch_mul
from diffusionjax.solvers import Solver


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


class AncestralSampling():
  """The ancestral sampling solver. Currently only supports VE/VP SDEs."""

  def __init__(self, sde):
      self.sde = sde
      self.ts = sde.ts

  def vesde_update_fn(self, rng, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = jnp.where(timestep == 0, jnp.zeros(t.shape), sde.discrete_sigmas[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + batch_mul(score, sigma ** 2 - adjacent_sigma ** 2)
    std = jnp.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, noise)
    return x, x_mean

  def vpsde_update_fn(self, rng, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
    beta = sde.discrete_betas[timestep]
    score = self.score_fn(x, t)
    x_mean = batch_mul((x + batch_mul(beta, score)), 1. / jnp.sqrt(1. - beta))
    noise = random.normal(rng, x.shape)
    x = x_mean + batch_mul(jnp.sqrt(beta), noise)
    return x, x_mean

  # TODO: maybe don't need get and set functions
  def update_fn(self, rng, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(rng, x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(rng, x, t)


class SSAnnealed(Solver):
    """Annealed Langevin numerical solver of an SDE.
    Functions are designed for a mini-batch of inputs."""

    def __init__(self, sde, num_steps=2, snr=1e-2):
        """Constructs an Annealed Langevin Solver.
        Args:
            sde: A valid SDE class.
        """
        super().__init__(num_steps)
        self.sde = sde
        self.snr = snr

    def update(self, rng, x, t):
        """
        Args:
            rng: A JAX random state.
            x: A JAX array representing the current state.
            t: A JAX array representing the current step.

        Returns:
            x: A JAX array of the next state:
            x_mean: A JAX array. The next state without random noise. Useful for denoising.
        """
        grad, diffusion = self.sde.sde(x, t)
        grad_norm = jnp.linalg.norm(
            grad.reshape((grad.shape[0], -1)), axis=-1).mean()
        # TODO: implement parallel mean across batches
        grad_norm = pmean(grad_norm, axis_name='batch')
        noise = random.normal(rng, x.shape)
        noise_norm = jnp.linalg.norm(
            noise.reshape((noise.shape[0], -1)), axis=-1).mean()
        noise_norm = pmean(noise_norm, axis_name='batch')
        # TODO: alpha need not be a mini-batch
        alpha = jnp.exp(2 * self.sde.log_mean_coeff(t))
        dt = (self.snr * noise_norm / grad_norm)**2 * 2 * alpha
        x_mean = x + batch_mul(grad, dt)
        x = x_mean + batch_mul(batch_mul(diffusion, jnp.sqrt(dt)), noise)
        return x, x_mean
