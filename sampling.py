"""Various sampling methods."""
import functools
from diffusionjax.utils import get_sampler
from diffusionjax.solvers import EulerMaruyama, Annealed
from diffusionjax.sde import UDLangevin


_correctors = {}
_predictors = {}


def get_predictor(name):
  return _predictors[name]


def get_corrector(name):
  return _correctors[name]


def register_solver(solvers, method=None, *, name=None):
    """A decorator for registering solver method."""

    def _register(method):
        if name is None:
            local_name = method.__name__
        else:
            local_name = name
        if local_name in _predictors:
            raise ValueError('Already registered solver with name {}'.format(local_name))
        solvers[local_name] = method
        return method

    if method is None:
        return _register
    else:
        return _register(method)


def get_sampling_fn(config, sde, model, shape, inverse_scaler, eps):
    """Create a sampling function.
    Args:
        config: A `ml_collections.ConfigDict` object that contains all configuration information.
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
        shape: A sequence of integers representing the expected shape of a single sample.
        inverse_scaler: The inverse data normalizer function.
        eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """
    sampler_name = config.sampling.method
    if sampler_name.lower()=='ode':
        return NotImplementedError()
    elif sampler_name.lower()=='pc':
        # method that takes config name and returns a coresponding solver
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        # TODO Will need to add score scaling in the config?
        score = mutils.get_score_fn(sde, model, state.params_ema, score_scaling=config.score_scaling)

        inner_solver = predictor(sde, score, num_steps, epsilon)
        outer_solver = corrector(sde, score, num_steps, snr)

        return get_sampler(
            shape, outer_solver, inner_solver, denoise=config.sampling.noise_removal, stack_samples=False)


@register_solver(_predictors, name='reverse_diffusion')
def reverse_diffusion_predictor(sde, score, num_steps, epsilon):
    return EulerMaruyama(sde.reverse(score), num_steps=num_steps, epsilon=epsilon)


@register_solver(_correctors, name='langevin')
def langevin_corrector(sde, score, num_steps, snr):
    return EulerMaruyama(sde.corrector(UDLangevin, score), num_steps=num_steps, snr=snr)


@register_solver(_correctors, name='ald')
def annealed_langevin_dynamics(sde, score, num_steps, snr):
    return Annealed(sde.corrector(UDLangevin, score), num_steps=num_steps, snr=snr)

