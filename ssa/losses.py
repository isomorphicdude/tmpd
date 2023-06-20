"""All functions related to loss computation and optimization."""
import flax
import jax
import jax.numpy as jnp
import jax.random as random
from ssa.models import utils as mutils
from diffusionjax.sde import VE, VP
from diffusionjax.utils import batch_mul, get_score
from diffusionjax.losses import get_loss, errors
import optax


def get_optimizer(config):
    """Returns an optax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        if config.optim.weight_decay:
            optimizer = optax.adamw(
                config.optim.lr, b1=config.optim.beta1, eps=config.optim.eps)
        else:
            optimizer = optax.adam(
                config.optim.lr, b1=config.optim.beta1, eps=config.optim.eps)
    else:
        raise NotImplementedError(
            'Optimiser {} not supported yet!'.format(config.optim.optimizer)
        )
    return optimizer


def SSget_optimizer(config):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = flax.optim.Adam(beta1=config.optim.beta1, eps=config.optim.eps,
                                weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(state,
                  grad,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    assert 0
    lr = state.lr
    if warmup > 0:
      lr = lr * jnp.minimum(state.step / warmup, 1.0)
    if grad_clip >= 0:
      # Compute global gradient norm
      grad_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
      # Clip gradient
      clipped_grad = jax.tree_map(
        lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
    else:  # disabling gradient clipping if grad_clip < 0
      clipped_grad = grad
    return state.optimizer.apply_gradient(clipped_grad, learning_rate=lr)

  return optimize_fn


def get_score(sde, model, params, states, score_scaling=True, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: A `flax.linen.Module` object the represent the architecture of score-based model.
    params: A dictionary that contains all trainable parameters.
    states: A dictionary that contains all mutable states.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def score(x, labels, rng=None):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
      rng: If present, it is the random state for dropout

    Returns:
      A tuple of (model output, new mutable states)
    """
    # states is a dictionary that contains all mutable states: it contains the keys state, target
    # params is a dictionary?
    variables = {'params': params, **states}
    if not train:
      # for i in params: print(i)
      # for i in params['ResnetBlockBigGANpp_0']['GroupNorm_0']: print(i)
      print(x.shape, "input_shape")
      print(labels.shape, "label_shape")
      print(params['ResnetBlockBigGANpp_0']['GroupNorm_0']['scale'].shape, "scale shape of params")
      print(params['ResnetBlockBigGANpp_0']['GroupNorm_0']['bias'].shape, "bias shape of params")

      model_eval = model.apply(variables, x, labels, train=False, mutable=False)
      # model_eval = model.apply(variables, x, labels, train=False, mutable=False), states
    else:
      rngs = {'dropout': rng}
      model_eval = model.apply(variables, x, labels, train=True, mutable=list(states.keys()), rngs=rngs)

    if score_scaling is True:
      return lambda x, t: -batch_mul(model_eval, 1. / sde.marginal_prob(x, t)[1])
    else:
      return lambda x, t: -model_eval
      # if states:
      #   return outputs
      # else:
      #   return outputs, states

  return score


def get_loss(sde, solver, model, score_scaling=True, likelihood_weighting=True, reduce_mean=True, pointwise_t=False):
    """
    Forked from diffusionjax to get the loss. NOTE: only tested for evaluation.
    TODO: Parameter shapes seem a little different, so maybe need to go as far as possible replicating
    get_sde_loss_fn

    Create a loss function for score matching training.
    Args:
        sde: Instantiation of a valid SDE class.
        solver: Instantiation of a valid Solver class.
        model: A valid flax neural network `:class:flax.linen.Module` class.
        score_scaling: Boolean variable, set to `True` if learning a score scaled by the marginal standard deviation.
        likelihood_weighting: Boolean variable, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
        reduce_mean: Boolean variable, set to `True` if taking the mean of the errors in the loss, set to `False` if taking the sum.
        pointwise_t: Boolean variable, set to `True` if returning a function that can evaluate the loss pointwise over time. Set to `False` if returns an expectation of the loss over time.

    Returns:
        A loss function that can be used for score matching training.
    """
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    if pointwise_t:
        def loss(t, params, states, rng, data):
            n_batch = data.shape[0]
            ts = jnp.ones((n_batch,)) * t
            score = get_score(sde, model, params, states, score_scaling, train=False)
            e = errors(ts, sde, score, rng, data, likelihood_weighting)
            losses = e**2
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
            if likelihood_weighting:
                g2 = sde.sde(jnp.zeros_like(data), ts)[1]**2
                losses = losses * g2
            return jnp.mean(losses)
    else:
        def loss(params, states, rng, data):
            rng, step_rng = random.split(rng)
            n_batch = data.shape[0]
            # which one is preferable?
            # ts = random.randint(step_rng, (n_batch,), 1, solver.num_steps) / (solver.num_steps - 1)
            ts = random.uniform(step_rng, (data.shape[0],), minval=1. / solver.num_steps, maxval=1.0)
            score = get_score(sde, model, params, states, score_scaling, train=False)
            e = errors(ts, sde, score, rng, data, likelihood_weighting)
            losses = e**2
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
            if likelihood_weighting:
                g2 = sde.sde(jnp.zeros_like(data), ts)[1]**2
                losses = losses * g2
            return jnp.mean(losses)
    return loss


# TODO: delete this probably not needed. What is the actual problem? check the shapes going in/out
def get_loss_tmp(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    """Compute the loss function.

    Args:
      rng: A JAX random state.
      params: A dictionary that contains trainable parameters of the score-based model.
      states: A dictionary that contains mutable states of the score-based model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
      new_model_state: A dictionary that contains the mutated states of the score-based model.
    """

    score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True)
    data = batch['image']

    rng, step_rng = random.split(rng)
    t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
    rng, step_rng = random.split(rng)
    z = random.normal(step_rng, data.shape)
    mean, std = sde.marginal_prob(data, t)
    perturbed_data = mean + batch_mul(std, z)
    rng, step_rng = random.split(rng)
    score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

    if not likelihood_weighting:
      losses = jnp.square(batch_mul(score, std) + z)
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
    else:
      g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
      losses = jnp.square(score + batch_mul(z, 1. / std))
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

    loss = jnp.mean(losses)
    return loss, new_model_state

  return loss_fn

def get_sde_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

  def loss_fn(rng, params, states, batch):
    """Compute the loss function.

    Args:
      rng: A JAX random state.
      params: A dictionary that contains trainable parameters of the score-based model.
      states: A dictionary that contains mutable states of the score-based model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
      new_model_state: A dictionary that contains the mutated states of the score-based model.
    """

    score_fn = mutils.get_score_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True)
    data = batch['image']

    rng, step_rng = random.split(rng)
    t = random.uniform(step_rng, (data.shape[0],), minval=eps, maxval=sde.T)
    rng, step_rng = random.split(rng)
    z = random.normal(step_rng, data.shape)
    mean, std = sde.marginal_prob(data, t)
    perturbed_data = mean + batch_mul(std, z)
    rng, step_rng = random.split(rng)
    score, new_model_state = score_fn(perturbed_data, t, rng=step_rng)

    if not likelihood_weighting:
      losses = jnp.square(batch_mul(score, std) + z)
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
    else:
      g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
      losses = jnp.square(score + batch_mul(z, 1. / std))
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

    loss = jnp.mean(losses)
    return loss, new_model_state

  return loss_fn


def get_step_fn(sde, solver, model, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training and `False` for evaluation.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    # TODO: Maybe won't be valid since get_model_fn uses slightly different
    # loss_fn = get_loss(sde, solver, model, likelihood_weighting=likelihood_weighting, reduce_mean=reduce_mean)  # stateless
    # TODO: this doesn't work because of some parameter shape that is different.
    loss_fn = get_loss(sde, solver, model, likelihood_weighting=likelihood_weighting, reduce_mean=reduce_mean)
    # TODO: but this doesn't involve the solver and skips that step so maybe a hybrid solution is needed
    # loss_fn = get_sde_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5)
    # loss_fn = get_loss_tmp(sde, solver, model, likelihood_weighting=likelihood_weighting, reduce_mean=reduce_mean)

  else:
    raise NotImplementedError(f"Discrete training for {sde.__class__.__name__} is not supported.")
    # assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    # if isinstance(sde, VE):
    #   loss_fn = get_smld_loss_fn(sde, model, train, reduce_mean=reduce_mean)
    # elif isinstance(sde, VP):
    #   loss_fn = get_ddpm_loss_fn(sde, model, train, reduce_mean=reduce_mean)
    # else:
    #   raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(carry_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
      batch: A mini-batch of training/evaluation data.

    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """

    (rng, state) = carry_state
    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    if train:
      raise NotImplementedError()
      # params = state.optimizer.target
      # states = state.model_state
      # (loss, new_model_state), grad = grad_fn(step_rng, params, states, batch)
      # grad = jax.lax.pmean(grad, axis_name='batch')
      # new_optimizer = optimize_fn(state, grad)
      # new_params_ema = jax.tree_multimap(
      #   lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
      #   state.params_ema, new_optimizer.target
      # )
      # step = state.step + 1
      # new_state = state.replace(
      #   step=step,
      #   optimizer=new_optimizer,
      #   model_state=new_model_state,
      #   params_ema=new_params_ema
      # )
    else:
      # loss = loss_fn(state.params_ema, rng, batch)
      loss = loss_fn(state.params_ema, state.model_state, step_rng, batch)
      print(loss)
      assert 0
      loss = loss_fn(step_rng, state.params_ema, state.model_state, batch)
      new_state = state

    loss = jax.lax.pmean(loss, axis_name='batch')
    new_carry_state = (rng, new_state)
    return new_carry_state, loss

  return step_fn
