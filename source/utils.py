"""Utility functions, including all functions related to
loss computation, optimization, sampling and inverse problems.
"""
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap, vjp, value_and_grad, jit
from diffusionjax.utils import get_score, batch_mul, errors
from functools import partial
import numpy as np
# import optax


def trunc_svd(X, rank, hermitian=False):
    (n_row, n_col) = jnp.shape(X)
    U, S, Vt = jnp.linalg.svd(X, hermitian=hermitian)
    return U[:, :rank], S[:rank], Vt[:rank, :]


def batch_matmul(A, B):
    return vmap(lambda A, B: A @ B)(A, B)


def batch_trace(A):
    return vmap(jnp.trace)(A)


def get_log_likelihood_full(H, mask=None):
    if mask is not None:
        # Just get grad of this wrt x, can substitute x_0 for any estimator
        def log_likelihood(vjp_estimate_x_0, x_0, t):
            """y is given as a full length vector."""
            innovation = (y - x_0) * mask
            Cyy = ratio(t) + noise_std**2
            return -0.5 * jnp.dot(innovation, innovation) / Cyy
    else:
        def log_likelihood(vjp_estimate_x_0, x_0, t):
            """y is given as just observations, H a linear operator."""
            vec_vjp_fn = vmap(vjp_estimate_x_0)
            H_grad_x_hat = vec_vjp_fn(H)[0]
            innovation = y - H @ x_0
            Cyy = ratio(t) * H @ H_grad_x_hat.T + noise_std**2 * jnp.eye(y.shape[0])
            f = jnp.linalg.solve(Cyy, innovation)
            return -0.5 * jnp.dot(innovation, f)
    return log_likelihood


def get_log_likelihood_diag(H, mask=None):
        if mask is not None:
            # posterior_score_approx0 using J Song approximation and posterior_score_approx1 using Chung approximation just use a different x_0
            # Just get grad of this wrt x, can substitute x_0 for any estimator
            def log_likelihood(x_0, t):
                """y is given as a full length vector."""
                innovation = (y - x_0) * mask
                Cyy = ratio(t) + noise_std**2
                return -0.5 * jnp.dot(innovation, innovation) / Cyy
        else:
            def log_likelihood(x_0, t):
                """y is given as just observations, H a linear operator."""
                _vjp_estimate_x_0 = lambda h: vjp_estimate_x_0(h)[0]
                vmap_vjp_estimate_x_0 = vmap(_vjp_estimate_x_0)
                H_Sigma_d_r_t = vmap_vjp_estimate_x_0(H)
                Sigma_d_r_t_diag = jnp.diag(H_Sigma_d_r_t.T @ H)
                Sigma_diag = Sigma_d_r_t_diag * ratio(t)
                Cyy = Sigma_diag + noise_std**2

                vec_vjp_fn = vmap(vjp_estimate_x_0)
                H_grad_x_hat = vec_vjp_fn(H)[0]
                innovation = y - H @ x_0
                Cyy = ratio(t) * jnp.diag(H @ H_grad_x_hat.T) + noise_std**2 * jnp.eye(y.shape[0])
                f = innovation / Cyy
                return -0.5 * jnp.dot(innovation, f)


def get_log_likelihood_scalar(H, mask=None):
    if mask is not None:
        # posterior_score_approx0 using J Song approximation and posterior_score_approx1 using Chung approximation just use a different x_0
        # Just get grad of this wrt x, can substitute x_0 for any estimator
        def log_likelihood(x_0, t):
            """y is given as a full length vector."""
            innovation = (y - x_0) * mask
            Cyy = ratio(t) + noise_std**2
            return -0.5 * jnp.dot(innovation, innovation) / Cyy
    else:
        def log_likelihood(x_0, t):
            """y is given as just observations, H a linear operator."""
            innovation = y - H @ x_0
            Cyy = ratio(t) * H @ H.T + noise_std**2 * jnp.eye(y.shape[0])
            f = jnp.linalg.solve(Cyy, innovation)
            return -0.5 * jnp.dot(innovation, f)


def get_matrix(batch_C_t, sde, model, params, score_scaling):
    if score_scaling is True:
        def normalized(t, in_size):
            As = model.evaluate(params, t, in_size)
            # Cs = batch_C_t(t)
            # prods = batch_matmul(As, Cs)
            # As = batch_mul(As, jnp.sqrt(in_size) / jnp.linalg.norm(prods, axis=(1, 2)))
            # As = batch_mul(As, in_size / batch_trace(prods))
            return batch_mul(As, 1. / jnp.sqrt(sde.variance(t)))
        return normalized
        # return lambda t, in_size: batch_mul(As, 1. / jnp.sqrt(sde.variance(t)))
        # return lambda t, in_size: batch_mul(model.evaluate(params, t, in_size), 1. / jnp.sqrt(sde.variance(t)))
    else:
        def normalized(t, in_size):
            As = model.evaluate(params, t, in_size)
            # Cs = batch_C_t(t)
            # prods = batch_matmul(As, Cs)
            # As = batch_mul(As, jnp.sqrt(in_size) / jnp.linalg.norm(prods, axis=(1, 2)))
            # As = batch_mul(As, in_size / batch_trace(prods))
            return As
        return normalized
        # return lambda t, in_size: model.evaluate(params, t, in_size)


# deterministic losses
def retrain_nn(
        update_step, num_epochs, step_rng, score_model, params,
        opt_state, loss):
    """TODO: This must be for deterministc loss, without data"""
    losses = jnp.zeros((num_epochs, 1))
    for i in range(num_epochs):
        rng, step_rng = random.split(step_rng)
        loss_eval, params, opt_state = update_step(params, step_rng, opt_state, score_model, loss)
        losses = losses.at[i].set(loss_eval)
        if i % 10 == 0:
            print("Epoch {:d}, Loss {:.2f} ".format(i, losses[i, 0]))
    return score_model, params, opt_state, losses


@partial(jit, static_argnums=[3, 4, 5])
def update_step(params, rng, opt_state, model, loss, has_aux=False):
    """
    # TODO: This must be for the deterministic loss, without data
    Takes the gradient of the loss function and updates the model weights (params) using it.
    Args:
        params: the current weights of the model
        rng: random number generator from jax
        opt_state: the internal state of the optimizer
        model: the score function
        loss: A loss function that can be used for score matching training.
        has_aux:
    Returns:
        The value of the loss function (for metrics), the new params and the new optimizer states function (for metrics), the new params and the new optimizer state.
    """
    val, grads = value_and_grad(loss, has_aux=has_aux)(params, model, rng)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, opt_state


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


def trace_loss(A, B):
    return jnp.trace(A.T @ A @ B) + 2. * jnp.trace(A) + jnp.trace(A @ B - jnp.eye(A.shape[0]))**2
#     return jnp.einsum('ji, ij ->', A @ B, A) + 2. * jnp.trace(A)


def batch_trace_loss(A, B):
    return vmap(lambda A, B: trace_loss(A, B))(A, B)


def get_operator_loss0(sde, model, shape, n_batch, C, score_scaling=True, likelihood_weighting=True, reduce_mean=True, pointwise_t=False):
    """Create a loss function for score matching training via operator.
    Use no data but parameterize a neural net to output matrix values
    that can then be applied
    could parameterize it as lower triangular, and take the transpose
    sampled at random, but restrict the network to be linear.
    Which could be computationally efficient?
    Args:
        sde: instantiation of a valid SDE class.
        model: a valid flax neural network `:class:flax.linen.Module` class
        score_scaling: Boolean variable, set to `True` if learning a score scaled by the marginal standard deviation.
        likelihood_weighting: Boolean variable, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
        reduce_mean: Boolean variable, set to `True` if taking the mean of the errors in the loss, set to `False` if taking the sum.
        pointwise_t: Boolean variable, set to `True` if returning a function that can evaluate the loss pointwise over time. Set to `False` if returns an expectation of the loss over time.
        C: A JAX array of the covariance matrix being trained.

    Returns:
        A loss function that can be used for score matching training.
    """
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    in_size = np.prod(shape)

    # I wonder if I need to do a time reversal at any point?

    def C_t(t):
        return sde.mean_coeff(t)**2 * C + sde.variance(t) * jnp.eye(shape[0])

    def batch_C_t(t):
        return vmap(lambda t: C_t(t))

    if pointwise_t:
        def loss(params, model, rng):
            rng, step_rng = random.split(rng)
            matrix_t = get_matrix(batch_C_t, sde, model, params, score_scaling)
            matrix_t_eval = matrix_t(t, in_size)
            C_t_eval = C_t(t)
            loss = trace_loss(matrix_t_eval, C_t_eval)
            if likelihood_weighting:
                g = sde.sde(jnp.zeros((1,) + shape), t)[1]
                print(g)
                assert 0
                loss = loss * g
            return loss

    else:
        batch_C_t = vmap(C_t)
        def loss(params, model, rng):
            rng, step_rng = random.split(rng)
            ts = random.randint(step_rng, (n_batch,), 1, sde.n_steps) / (sde.n_steps - 1)
            # matrix_t should be over a batch of t inputs
            matrix_t = get_matrix(batch_C_t, sde, model, params, score_scaling)
            matrix_t_eval = matrix_t(ts, in_size)
            C_t_eval = batch_C_t(ts)
            losses = batch_trace_loss(matrix_t_eval, C_t_eval)
            if likelihood_weighting:
                g = sde.sde(jnp.zeros((n_batch,) + shape), ts)[1]
                losses = losses * g
            return reduce_op(losses)
        return loss


def get_operator_loss1(sde, model, shape, n_batch, C, score_scaling=True, likelihood_weighting=True, reduce_mean=True, pointwise_t=False):
    """Create a loss function for score matching training via operator.
    Use data sampled at random, but restrict the network to be linear.
    Which could be computationally efficient?
    Args:
        sde: instantiation of a valid SDE class.
        model: a valid flax neural network `:class:flax.linen.Module` class
        score_scaling: Boolean variable, set to `True` if learning a score scaled by the marginal standard deviation.
        likelihood_weighting: Boolean variable, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
        reduce_mean: Boolean variable, set to `True` if taking the mean of the errors in the loss, set to `False` if taking the sum.
        pointwise_t: Boolean variable, set to `True` if returning a function that can evaluate the loss pointwise over time. Set to `False` if returns an expectation of the loss over time.
        C: A JAX array of the covariance matrix being trained.

    Returns:
        A loss function that can be used for score matching training.
    """

    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)

    def C_t(t):
        return sde.mean_coeff(t) ** 2 * C + sde.variance(t) * jnp.eye(shape[0])

    # No batching required, since no data
    if pointwise_t:
        def loss(params, model, rng):
            rng, step_rng = random.split(rng)
            score = get_score(sde, model, params, score_scaling)
            noise = random.normal(step_rng, np.prod(shape))
            score, vjp_estimate = vjp(lambda x: score(x, t), noise)
            C_t_eval = C_t(t)
            vjp_eval = vjp_estimate(noise)
            return vjp_eval.T @ C_t_eval @ vjp_eval + 2 * vjp_eval.T @ vjp_eval
    else:
        def loss(params, model, rng):
            # Need to train the score in regions of space where x is likely to be
            # or, if it doesn't matter where x is, in a way that is independent of x
            # This can be done by constraining network to be a linear one
            # (although, dependent nonlinearly on t)
            # and evaluating the jacobian of this network
            # perhaps it trains well, maybe try some things with UNet first
            # Could still evaluate it at x at samples from this distribution,
            # but if it is linear, the covariance should be independent of x
            # so can train score to be accurate anywhere?
            # Score, in linear case should be independent of x
            # but in practice, some score as a linear solve of x seems to work
            rng, step_rng = random.split(rng)
            score = get_score(sde, model, params, score_scaling)
            ts = random.randint(step_rng, (n_batch, 1), 1, sde.n_steps) / (sde.n_steps - 1)
            print(ts.shape)
            noise = random.normal(step_rng, (n_batch,) + (shape))
            print(noise.shape)
            score, vjp_estimate = vmap(vjp(lambda x: score(x, ts), noise))
            C_t_eval = batch_C_t(ts)
            vjp_eval = vjp_estimate(noise)
            print(C_t_eval.shape)
            print(vjp_eval.shape)
            assert 0
            return vjp_eval.T @ C_t_eval @ vjp_eval + 2 * vjp_eval.T @ vjp_eval
        return loss


