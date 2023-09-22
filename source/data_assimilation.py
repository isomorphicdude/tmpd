"""Data Assimilation methods, such as 4DVAR.
"""
import abc
import jax
import jax.numpy as jnp
from jax.lax import scan, while_loop
import jax.random as random
from jax import vmap, jacfwd, vjp, jvp
import functools


def get_estimate_x_n(rng, outer_solver):
    def estimate_x_n(x, t):
        x = x.reshape(1, -1)
        t = t.reshape(1, -1)
        drift, diffusion = outer_solver.sde.sde(x, t)
        f = drift * outer_solver.dt
        G = diffusion * jnp.sqrt(outer_solver.dt)
        noise = random.normal(rng, x.shape)
        x_mean = x + f
        return x_mean.flatten(), G[0]
    return estimate_x_n


def get_WC4DVAR_representer_sampler(adjoint_observation_map, observation_map, samples, shape, outer_solver, y, noise_std, stack_samples=False):
    """Get an ensemble kalman sampler from (possibly interleaved) numerical solver(s).

    Args:
        samples: Optional initiation of the sample path of array, x. (num_steps,) + x_shape.
        shape: Shape of array, x. (num_steps,) + x_shape, where x_shape is the shape
            of the object being sampled from, for example, an image may have
            x_shape==(H, W, C), and so shape==(num_steps, H, W, C) where num_steps is the number of
            time steps of the solver.
        solver: A valid numerical solver class that is the numerical solver of the reverse-SDE.
        stack_samples: Boolean variable that if `True` return all iterations of the sample path or
            just returns the last sample path.
    Returns:
        A sampler.
    """

    def map_sampler(rng, samples=samples):
        if samples is None:
            samples = prior_sampler(rng)
        outer_ts = outer_solver.ts

        estimate_x_n = get_estimate_x_n(rng, outer_solver)
        d_x = shape[1]
        d_y = y.shape[0]
        shape_alt = (shape[0] -1, shape[1])

        def forwards_representer_step(carry, x):
            psi_n = carry
            x_n = x[0:d_x]
            t = x[-1]
            x_n_prev_estimate, vjp_estimate_x_n, _ = vjp(
                lambda x: estimate_x_n(x, t), x_n, has_aux=True)
            vjp_estimate_x_n_psi_n = vjp_estimate_x_n(psi_n)[0]
            psi_n = vjp_estimate_x_n_psi_n
            return (psi_n), (psi_n)

        def backwards_representer_step(carry, x):
            c_n = carry
            psi_n = x[0:d_x]
            t_n = x[-1]
            c_n_prev_estimate, G = estimate_x_n(c_n, t_n)
            c_n = c_n_prev_estimate + G**2 * psi_n
            return (c_n), (c_n)

        def representer_step(carry):
            b, rho, samples, j = carry

            # Need to calculate first guess solution for dx_0
            delta_x_N = -samples[-1]
            xs = jnp.hstack((jnp.zeros(shape_alt), samples[:-1], outer_solver.ts[:-1]))
            (_), delta_x_ = scan(backwards_step, (delta_x_N), xs, reverse=True)
            delta_x_0 = delta_x_[0]

            psi_0 = adjoint_observation_map(b, outer_ts[0])
            # xs = jnp.hstack((samples[1:], outer_ts[1:]))  # works, seems correct and leads to different solution
            xs = jnp.hstack((samples[:-1], outer_ts[:-1]))
            c_N, psi = scan(forwards_representer_step, (psi_0), xs)

            # xs = jnp.hstack((samples[1:], outer_ts[1:]))  # works, seems correct and leads to different solution
            xs = jnp.hstack((samples[:-1], outer_ts[:-1]))
            c_0, c = scan(backwards_representer_step, (c_N), xs, reverse=True)

            eta = y - observation_map(samples[0], outer_ts[0])

            rho = observation_map(c_0, outer_ts[0]) + noise_std**2 * b - observation_map(delta_x_0, outer_ts[0]) + eta
            C = 1. + noise_std**2  # H is (d_y * d_x), R is (d_x * 1) so C is (d_y, 1) not sure
            gamma = 1. / C  # approximate preconditioner
            b = b - gamma * rho
            j += 1
            return b, rho, samples, j

        def forwards_step(carry, x):
            delta_lambda_n, delta_x_0, eta = carry
            delta_lambda_n_prev = delta_lambda_n.copy()
            x_n = x[:d_x]
            t = x[-1]
            x_n_prev_estimate, vjp_estimate_x_n, _ = vjp(
                lambda x: estimate_x_n(x, t), x_n, has_aux=True)
            delta_lambda_n = vjp_estimate_x_n(delta_lambda_n)[0]
            return (delta_lambda_n, delta_x_0, eta), (delta_lambda_n_prev)

        def backwards_step(carry, x):
            delta_x_n = carry
            delta_lambda_n = x[:d_x]
            x_n = x[d_x: 2 * d_x]
            t = x[-1]
            x_n_prev_estimate, jvp_estimate_x_n_delta_x_n, G = jvp(
                lambda x: estimate_x_n(x_n, t), (x_n,), (delta_x_n,), has_aux=True)
            xi_n = x_n - x_n_prev_estimate
            delta_x_n = jvp_estimate_x_n_delta_x_n - xi_n + G**2 * delta_lambda_n
            return (delta_x_n), (delta_x_n)

        def representer_condition_function(carry):
            eps = 1e-2 # l2 norm distance
            b, rho, *_ = carry
            return jnp.linalg.norm(rho) > eps

        def outer_condition_function(carry):
            eps = 1e-3 # l2 norm distance
            _, _, delta_x, *_ = carry
            return jnp.linalg.norm(delta_x) > eps

        def outer_step(carry):
            samples, delta_lambda_prev, delta_x_prev, i, _ = carry
            delta_lambda = jnp.zeros(shape_alt)
            delta_x = jnp.zeros(shape)

            (b, rho, samples, j) = while_loop(representer_condition_function, representer_step, ((jnp.zeros(d_y,), jnp.full((d_y,), jnp.inf), samples, 0)))

            delta_x_N = samples[-1] + delta_lambda[-1]
            xs = jnp.hstack((delta_lambda, samples[:-1], outer_solver.ts[1:]))
            (_), delta_x_ = scan(backwards_step, (delta_x_N), xs, reverse=True)
            delta_x = delta_x.at[:-1].set(delta_x_)

            xs = jnp.hstack((samples[1:], outer_solver.ts[1:]))
            eta = y - observation_map(samples[0], outer_solver.ts[0])
            # Now scan across the xs, computing a state, start with delta lambda_0 = 0
            delta_lambda_0 = jnp.zeros((d_x,))
            (_, _, _), delta_lambda = scan(forwards_step, (delta_lambda_0, delta_x[0], eta), xs)

            samples = samples + delta_x
            i += 1

            return (samples, delta_lambda, delta_x, i, j)

        def outer_step_scan(carry, x):
            return outer_step(carry), ()

        delta_lambda = jnp.full(shape_alt, jnp.inf)
        delta_x = jnp.full(shape, jnp.inf)
        # for i in range(2000):
        #     (samples, delta_lambda, delta_x, i, j) = outer_step((samples, delta_lambda, delta_x, i, 0))
        #     print("j", j)
        #     print("delta_x norm", jnp.linalg.norm(delta_x))

        (samples, delta_lambda, delta_x, i, j), () = scan(outer_step_scan, (samples, delta_lambda, delta_x, 0, 0), jnp.arange(100))

        # (samples, delta_lambda, delta_x, i, j) = while_loop(outer_condition_function, outer_step, (samples, delta_lambda, delta_x, 0, 0))

        return samples, delta_lambda, delta_x

    # return jax.pmap(sampler, axis_name='batch')
    return map_sampler


def get_WC4DVAR_sampler(adjoint_observation_map, observation_map, samples, shape, outer_solver, y, noise_std, stack_samples=False):
    """Get an ensemble kalman sampler from (possibly interleaved) numerical solver(s).

    Args:
        samples: Optional initiation of the sample path of array, x. (num_steps,) + x_shape.
        shape: Shape of array, x. (num_steps,) + x_shape, where x_shape is the shape
            of the object being sampled from, for example, an image may have
            x_shape==(H, W, C), and so shape==(num_steps, H, W, C) where num_steps is the number of
            time steps of the solver.
        solver: A valid numerical solver class that is the numerical solver of the reverse-SDE.
        stack_samples: Boolean variable that if `True` return all iterations of the sample path or
            just returns the last sample path.
    Returns:
        A sampler.
    """

    def map_sampler(rng, samples=samples):
        if samples is None:
            samples = prior_sampler(rng)
        outer_ts = outer_solver.ts
        estimate_x_n = get_estimate_x_n(rng, outer_solver)
        d_x = shape[1]
        shape_alt = (shape[0] -1, shape[1])

        def forwards_step(carry, x):
            delta_lambda_n, delta_x_0, eta = carry
            delta_lambda_n_prev = delta_lambda_n.copy()
            x_n = x[:d_x]
            t = x[-1]

            x_n_prev_estimate, vjp_estimate_x_n, _ = vjp(
                lambda x: estimate_x_n(x, t), x_n, has_aux=True)
            delta_lambda_n = vjp_estimate_x_n(delta_lambda_n)[0]

            # # I think it must be this step that is not correct, since it is the step that means the iteration does not work
            # x_n_prev_estimate, jvp_estimate_x_n_delta_lambda_n, _ = jvp(
            #     lambda x: estimate_x_n(x_n, t), (x_n,), (delta_lambda_n,), has_aux=True)
            # delta_lambda_n = jvp_estimate_x_n_delta_lambda_n - adjoint_observation_map((observation_map(delta_x_0, t) - eta) / noise_std**2, t)
            return (delta_lambda_n, delta_x_0, eta), (delta_lambda_n_prev)

        def backwards_step(carry, x):
            delta_x_n = carry
            delta_lambda_n = x[:d_x]
            x_n = x[d_x: 2 * d_x]
            t = x[-1]
            x_n_prev_estimate, jvp_estimate_x_n_delta_x_n, G = jvp(
                lambda x: estimate_x_n(x_n, t), (x_n,), (delta_x_n,), has_aux=True)
            xi_n = x_n - x_n_prev_estimate
            delta_x_n = jvp_estimate_x_n_delta_x_n - xi_n + G**2 * delta_lambda_n
            return (delta_x_n), (delta_x_n)

        def joint_step(carry):
            delta_lambda, delta_lambda_prev, delta_x, delta_x_prev, samples, j = carry
            delta_lambda_prev = delta_lambda.copy()
            delta_x_prev = delta_x.copy()

            # delta_x_N = -samples[-1] + delta_lambda[-1]
            delta_x_N = -samples[-1]
            # is delta lambda correct, here, doesn't it need to be replaced
            xs = jnp.hstack((delta_lambda, samples[:-1], outer_solver.ts[:-1]))
            # xs = jnp.hstack((delta_lambda, samples[1:], outer_solver.ts[1:]))  # I think this is correct but it doesn't work
            (_), delta_x_ = scan(backwards_step, (delta_x_N), xs, reverse=True)
            delta_x = delta_x.at[-1].set(delta_x_N)
            delta_x = delta_x.at[:-1].set(delta_x_)

            xs = jnp.hstack((samples[:-1], outer_solver.ts[:-1]))
            eta = y - observation_map(samples[0], outer_solver.ts[0])
            # # Now scan across the xs, computing a state, start with delta lambda_0 = 0
            # delta_lambda_0 = jnp.zeros((d_x,))
            delta_lambda_0 = - adjoint_observation_map((observation_map(delta_x[0], outer_solver.ts[0]) - eta) / noise_std**2, outer_solver.ts[0])
            (_, _, _), delta_lambda = scan(forwards_step, (delta_lambda_0, delta_x[0], eta), xs)

            j += 1
            return (delta_lambda, delta_lambda_prev, delta_x, delta_x_prev, samples, j)

        def inner_condition_function(carry):
            eps = 1e-3  # l2 norm distance
            delta_lambda, delta_lambda_prev, delta_x, delta_x_prev, *_ = carry
            return jnp.linalg.norm(delta_x - delta_x_prev) > eps
            # return jnp.linalg.norm(delta_lambda - delta_lambda_prev) > eps

        def outer_condition_function(carry):
            eps = 1e-4  # l2 norm distance
            _, _, delta_x, *_ = carry
            return jnp.linalg.norm(delta_x) > eps

        def outer_step_scan(carry, x):
            return outer_step(carry), ()

        def outer_step(carry):
            samples, delta_lambda_prev, delta_x_prev, i, _ = carry
            delta_lambda = jnp.zeros(shape_alt)
            delta_x = jnp.zeros(shape)

            # for i in range(2):
            #     (delta_lambda, delta_lambda_prev, delta_x, delta_x_prev, samples, j) = joint_step((delta_lambda, delta_lambda_prev, delta_x, delta_x_prev, samples, 0))
            #     print("inner norm lambda", jnp.linalg.norm(delta_lambda - delta_lambda_prev))
            #     print("inner norm x", jnp.linalg.norm(delta_x - delta_x_prev))
            # (delta_lambda, delta_lambda_prev, delta_x, delta_x_prev, samples, j) = joint_step((delta_lambda, delta_lambda_prev, delta_x, delta_x_prev, samples, 0))

            (delta_lambda, delta_lambda_prev, delta_x, delta_x_prev, samples, j) = while_loop(inner_condition_function, joint_step, (delta_lambda, delta_lambda_prev, delta_x, delta_x_prev, samples, 0))

            samples = samples + delta_x
            # print(samples[-1])
            i += 1

            return (samples, delta_lambda, delta_x, i, j)

        delta_lambda = jnp.full(shape_alt, jnp.inf)
        delta_x = jnp.full(shape, jnp.inf)

        # for i in range(1000):
        #     (samples, delta_lambda, delta_x, i, j) = outer_step((samples, delta_lambda, delta_x, i, 0))
        #     print(delta_x)
        #     print("outer norm x", jnp.linalg.norm(delta_x[:-1]))

        (samples, delta_lambda, delta_x, i, j), () = scan(outer_step_scan, (samples, delta_lambda, delta_x, 0, 0), jnp.arange(1000))

        # (samples, delta_lambda, delta_x, i, j) = while_loop(outer_condition_function, outer_step, (samples, delta_lambda, delta_x, 0, 0))

        return samples, delta_lambda, delta_x

    # return jax.pmap(sampler, axis_name='batch')
    return map_sampler
