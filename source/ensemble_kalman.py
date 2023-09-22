"""Ensemble Kalman filter.

Ported from the Julia code: https://github.com/Zhengyu-Huang/InverseProblems.jl/blob/master/Inversion/IKF.jl
"""
import jax
import jax.numpy as jnp
from jax.lax import scan
import jax.random as random
from jax import vmap, jacfwd, vjp
from diffusionjax.utils import batch_mul
from diffusionjax.solvers import Solver
from grfjax.utils import trunc_svd
from grfjax.inverse_problems import get_estimate_x_0
import functools
import numpy as np


def batch_matmul(A, x):
    return vmap(lambda x: A @ x.T)(x)


def get_ensemble_sampler(shape, outer_solver, y, noise_std, denoise=False, stack_samples=False, inverse_scaler=None):
    """Get an ensemble kalman sampler from (possibly interleaved) numerical solver(s).

    Args:
        shape: Shape of array, x. (num_samples,) + x_shape, where x_shape is the shape
            of the object being sampled from, for example, an image may have
            x_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
        outer_solver: A valid numerical solver class that will act on an outer loop.
        inner_solver: '' that will act on an inner loop.
        denoise: Boolean variable that if `True` applies one-step denoising to final samples.
        stack_samples: Boolean variable that if `True` return all of the sample path or
            just returns the last sample.
    Returns:
        Samples and the number of score function (model) evaluations.
    """
    if inverse_scaler is None: inverse_scaler = lambda x: x

    def sampler(rng, x_0=None):
        """
        Args:
            rng: A JAX random state.
            x_0: Initial condition. If `None`, then samples an initial condition from the
                sde's initial condition prior. Note that this initial condition represents
                `x_T \sim \text{Normal}(O, I)` in reverse-time diffusion.
        Returns:
            Samples.
        """
        outer_update = functools.partial(outer_solver.update, y=y, noise_std=noise_std)
        outer_ts = outer_solver.ts
        num_function_evaluations = jnp.size(outer_ts) * shape[0]
        def outer_step(carry, t):
            rng, x, x_mean = carry
            vec_t = jnp.full((shape[0], 1), t)  # TODO ref0 This causes a shape bug? working with example1
            # vec_t = jnp.full((shape[0],), t)  # TODO ref1
            rng, step_rng = random.split(rng)
            x, x_mean = outer_update(step_rng, x, vec_t)
            if not stack_samples:
                return (rng, x, x_mean), ()
            else:
                return ((rng, x, x_mean), x_mean) if denoise else ((rng, x, x_mean), x)

        rng, step_rng = random.split(rng)

        # Kalman filter computations require x to be flattened
        if x_0 is None:
            x = outer_solver.sde.prior(step_rng, shape).reshape(shape[0], -1)
        else:
            assert(x_0.shape==shape)
            x = x_0.reshape(shape[0], -1)
        if not stack_samples:
            (_, x, x_mean), _ = scan(outer_step, (rng, x, x), outer_ts, reverse=True)
            return inverse_scaler(x_mean if denoise else x), num_function_evaluations
        else:
            (_, _, _), xs = scan(outer_step, (rng, x, x), outer_ts, reverse=True)
            return inverse_scaler(xs), num_function_evaluations
    # return jax.pmap(sampler, axis_name='batch')
    return sampler


class EnsembleKalmanFilter(Solver):
    """EKF abstract class for all concrete Ensemble Kalman Filter solvers.
    Functions are designed for an ensemble of inputs.
    TODO: It really requires inheriting a Solver class?
    """

    def __init__(self, shape, sde, observation_map, innovation, num_steps):
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
        if innovation is True:
            self.innovation = self.noised_innovation
        elif innovation is False:
            self.innovation = self.not_noised_innovation
        elif innovation is not None:
            self.innovation = innovation

    def batch_observation_map(self, x, t):
        return vmap(lambda x: self.observation_map(x, t))(x)

    def predict(self, rng, x, t, noise_std):
        x = x.reshape(self.shape)
        drift, diffusion = self.sde.sde(x, t)  # ref0
        # t_flat = t[:, 0]  # ref1
        # drift, diffusion = self.sde.sde(x, t_flat)  # ref1
        # TODO: possible reshaping needs to occur here, if score
        # applies to an image vector
        alpha = self.sde.mean_coeff(t)[0]**2  # ref0
        # alpha = self.sde.mean_coeff(t)**2  # ref1
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

    # def preconditioned_predict(self, rng, x, t, noise_std, X_t):
    #     drift, diffusion = self.sde.sde(x, t)
    #     alpha = self.sde.mean_coeff(t)[0]**2  # ref0
    #     R = alpha * noise_std**2
    #     eta = jnp.sqrt(R)
    #     f = drift * self.dt
    #     G = diffusion * jnp.sqrt(self.dt)

    #     X = X_t.T @ X_t  # (np.prod(x_shape), np.prod(x_shape))  NOTE this matrix is rank deficiant when num_ensemble > np.prod(x_shape) (it has rank np.prod(x_shape))
    #     U, S, _ = trunc_svd(X, np.prod(self.shape[1:]), hermitian=True)  # O(num_ensemble^{3}) Truncate due to rank deficiency
    #     noise = batch_matmul(U, random.normal(rng, x.shape) * jnp.sqrt(S))
    #     # noise = batch_matmul(X_t.T, random.normal(rng, (self.shape[0], self.shape[0])))
    #     x_hat_mean = x + batch_matmul(X, f)
    #     x_hat = x_hat_mean + G * noise
    #     return x_hat, x_hat_mean

    def not_noised_innovation(self, rng, h_hat, y, noise_std, t):
        sqrt_alpha = self.sde.mean_coeff(t)[0]
        v = self.sde.variance(t)[0]
        R = v / sqrt_alpha + noise_std**2

        rng, step_rng = random.split(rng)
        noise = random.normal(rng, (self.shape[0], y.shape[0]))
        y_hat = h_hat + jnp.sqrt(R) * noise  # Simulate stochastic data (self.shape[0], num_y)
        y_dagger = jnp.expand_dims(y, axis=0)  # (1, num_y)
        return y_dagger, y_hat, noise_std**2

    def noised_innovation(self, rng, h_hat, y, noise_std, t):
        # ref0
        sqrt_alpha = self.sde.mean_coeff(t)[0]
        alpha = sqrt_alpha**2
        v = self.sde.variance(t)[0]
        # # ref1
        # sqrt_alpha = self.sde.mean_coeff(t)
        # alpha = sqrt_alpha**2
        # v = self.sde.variance(t)
        R = alpha * noise_std**2
        rng, step_rng = random.split(rng)
        noise = random.normal(rng, (self.shape[0], y.shape[0]))
        # y_hat = h_hat + jnp.sqrt(R) * noise  # Simulate stochastic data
        y_hat = h_hat
        # TODO: should this be different noise?
        rng, step_rng = random.split(rng)
        # TODO: before I was not adding noise to y, just y_hat, this seemed to work just as well
        # TODO: should generate noise for each particle, since it should be an ensemble

        # no noise
        # noise = random.normal(rng, (self.shape[0], np.prod(self.shape[1:])))
        # y_dagger = sqrt_alpha * y + jnp.sqrt(v) * self.batch_observation_map(noise, t) * 0.0
        y_dagger = sqrt_alpha * y

        # independent noise each particle - default
        # noise = random.normal(rng, (self.shape[0], np.prod(self.shape[1:])))
        # y_dagger = sqrt_alpha * y + jnp.sqrt(v) * self.batch_observation_map(noise, t)

        # same noise for each particle
        # noise = random.normal(rng, (1, np.prod(self.shape[1:])))
        # y_dagger = sqrt_alpha * y + jnp.sqrt(v) * self.batch_observation_map(noise, t)
        return y_dagger, y_hat, R

    def empirical_square_roots(self, x_hat, h_hat):
        x_m_hat = jnp.mean(x_hat, axis=0)
        h_m_hat = jnp.mean(h_hat, axis=0)

        # Construct square root matrix for \hat{x} - \hat{m}
        X_hat_t = x_hat - x_m_hat  # (num_ensemble, N_x)
        X_hat_t = X_hat_t / jnp.sqrt(self.shape[0] - 1)

        # Construct the square root matrix for g - g_mean
        H_hat_t = h_hat - h_m_hat  # (num_ensemble, N_y)
        H_hat_t = H_hat_t / jnp.sqrt(self.shape[0] - 1)
        return x_m_hat, h_m_hat, X_hat_t, H_hat_t

    def update(self):
        r"""Return the drift and diffusion coefficients of the SDE.

        Args:

        Returns:
            x: A JAX array of the next state.
            x_mean: A JAX array of the next state without noise, for denoising.
        """


class StochasticSVD(EnsembleKalmanFilter):
    """Stochastic Ensemble Kalman Filter"""
    def __init__(self, num_y, shape, sde, observation_map, innovation, num_steps):
        super().__init__(shape, sde, observation_map, innovation, num_steps)
        if num_y < shape[0]:
            self.nudge_y = self.nudge_y_pseudo
        else:
            self.nudge_y = self.nudge_y_full
        self.update = self._update

    def nudge_y_pseudo(self, X_hat_t, H_hat_t, R, y, y_hat):
        # TODO: WARNING trunc_svd still scales with num_ensemble^{3}, since truncation happens after!
        Y = H_hat_t @ H_hat_t.T  # (num_ensemble, num_ensemble)  NOTE this matrix is rank deficiant when num_ensemble > N_y (it has rank N_y)
        U, S, _ = trunc_svd(Y, H_hat_t.shape[1], hermitian=True)  # O(num_ensemble^{3}) Truncate due to rank deficiency. This is wasteful computationally?
        # return (y - y_hat) @ A.T
        return batch_matmul(X_hat_t.T @ ((U / (R + S)) @ U.T) @ H_hat_t, y - y_hat)

    def nudge_y_full(self, X_hat_t, H_hat_t, R, y, y_hat):
        Y = H_hat_t @ H_hat_t.T  # (num_ensemble, num_ensemble)
        U, S, _ = jnp.linalg.svd(Y, hermitian=True)  # O(num_ensemble^{3})
        return batch_matmul(X_hat_t.T, batch_matmul(U, (batch_matmul(U.T, (batch_matmul(H_hat_t, (y - y_hat))) / (R + S)))))

    def nudge_y_pseudo_alt(self, X_hat_t, H_hat_t, R, y, y_hat):
        """TODO: Alternate algorithms to consider, that should be stable when R->0 as t->1"""
        Y = H_hat_t.T @ H_hat_t + R * jnp.identity(H_hat_t.shape[1])  # (N_y, N_y)
        # TODO: since assumption here is that N_y << num_ensemble, then matrix computations should be done in N_y space, not num_ensemble space
        # TODO: IS THIS CORRECT? SHOULD THERE NOT BE A H_hat_t term after
        return batch_matmul(X_hat_t.T, batch_matmul(H_hat_t, jnp.linalg.solve(Y, (y - y_hat).T).T))
        # U, S, _ = jnp.linalg.svd(Y, hermitian=True)
        # return batch_matmul(X_hat_t.T, batch_matmul(H_hat_t, batch_matmul(U, batch_matmul(U.T, y - y_hat) / S)))

    def nudge_y_full_alt(self, X_hat_t, H_hat_t, R, y, y_hat):
        Y = H_hat_t @ H_hat_t.T  # (num_ensemble, num_ensemble)  NOTE this matrix is rank deficiant when num_ensemble > N_y (it has rank N_y)
        U, S, _ = jnp.linalg.svd(Y, hermitian=True)
        return batch_matmul(X_hat_t.T @ ((U @ U.T) / (R + S)) @ H_hat_t, y - y_hat)

    def _update(self, rng, x, t, y, noise_std):
        x_hat, x_hat_mean = self.predict(rng, x, t, noise_std)
        h_hat = self.batch_observation_map(x_hat, t)
        y_dagger, y_hat, R = self.innovation(rng, h_hat, y, noise_std, t)
        x_m_hat, h_m_hat, X_hat_t, H_hat_t = self.empirical_square_roots(x_hat, h_hat)

        nudge_y = self.nudge_y(X_hat_t, H_hat_t, R, y_dagger, y_hat)
        drift, diffusion = self.sde.forward_sde(x, t)
        # None of these scalings work so well
        # nudge_y = self.dt * batch_mul(diffusion**2, nudge_y) * self.sde.mean_coeff(t)[0] / self.sde.variance(t)[0]  # 1 too low variance w2 rate -0.5, d2 rate -1.0
        # nudge_y = nudge_y * self.sde.variance(t)[0]  # 0 promising but w2 error levels off 10^0 slower than -0.5 rate and much higher error than exact sampling
        # nudge_y = nudge_y * self.sde.mean_coeff(t)[0]  # 2 way too low variance w^2 error levels off 10^0
        # nudge_y = self.dt * batch_mul(diffusion**2, nudge_y)  # 3 too high variance w2 error levels off 10^1
        # nudge_y = self.dt * batch_mul(diffusion**2, nudge_y) * self.sde.mean_coeff(t)[0]  # 4 too high variance w2 error levels off 10^1
        # nudge_y = nudge_y * self.sde.variance(t)[0] / self.sde.mean_coeff(t)[0]  # seems promising for small dimensions, unstable high dims 5
        # nudge_y = self.dt * batch_mul(diffusion**2, nudge_y) * self.sde.variance(t)[0] / self.sde.mean_coeff(t)[0]  # seems wrong, unstable high dim
        # nudge_y = batch_mul(diffusion**2, nudge_y) * self.sde.variance(t)[0] / self.sde.mean_coeff(t)[0]  # Unstable low and high dims
        # nudge_y = batch_mul(diffusion**2, nudge_y) / self.sde.mean_coeff(t)[0]  # Unstable and high low dims
        # nudge_y = nudge_y * self.sde.mean_coeff(t)[0] / self.sde.variance(t)[0]  # Unstable low and high dims

        x = x_hat + nudge_y
        x_mean = x_hat_mean + nudge_y
        return x, x_mean

    # def preconditioned_update(self, rng, x, t, y, noise_std):
    #     x_m = jnp.mean(x, axis=0)
    #     # Construct square root matrix for x - m
    #     X_t = x- x_m  # (num_ensemble, N_x)
    #     X_t = X_t / jnp.sqrt(self.shape[0] - 1)

    #     x_hat, x_hat_mean = self.preconditioned_predict(rng, x, t, noise_std, X_t)
    #     h_hat = self.batch_observation_map(x_hat, t)
    #     y_dagger, y_hat, R = self.innovation(rng, h_hat, y, noise_std, t)
    #     x_m_hat, h_m_hat, X_hat_t, H_hat_t = self.empirical_square_roots(x_hat, h_hat)

    #     nudge_y = self.nudge_y(X_hat_t, H_hat_t, R, y_dagger, y_hat)
    #     x = x_hat + nudge_y
    #     x_mean = x_hat_mean + nudge_y
    #     return x, x_mean


class StochasticChol(StochasticSVD):
    """Stochastic Ensemble Kalman Filter"""
    def __init__(self, num_y, shape, sde, observation_map, innovation, num_steps):
        super().__init__(num_y, shape, sde, observation_map, innovation, num_steps)
        if num_y < shape[0]:  # num_y < num_ensemble
            # for num_y < num_ensemble, H_hat @ H_hat.T is not full rank
            self.nudge_y = self.nudge_y_pseudo
            # raise NotImplementedError("It is necessary to use the SVD pseudo-inverse method in this case.")
        else:
            self.nudge_y = self.nudge_y_full

    def nudge_y_pseudo(self, X_hat_t, H_hat_t, R, y, y_hat):
        Y = H_hat_t.T @ H_hat_t + R * jnp.identity(H_hat_t.shape[1])  # (N_y, N_y)
        return batch_matmul(X_hat_t.T, batch_matmul(H_hat_t, jnp.linalg.solve(Y, (y - y_hat).T).T))  # O(N_y^{3})

    def nudge_y_full(self, X_hat_t, H_hat_t, R, y, y_hat):
        Y = (H_hat_t / R) @ H_hat_t.T + jnp.identity(H_hat_t.shape[0])  # (num_ensemble, num_ensemble)
        return batch_matmul(X_hat_t.T, jnp.linalg.solve(Y, batch_matmul(H_hat_t, ((y - y_hat) / R)).T))


class AdjustmentSVD(EnsembleKalmanFilter):
    # TODO: supercede with Stuart filters
    """Square-root (determinstic) Ensemble Adjustment Kalman Filter"""
    def __init__(self, num_y, shape, sde, observation_map, innovation, num_steps):
        super().__init__(shape, sde, observation_map, innovation, num_steps)
        if num_y < shape[0]:
            # Invert in data space
            self.nudge_y = self.nudge_y_pseudo
            self.nudge_x = self.nudge_x_pseudo
        else:
            # Invert in ensemble space
            self.nudge_y = self.nudge_y_full
            self.nudge_x = self.nudge_x_full

    def nudge_y_pseudo(self, X_hat_t, H_hat_t, R, y, h_m_hat):
        X = H_hat_t @ H_hat_t.T / R  # (num_ensemble, num_ensemble)
        P, S, _ = trunc_svd(X, H_hat_t.shape[1], hermitian=True)
        # If have multiple copies of noised data y_dagger, then this operation does not work - could batch it
        # return X_hat_t.T @ ((P / (S + 1.0)) @ P.T) @ (H_hat_t @ ((y - h_m_hat) / R))
        # TODO no point batch multiplying across particles that what is not necessary
        # Depends on which is larger dimension
        # return batch_matmul(X_hat_t.T, batch_matmul(P / (S + 1.0), batch_matmul(P.T, (H_hat_t @ (y - h_m_hat) / R))))
        return batch_matmul(X_hat_t.T, batch_matmul(P / (S + 1.0), batch_matmul(P.T, batch_matmul(H_hat_t, (y - h_m_hat) / R))))

    def nudge_x_pseudo(self, X_hat_t, H_hat_t, R, x_hat, x_m_hat):
        X = H_hat_t @ H_hat_t.T / R  # (num_ensemble, num_ensemble) # TODO: not the correct SVD
        P, Gamma, _ = trunc_svd(X, H_hat_t.shape[1], hermitian=True)
        F, sqrt_D, V_t = jnp.linalg.svd(X_hat_t.T, full_matrices=False)
        # F, sqrt_D, V_t = trunc_svd(X_hat_t.T, Z_t.shape[1], full_matrices=False)
        # print(F.shape)  # (N_x, N_x)
        # print(V_t.shape)  # (num_ensemble, N_x)
        # print(P.shape)  # (num_ensemble, num_ensemble)
        Y_ = V_t @ P
        Y = (Y_ / (Gamma + 1.0)) @ Y_.T
        # print(Y.shape)  # (N_x, N_x)
        # U, D, _ = trunc_svd(Y, X_hat_t.shape[1], hermitian=True)
        U, D, _ = jnp.linalg.svd(Y, hermitian=True)
        # print(U.shape)  # (N_x, N_x)
        A = (F * sqrt_D) @ (U * jnp.sqrt(D)) @ (F.T / sqrt_D)
        return (x_hat - x_m_hat) @ A

    def nudge_y_full(self, X_hat_t, H_hat_t, R, y, h_m_hat):
        # TODO: This won't work for N_y < num_ensemble
        X = H_hat_t @ H_hat_t.T / R  # (num_ensemble, num_ensemble)
        P, Gamma, _ = jnp.linalg.svd(X, hermitian=True)
        # TODO: may need to batch this to be compatible with y_dagger
        return X_hat_t.T @ (P @ (P.T @ (H_hat_t @ ((y - h_m_hat) / R))) / (Gamma + 1.0))
        # TODO no point batch multiplying across particles that what is not necessary
        # Depends on which is larger dimension
        # return batch_matmul(X_hat_t.T, batch_matmul(P, batch_matmul(P.T, (H_hat_t @ (y - h_m_hat) / R) / (Gamma + 1.0))))
        # return batch_matmul(X_hat_t.T, batch_matmul(P, (batch_matmul(P.T, batch_matmul(H_hat_t, (y - h_m_hat) / R)) / (Gamma + 1.0))))

    def nudge_x_full(self, X_hat_t, H_hat_t, R, x_hat, x_m_hat):
        # TODO: This won't work for num_y < num_ensemble
        X = H_hat_t @ H_hat_t.T / R  # (num_ensemble, num_ensemble) # TODO: not the correct SVD
        P, Gamma, _ = jnp.linalg.svd(X, hermitian=True)
        F, sqrt_D, V_t = jnp.linalg.svd(X_hat_t.T, full_matrices=False)
        # print(F.shape)  # (N_x, N_x)
        # print(V_t.shape)  # (num_ensemble, N_x)
        # print(P.shape)  # (num_ensemble, num_ensemble)
        Y_ = V_t @ P
        Y = (Y_ / (Gamma + 1.0)) @ Y_.T
        # print(Y.shape)  # (N_x, N_x)
        U, D, _ = jnp.linalg.svd(Y, hermitian=True)
        # print(U.shape)  # (N_x, N_x)
        A = (F * sqrt_D) @ (U * jnp.sqrt(D)) @ (F.T / sqrt_D)
        return (x_hat - x_m_hat) @ A

    def update(self, rng, x, t, y, noise_std):
        x_hat, x_hat_mean = self.predict(rng, x, t, noise_std)
        h_hat = self.batch_observation_map(x_hat, t)
        y_dagger, y_hat, R = self.innovation(rng, h_hat, y, noise_std, t)
        x_m_hat, h_m_hat, X_hat_t, H_hat_t = self.empirical_square_roots(x_hat, h_hat)

        nudge_y = self.nudge_y(X_hat_t, H_hat_t, R, y_dagger, h_m_hat)
        x_m = x_m_hat + nudge_y
        x_mean = x_hat_mean + nudge_y

        nudge_x = self.nudge_x(X_hat_t, H_hat_t, R, x_hat, x_m_hat)
        x = x_m + nudge_x
        x_mean = x_mean + nudge_x
        return x, x_mean


class AdjustmentChol(EnsembleKalmanFilter):
    """Square-root (deterministic) Ensemble Adjustment Kalman Filter, as written in (Calvello 2022)
    with SVD."""
    def __init__(self, num_y, shape, sde, observation_map, innovation, num_steps):
        super().__init__(shape, sde, observation_map, innovation, num_steps)
        if num_y < shape[0]:
            # TODO: check this
            # raise NotImplementedError("It is necessary to use the SVD pseudo-inverse method in this case.")
            # Invert in data space
            self.nudge_y = self.nudge_y_pseudo
            self.nudge_x = self.nudge_x_pseudo
            self.update = self.update_pseudo
        else:
            # Invert in ensemble space
            self.nudge_y = self.nudge_y_full
            self.nudge_x = self.nudge_x_full
            self.update = self.update_full

    def nudge_y_full(self, X_hat_t, H_hat_t, R, x_hat, x_m_hat):
        Y = (H_hat_t / R) @ H_hat_t.T + jnp.identity(H_hat_t.shape[0])  # (num_ensemble, num_ensemble)
        L = jax.scipy.linalg.cholesky(Y, lower=True)
        x = jax.scipy.linalg.solve_triangular(L, batch_matmul(H_hat_t, ((y - h_m_hat) / R)).T, lower=True)
        x = jax.scipy.linalg.solve_triangular(L.T, x, lower=False)
        return L, batch_matmul(X_hat_t.T, x)

    def nudge_x_full(self, L, X_hat_t, H_hat_t, R, x_hat, x_m_hat):
        print((x_hat - x_m_hat).T.shape)
        print((jnp.linalg.solve(X_hat_t.T, (x_hat - x_m_hat).T).T).shape)
        print((jax.scipy.linalg.solve_triangular(L, jnp.linalg.solve(X_hat_t.T, (x_hat - x_m_hat).T).T)).shape)
        assert 0  # TODO: complete
        return batch_matmul(X_hat_t.T, jax.scipy.linalg.solve_triangular(L, jnp.linalg.solve(X_hat_t.T, (x_hat - x_m_hat).T).T))

    def nudge_y_pseudo(self, X_hat_t, H_hat_t, R, y, h_m_hat):
        Y = H_hat_t.T @ H_hat_t + jnp.identity(H_hat_t.shape[1]) * R  # (num_y, num_y)
        L = jax.scipy.linalg.cholesky(Y, lower=True)
        i = jax.scipy.linalg.solve_triangular(L, (y - h_m_hat).T, lower=True)  # O(num_y^{3} * num_ensemble)
        i = jax.scipy.linalg.solve_triangular(L.T, i, lower=False)
        # TODO: works for an ensemble of datapoints, y
        return L, batch_matmul(X_hat_t.T, batch_matmul(H_hat_t, i.T))
        # TODO: for one datapoint, y
        # return L, X_hat_t.T @ H_hat_t @ x

    def nudge_x_pseudo(self, L, X_hat_t, H_hat_t, R, h_hat, h_m_hat):
        Y = H_hat_t.T @ H_hat_t + jnp.identity(H_hat_t.shape[1]) * R  # (num_y, num_y)
        Y  = Y + L * jnp.sqrt(R)
        return batch_matmul(X_hat_t.T, batch_matmul(H_hat_t, jnp.linalg.solve(Y, (h_hat - h_m_hat).T).T))

    def update_pseudo(self, rng, x, t, y, noise_std):
        x_hat, x_hat_mean = self.predict(rng, x, t, noise_std)
        h_hat = self.batch_observation_map(x_hat, t)
        y_dagger, y_hat, R = self.innovation(rng, h_hat, y, noise_std, t)
        x_m_hat, h_m_hat, X_hat_t, H_hat_t = self.empirical_square_roots(x_hat, h_hat)

        L, nudge_y = self.nudge_y(X_hat_t, H_hat_t, R, y_dagger, h_m_hat)
        x_m = x_m_hat + nudge_y
        x_mean = x_hat_mean + nudge_y

        nudge_x = self.nudge_x(L, X_hat_t, H_hat_t, R, h_hat, h_m_hat)
        x = x_m + (x_hat - x_m_hat) - nudge_x
        x_mean = x_mean + (x_hat - x_hat_mean) - nudge_x
        return x, x_mean

    def update_full(self, rng, x, t, y, noise_std):
        x_hat, x_hat_mean = self.predict(rng, x, t, noise_std)
        h_hat = self.batch_observation_map(x_hat, t)
        y_dagger, y_hat, R = self.innovation(rng, h_hat, y, noise_std, t)
        x_m_hat, h_m_hat, X_hat_t, H_hat_t = self.empirical_square_roots(x_hat, h_hat)

        L, nudge_y = self.nudge_y(X_hat_t, H_hat_t, R, y_dagger, h_m_hat)
        x_m = x_m_hat + nudge_y
        x_mean = x_hat_mean + nudge_y

        nudge_x = self.nudge_x(L, X_hat_t, H_hat_t, R, x_hat, x_m_hat)
        x = x_m + nudge_x
        x_mean = x_mean + nudge_x
        return x, x_mean


class TransformSVD(EnsembleKalmanFilter):
    # TODO: supercede with Stuart filters
    """Square-root (determinstic) Ensemble Transform Kalman Filter"""
    def __init__(self, num_y, shape, sde, observation_map, innovation, num_steps):
        super().__init__(shape, sde, observation_map, innovation, num_steps)
        if num_y < shape[0]:
            self.nudge_y = self.nudge_y_full
            # self.nudge_x = self.nudge_x_full
            self.update = self.update_pseudo
        else:
            self.nudge_y = self.nudge_y_full
            # self.nudge_x = self.nudge_x_full
            self.update = self.update_full

    def nudge_y_full(self, P, S, X_hat_t, H_hat_t, R, y, h_m_hat):
        # return X_hat_t.T @ ((P / (S + 1.0)) @ P.T) @ (H_hat_t @ ((y - h_m_hat) / R))
        return batch_matmul(X_hat_t.T, batch_matmul(batch_matmul((P / (S + 1.0)), P.T), batch_matmul(H_hat_t, ((y - h_m_hat) / R))))

    def nudge_y_full_alt(self, P, S, X_hat_t, H_hat_t, R, y, h_m_hat):
        X = H_hat_t @ Y_p_t.T  # (num_ensemble, num_ensemble)
        return X_hat_t.T @ ((P / (S + R)) @ P.T) @ H_hat_t @ (y - h_m_hat)
        # return X_hat_t.T @ (P @ (P.T @ (H_hat_t @ (y - h_m_hat)) / (R + S)))

    def update_full(self, rng, x, t, y, noise_std):
        x_hat, x_hat_mean = self.predict(rng, x, t, noise_std)
        h_hat = self.batch_observation_map(x_hat, t)
        y_dagger, y_hat, R = self.innovation(rng, h_hat, y, noise_std, t)
        x_m_hat, h_m_hat, X_hat_t, H_hat_t = self.empirical_square_roots(x_hat, h_hat)

        X = (H_hat_t / R) @ H_hat_t.T  # (num_ensemble, num_ensemble)
        P, S, _ = jnp.linalg.svd(X, hermitian=True)

        nudge_y = self.nudge_y(P, S, X_hat_t, H_hat_t, R, y_dagger, h_m_hat)
        x_m = x_m_hat + nudge_y
        x_mean = x_hat_mean + nudge_y

        # Original ETKF is T = P @ (Gamma + 1.)^{-1/2}, but it is biased
        T = (P / jnp.sqrt(S + 1.0)) @ P.T
        nudge_x = T.T @ (x_hat - x_m_hat)  # (num_ensemble, num_ensemble)
        x = nudge_x + x_m
        x_mean = nudge_x + x_mean
        return x, x_mean

    def update_pseudo(self, rng, x, t, y, noise_std):
        x_hat, x_hat_mean = self.predict(rng, x, t, noise_std)
        h_hat = self.batch_observation_map(x_hat, t)
        y_dagger, y_hat, R = self.innovation(rng, h_hat, y, noise_std, t)
        x_m_hat, h_m_hat, X_hat_t, H_hat_t = self.empirical_square_roots(x_hat, h_hat)

        X = (H_hat_t / R) @ H_hat_t.T  # (num_ensemble, num_ensemble)
        P, S, _ = trunc_svd(X, rank=H_hat_t.shape[0], hermitian=True)

        nudge_y = self.nudge_y(P, S, X_hat_t, H_hat_t, R, y_dagger, h_m_hat)
        x_m = x_m_hat + nudge_y
        x_mean = x_hat_mean + nudge_y

        # Original ETKF is T = P @ (Gamma + 1.)^{-1/2}, but it is biased
        T = (P / jnp.sqrt(S + 1.0)) @ P.T
        nudge_x = T.T @ (x_hat - x_m_hat)  # (num_ensemble, num_ensemble)
        x = nudge_x + x_m
        x_mean = nudge_x + x_mean
        return x, x_mean


class TransformChol(EnsembleKalmanFilter):
    """Square-root (deterministic) Ensemble Transform Kalman Filter, as written in (Calvello 2022)
    with Cholesky/ linear solve."""
    def __init__(self, num_y, shape, sde, observation_map, innovation, num_steps):
        super().__init__(shape, sde, observation_map, innovation, num_steps)
        if num_y < shape[0]:
            raise NotImplementedError("It is necessary to use the SVD pseudo-inverse method in this case.")
            # self.nudge_y = self.nudge_y_pseudo
            # self.nudge_x = self.nudge_x_pseudo
            # self.update = self.update_psuedo
        else:
            self.nudge_y = self.nudge_y_full
            self.nudge_x = self.nudge_x_full
            self.update = self.update_full

    def nudge_y_pseudo(self, L, X_hat_t, H_hat_t, R, h_m_hat):
        return

    def nudge_x_pseudo(self, L, X_hat_t, H_hat_t, R, h_m_hat):
        return

    def nudge_y_full(self, L, X_hat_t, H_hat_t, R, y, h_m_hat):
        x = jax.scipy.linalg.solve_triangular(L, (H_hat_t @ ((y - h_m_hat) / R)).T, lower=True)
        x = jax.scipy.linalg.solve_triangular(L.T, x, lower=False)
        return X_hat_t.T @ x

    def nudge_x_full(self, L, x_hat, x_m_hat):
        return jnp.linalg.solve(L, (x_hat - x_m_hat))
    
    def update_full(self, rng, x, t, y, noise_std):
        x_hat, x_hat_mean = self.predict(rng, x, t, noise_std)
        h_hat = self.batch_observation_map(x_hat, t)
        y_dagger, y_hat, R = self.innovation(rng, h_hat, y, noise_std, t)
        x_m_hat, h_m_hat, X_hat_t, H_hat_t = self.empirical_square_roots(x_hat, h_hat)

        Y = (H_hat_t / R) @ H_hat_t.T + jnp.identity(H_hat_t.shape[0])  # (num_ensemble, num_ensemble)
        L = jax.scipy.linalg.cholesky(Y, lower=True)

        nudge_y = self.nudge_y(L, X_hat_t, H_hat_t, R, y_dagger, h_m_hat)
        x_m = x_m_hat + nudge_y
        x_mean = x_hat_mean + nudge_y

        nudge_x = self.nudge_x(L, x_hat, x_m_hat)
        x = x_m + nudge_x
        x_mean = x_mean + nudge_x

        return x, x_mean

    def update_psuedo(self, rng, x, t, y, noise_std):
        x_hat, x_hat_mean = self.predict(rng, x, t, noise_std)
        h_hat = self.batch_observation_map(x_hat, t)
        y_dagger, y_hat, R = self.innovation(rng, h_hat, y, noise_std, t)
        x_m_hat, h_m_hat, X_hat_t, H_hat_t = self.empirical_square_roots(x_hat, h_hat)

        Y = H_hat_t.T @ H_hat_t + R * jnp.identity(H_hat_t.shape[1])  # (num_y, num_y)
        L = jax.scipy.linalg.cholesky(Y, lower=True)

        nudge_y = self.nudge_y(L, X_hat_t, H_hat_t, R, y_dagger, h_m_hat)
        x_m = x_m_hat + nudge_y
        x_mean = x_hat_mean + nudge_y

        nudge_x = self.nudge_x(L, x_hat, x_m_hat)
        x = x_m + nudge_x
        x_mean = x_mean + nudge_x

        return x, x_mean


class ProjectionSVD(EnsembleKalmanFilter):
    """Projection Ensemble Kalman Filter, TODO: could inherit StochasticSVD? Do this once decide on a paper"""
    def __init__(self, num_y, shape, sde, observation_map, innovation, num_steps):
        super().__init__(shape, sde, observation_map, innovation, num_steps)
        if num_y < shape[0]:
            self.nudge_y = self.nudge_y_pseudo
        else:
            self.nudge_y = self.nudge_y_full
        self.estimate_x_0 = get_estimate_x_0(self.sde, self.sde.score, shape[1:])
        self.update = self.projection_update
        self.innovation = self.projection_innovation

    def batch_estimate(self, x, t):
        return vmap(lambda x, t: self.estimate_x_0(x, t), in_axes=(0, 0), out_axes=(0, 0))(x, t)

    def nudge_y_pseudo(self, X_hat_t, H_hat_t, R, y, y_hat):
        # TODO: WARNING trunc_svd still scales with num_ensemble^{3}, since truncation happens after!
        Y = H_hat_t @ H_hat_t.T  # (num_ensemble, num_ensemble)  NOTE this matrix is rank deficiant when num_ensemble > N_y (it has rank N_y)
        U, S, _ = trunc_svd(Y, H_hat_t.shape[1], hermitian=True)  # O(num_ensemble^{3}) Truncate due to rank deficiency. This is wasteful computationally?
        # return (y - y_hat) @ A.T
        return batch_matmul(X_hat_t.T @ ((U / (R + S)) @ U.T) @ H_hat_t, y - y_hat)

    def nudge_y_full(self, X_hat_t, H_hat_t, R, y, y_hat):
        Y = H_hat_t @ H_hat_t.T  # (num_ensemble, num_ensemble)
        U, S, _ = jnp.linalg.svd(Y, hermitian=True)  # O(num_ensemble^{3})
        return batch_matmul(X_hat_t.T, batch_matmul(U, (batch_matmul(U.T, (batch_matmul(H_hat_t, (y - y_hat))) / (R + S)))))

    def projection_innovation(self, rng, h_hat, y, noise_std):
        y_dagger = jnp.expand_dims(y, axis=0)  # (1, num_y)
        rng, step_rng = random.split(rng)
        noise = random.normal(rng, (self.shape[0], y.shape[0]))
        y_hat = h_hat + noise_std * noise  # Simulate stochastic data (self.shape[0], num_y)
        return y_dagger, y_hat, noise_std**2
        # sqrt_alpha = self.sde.mean_coeff(t)[0]
        # v = self.sde.variance(t)[0]
        # R = v / sqrt_alpha + noise_std**2

        # rng, step_rng = random.split(rng)
        # noise = random.normal(rng, (self.shape[0], y.shape[0]))
        # y_hat = h_hat + jnp.sqrt(R) * noise  # Simulate stochastic data (self.shape[0], num_y)
        # y_dagger = jnp.expand_dims(y, axis=0)  # (1, num_y)
        # return y_dagger, y_hat, noise_std**2

    def projection_update(self, rng, x, t, y, noise_std):
        x_hat, x_hat_mean = self.predict(rng, x, t, noise_std)  # does this improve things?
        # x_hat = x
        m_hat, scores = self.batch_estimate(x_hat, t)
        h_hat = self.batch_observation_map(m_hat, t)
        y_dagger, y_hat, R = self.innovation(rng, h_hat, y, noise_std)
        x_m_hat, h_m_hat, X_hat_t, H_hat_t = self.empirical_square_roots(m_hat, h_hat)

        nudge_y = self.nudge_y(X_hat_t, H_hat_t, R, y_dagger, y_hat)
        m = m_hat + nudge_y
        v = self.sde.variance(t)[0]
        sqrt_v = jnp.sqrt(v)
        sqrt_alpha = self.sde.mean_coeff(t)[0]

        x = sqrt_alpha * m + sqrt_v * random.normal(rng, x.shape)  # renoise denoised x
        return x, m


class ProjectionChol(ProjectionSVD):
    """Projection Ensemble Kalman Filter"""
    def __init__(self, num_y, shape, sde, observation_map, innovation, num_steps):
        super().__init__(num_y, shape, sde, observation_map, innovation, num_steps)
        if num_y < shape[0]:
            # for num_y < num_ensemble, H_hat @ H_hat.T is not full rank
            self.nudge_y = self.nudge_y_pseudo
            # raise NotImplementedError("It is necessary to use the SVD pseudo-inverse method in this case.")
        else:
            self.nudge_y = self.nudge_y_full

    def nudge_y_pseudo(self, X_hat_t, H_hat_t, R, y, y_hat):
        Y = H_hat_t.T @ H_hat_t + R * jnp.identity(H_hat_t.shape[1])  # (N_y, N_y)
        return batch_matmul(X_hat_t.T, batch_matmul(H_hat_t, jnp.linalg.solve(Y, (y - y_hat).T).T))  # O(N_y^{3})

    def nudge_y_full(self, X_hat_t, H_hat_t, R, y, y_hat):
        Y = (H_hat_t / R) @ H_hat_t.T + jnp.identity(H_hat_t.shape[0])  # (num_ensemble, num_ensemble)
        return batch_matmul(X_hat_t.T, jnp.linalg.solve(Y, batch_matmul(H_hat_t, ((y - y_hat) / R)).T))


# This projection filter is experimental
class ProjectionKalmanFilter(Solver):
    """Not an ensemble Kalman filter. Abstract class for all concrete Projection Kalman Filter solvers.
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
        self.estimate_x_0 = get_estimate_x_0(self.sde, self.sde.score, shape[1:])
        self.num_y = num_y
        self.H = H

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

    def SSanalysis(self, x_hat, t, y, noise_std, ratio):
        r"""
        Using jacfwd instead. There is no way to return value as well as Jacobian, unless in aux data.
        """
        _estimate_x_0 = lambda _x: self.estimate_x_0(_x, t)
        inv_hessian = jacfwd(_estimate_x_0, has_aux=True)  # TODO: Is there no way to define this in __init__
        C_hat_d_r_t, score = inv_hessian(x_hat)
        m_hat = (x_hat + v * score) / sqrt_alpha
        C_hat = C_hat_d_r_t * ratio
        o_hat = self.observation_map(m_hat, t)
        C_xh_hat = self.batch_observation_map(C_hat, t)
        C_hh_hat = self.batch_observation_map(C_xh_hat.T, t).T
        C_yy_hat = C_hh_hat + noise_std**2 * jnp.eye(self.num_y)
        return m_hat + C_xh_hat @ jnp.linalg.solve(C_yy_hat, y - o_hat)  # denoised x (Tweedie's + Kalman update)

    def analysis(self, x_hat, t, y, noise_std, ratio):
        _estimate_x_0 = lambda x: self.estimate_x_0(x, t)
        m_hat, vjp_estimate_x_0, score = vjp(
            _estimate_x_0, x_hat, has_aux=True)
        o_hat = self.H @ m_hat
        batch_vjp_estimate_x_0 = vmap(lambda x: vjp_estimate_x_0(x)[0])
        C_yy = ratio * batch_vjp_estimate_x_0(self.H) @ self.H.T + noise_std**2 * jnp.eye(self.num_y)
        ls = ratio * vjp_estimate_x_0(self.H.T @ jnp.linalg.solve(C_yy, y - o_hat))[0]
        return m_hat + ls # , score


class ApproxProjectionKalmanFilter(ProjectionKalmanFilter):
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
