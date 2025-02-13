"""Markov Chains."""
import jax.numpy as jnp
from jax import random, grad, vmap, vjp, jacrev
from diffusionjax.utils import batch_mul
from diffusionjax.solvers import DDIMVP, DDIMVE, SMLD, DDPM


class MPGD(DDIMVP):
  """
  TODO: This method requires a pretrained autoencoder.
  MPGD He, Murata et al. https://arxiv.org/pdf/2311.16424.pdf"""
  
  def __init__(self, scale, y, observation_map, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.):
    super().__init__(model, eta, num_steps, dt, epsilon, beta_min, beta_max)
    self.observation_map = observation_map
    self.batch_observation_map = vmap(observation_map)
    self.likelihood_score = self.get_likelihood_score(self.batch_observation_map)
    self.likelihood_score_vmap = self.get_likelihood_score_vmap(self.observation_map)
    self.scale = scale
    self.y = y
    self.noise_std = noise_std
    self.num_y = y.shape[1]
    self.axes = (0,) + tuple(range(len(shape) + 2)[2:]) + (1,)
    self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

  def get_likelihood_score_vmap(self, observation_map):
    def l2_norm(x_0, y):
      x_0 = x_0
      norm = jnp.linalg.norm(y - observation_map(x_0))
      return norm**2 / self.noise_std
    grad_l2_norm = grad(l2_norm)
    return vmap(grad_l2_norm)

  def get_likelihood_score(self, batch_observation_map):
    batch_norm = vmap(jnp.linalg.norm)
    def l2_norm(x_0, y):
      norm = batch_norm(y - batch_observation_map(x_0))
      squared_norm = jnp.sum(norm**2)
      return squared_norm / self.noise_std
    grad_l2_norm = grad(l2_norm)
    return grad_l2_norm

  def posterior(self, x, t):
    epsilon = self.model(x, t)
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
    m = self.sqrt_alphas_cumprod[timestep]
    sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
    v = sqrt_1m_alpha**2
    alpha = m**2
    m_prev = self.sqrt_alphas_cumprod_prev[timestep]
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
    alpha_prev = m_prev**2
    x_0 = batch_mul((x - batch_mul(sqrt_1m_alpha, epsilon)), 1. / m)
    # TODO: manifold projection step here which requires autoencoder to project x_0 such that likelihood score is projected onto tangent space of autoencoder
    # ls = self.likelihood_score(x_0, self.y)
    ls = self.likelihood_score_vmap(x_0, self.y)
    x_0 = x_0 - self.scale * ls
    coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
    coeff2 = jnp.sqrt(v_prev - coeff1**2)
    x_mean = batch_mul(m_prev, x_0) + batch_mul(coeff2, epsilon)
    std = coeff1
    return x_mean, std


class KGDMVP(DDIMVP):
  """PiGDM Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""
  def __init__(self, y, observation_map, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.):
    super().__init__(model, eta, num_steps, dt, epsilon, beta_min, beta_max)
    self.estimate_h_x_0 = self.get_estimate_x_0(observation_map)
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
    self.batch_analysis_vmap = vmap(self.analysis)
    self.y = y
    self.noise_std = noise_std
    self.num_y = y.shape[1]
    self.observation_map = observation_map
    self.batch_observation_map = vmap(observation_map)
    self.batch_batch_observation_map = vmap(vmap(observation_map))
    self.jacrev_vmap = vmap(jacrev(lambda x, t, timestep: self.estimate_h_x_0_vmap(x, t, timestep)[0]))
    self.axes = (0,) + tuple(range(len(shape) + 2)[2:]) + (1,)
    self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, (epsilon, _) = self.estimate_h_x_0_vmap(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
    grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
    H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
    C_yy = ratio * H_grad_H_x_0 + self.noise_std**2 * jnp.eye(self.num_y)
    f = jnp.linalg.solve(C_yy, y - h_x_0)
    ls = grad_H_x_0.transpose(self.axes_vmap) @ f
    return epsilon.squeeze(axis=0), ls

  def posterior(self, x, t):
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
    m = self.sqrt_alphas_cumprod[timestep]
    sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
    v = sqrt_1m_alpha**2
    ratio = v / m
    alpha = m**2
    epsilon, ls = self.batch_analysis_vmap(self.y, x, t, timestep, ratio)
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
  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (epsilon, _) = vjp(
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    C_yy = ratio * self.observation_map(vjp_h_x_0(self.observation_map(jnp.ones_like(x)))[0]) + self.noise_std**2
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return epsilon.squeeze(axis=0), ls


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
    self.num_y = y.shape[1]
    self.estimate_h_x_0 = self.get_estimate_x_0(observation_map)
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
    self.batch_analysis_vmap = vmap(self.analysis)
    self.observation_map = observation_map
    self.batch_observation_map  = vmap(observation_map)
    self.batch_batch_observation_map  = vmap(vmap(observation_map))
    self.jacrev_vmap = vmap(jacrev(lambda x, t, timestep: self.estimate_h_x_0_vmap(x, t, timestep)[0]))
    self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)
    self.axes = (0,) + tuple(range(len(shape) + 2)[2:]) + (1,)

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, (epsilon, _) = self.estimate_h_x_0_vmap(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
    grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
    H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
    C_yy = ratio * H_grad_H_x_0 + self.noise_std**2 * jnp.eye(self.num_y)
    f = jnp.linalg.solve(C_yy, y - h_x_0)
    ls = grad_H_x_0.transpose(self.axes_vmap) @ f
    return epsilon.squeeze(axis=0), ls

  def posterior(self, x, t):
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
    sigma = self.discrete_sigmas[timestep]
    sigma_prev = self.discrete_sigmas_prev[timestep]
    epsilon, ls = self.batch_analysis_vmap(self.y, x, t, timestep, sigma**2)
    coeff1 = self.eta * jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
    coeff2 = jnp.sqrt(sigma_prev**2  - coeff1**2)
    std = coeff1
    posterior_score = - batch_mul(1. / sigma, epsilon) + ls
    x_mean = x + batch_mul(sigma * (sigma - coeff2), posterior_score)
    return x_mean, std


class KGDMVEplus(KGDMVE):
  """KGDMVE with a mask."""
  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (epsilon, _) = vjp(
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    C_yy = ratio * self.observation_map(vjp_h_x_0(self.observation_map(jnp.ones_like(x)))[0]) + self.noise_std**2
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return epsilon.squeeze(axis=0), ls


class PiGDMVP(DDIMVP):
  """PiGDM Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""
  def __init__(self, y, observation_map, noise_std, shape, model, data_variance=1.0, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.):
    super().__init__(model, eta, num_steps, dt, epsilon, beta_min, beta_max)
    self.estimate_h_x_0 = self.get_estimate_x_0(observation_map)
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
    self.batch_analysis_vmap = vmap(self.analysis)
    self.y = y
    self.noise_std = noise_std
    self.data_variance = data_variance
    self.num_y = y.shape[1]
    self.observation_map = observation_map
    self.batch_observation_map = vmap(observation_map)
    self.jacrev_vmap = vmap(jacrev(lambda x, t, timestep: self.estimate_h_x_0_vmap(x, t, timestep)[0]))
    self.axes = (0,) + tuple(range(len(shape) + 4)[2:]) + (1,)
    self.axes_vmap = tuple(range(len(shape) + 3)[1:]) + (0,)

  def analysis(self, y, x, t, timestep, v, alpha):
    h_x_0, vjp_h_x_0, (epsilon, _) = vjp(
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    # Value suggested for VPSDE in original PiGDM paper
    r = v * self.data_variance  / (v + alpha * self.data_variance)
    C_yy = 1. + self.noise_std**2 / r
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return epsilon.squeeze(axis=0), ls

  def posterior(self, x, t):
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
    m = self.sqrt_alphas_cumprod[timestep]
    sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
    v = sqrt_1m_alpha**2
    alpha = m**2
    epsilon, ls = self.batch_analysis_vmap(self.y, x, t, timestep, v, alpha)
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
  def analysis(self, y, x, t, timestep, v, alpha):
    h_x_0, vjp_estimate_h_x_0, (epsilon, _) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    # Value suggested for VPSDE in original PiGDM paper
    r = v * self.data_variance  / (v + alpha * self.data_variance)
    C_yy = r + self.noise_std**2
    ls = vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
    return epsilon.squeeze(axis=0), ls


class SSPiGDMVP(DDIMVP):
  """
  Note: We found this method to be unstable on all datasets.
  PiGDM Song et al. 2023. Markov chain using the DDIM Markov Chain or VP SDE."""
  def __init__(self, y, observation_map, noise_std, shape, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.):
    super().__init__(model, eta, num_steps, dt, epsilon, beta_min, beta_max)
    self.estimate_h_x_0 = self.get_estimate_x_0_vmap(observation_map)
    self.batch_analysis_vmap = vmap(self.analysis)
    self.y = y
    self.noise_std = noise_std
    self.num_y = y.shape[1]

  def analysis(self, y, x, t, timestep, v, alpha):
    h_x_0, vjp_estimate_h_x_0, (epsilon, x_0) = vjp(
      lambda x: self.estimate_h_x_0(x, t, timestep), x, has_aux=True)
    # Value suggested for VPSDE in original PiGDM paper
    r = v * self.data_variance  / (v + alpha * self.data_variance)
    C_yy = r + self.noise_std**2
    ls = r * vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0), ls, epsilon.squeeze(axis=0)

  def posterior(self, x, t):
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
    m = self.sqrt_alphas_cumprod[timestep]
    sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
    v = sqrt_1m_alpha**2
    alpha = m**2
    x_0, ls, epsilon = self.batch_analysis_vmap(x, t, timestep, v, alpha)
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
    self.num_y = y.shape[1]
    self.estimate_h_x_0 = self.get_estimate_x_0(observation_map)
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
    self.batch_analysis_vmap = vmap(self.analysis)
    self.batch_observation_map = vmap(observation_map)
    self.jacrev_vmap = vmap(jacrev(lambda x, t, timestep: self.estimate_h_x_0_vmap(x, t, timestep)[0]))
    self.axes = (0,) + tuple(range(len(shape) + 2)[2:]) + (1,)
    self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

  def analysis(self, y, x, t, timestep, v):
    h_x_0, vjp_h_x_0, (epsilon, x_0) = vjp(
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    r = v * self.data_variance / (v + self.data_variance)
    C_yy = 1. + self.noise_std**2 / r
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0), ls, epsilon.squeeze(axis=0)

  def posterior(self, x, t):
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
    sigma = self.discrete_sigmas[timestep]
    sigma_prev = self.discrete_sigmas_prev[timestep]
    x_0, ls, epsilon = self.batch_analysis_vmap(self.y, x, t, timestep, sigma**2)
    coeff1 = self.eta * jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
    coeff2 = jnp.sqrt(sigma_prev**2  - coeff1**2)
    x_mean = x_0 + batch_mul(coeff2, epsilon) + ls
    std = coeff1
    return x_mean, std


class PiGDMVEplus(PiGDMVE):
  """KGDMVE with a mask."""
  def analysis(self, y, x, t, timestep, v):
    h_x_0, vjp_h_x_0, (epsilon, x_0) = vjp(
      lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    # Value suggested for VPSDE in original PiGDM paper
    r = v * self.data_variance  / (v + self.data_variance)
    C_yy = 1. + self.noise_std**2 / r
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0), ls, epsilon.squeeze(axis=0)


class DPSSMLD(SMLD):
  """DPS for SMLD ancestral sampling."""
  def __init__(self, scale, y, observation_map, score, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
    super().__init__(score, num_steps, dt, epsilon, sigma_min, sigma_max)
    self.y = y
    self.scale = scale
    self.likelihood_score = self.get_likelihood_score(self.get_estimate_x_0(observation_map))
    self.likelihood_score_vmap = self.get_likelihood_score_vmap(self.get_estimate_x_0_vmap(observation_map))

  def get_likelihood_score_vmap(self, estimate_h_x_0_vmap):
    def l2_norm(x, t, timestep, y):
      h_x_0, (s, _) = estimate_h_x_0_vmap(x, t, timestep)
      norm = jnp.linalg.norm(y - h_x_0)
      return norm, s.squeeze(axis=0)
    grad_l2_norm = grad(l2_norm, has_aux=True)
    return vmap(grad_l2_norm)

  def get_likelihood_score(self, estimate_h_x_0):
    batch_norm = vmap(jnp.linalg.norm)
    def l2_norm(x, t, timestep, y):
      h_x_0, (s, _) = estimate_h_x_0(x, t, timestep)
      norm = batch_norm(y - h_x_0)
      norm = jnp.sum(norm)
      return norm, s
    grad_l2_norm = grad(l2_norm, has_aux=True)
    return grad_l2_norm

  def update(self, rng, x, t):
    """Return the update of the state and any auxilliary variables."""
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
    # ls, score = self.likelihood_score(x, t, timestep, self.y)
    ls, score = self.likelihood_score_vmap(x, t, timestep, self.y)
    x_mean, std = self.posterior(score, x, timestep)
    x_mean -= self.scale * ls  # DPS
    z = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, z)
    return x, x_mean


DPSSMLDplus = DPSSMLD


class DPSDDPM(DDPM):
  """DPS for DDPM ancestral sampling."""
  def __init__(self, scale, y, observation_map, score, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.0):
    super().__init__(score, num_steps, dt, epsilon, beta_min, beta_max)
    self.y = y
    self.scale = scale
    self.likelihood_score = self.get_likelihood_score(self.get_estimate_x_0(observation_map))
    self.likelihood_score_vmap = self.get_likelihood_score_vmap(self.get_estimate_x_0_vmap(observation_map))

  def get_likelihood_score_vmap(self, estimate_h_x_0_vmap):
    def l2_norm(x, t, timestep, y):
      h_x_0, (s, _) = estimate_h_x_0_vmap(x, t, timestep)
      norm = jnp.linalg.norm(y - h_x_0)
      return norm, s.squeeze(axis=0)
    grad_l2_norm = grad(l2_norm, has_aux=True)
    return vmap(grad_l2_norm)

  def get_likelihood_score(self, estimate_h_x_0):
    batch_norm = vmap(jnp.linalg.norm)
    def l2_norm(x, t, timestep, y):
      h_x_0, (s, _) = estimate_h_x_0(x, t, timestep)
      norm = batch_norm(y - h_x_0)
      norm = jnp.sum(norm)
      return norm, s
    grad_l2_norm = grad(l2_norm, has_aux=True)
    return grad_l2_norm

  def update(self, rng, x, t):
    """Return the update of the state and any auxilliary variables."""
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
    # ls, score = self.likelihood_score(x, t, timestep, self.y)
    ls, score = self.likelihood_score_vmap(x, t, timestep, self.y)
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
    self.num_y = y.shape[1]
    self.estimate_h_x_0 = self.get_estimate_x_0(observation_map)
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
    self.batch_analysis_vmap = vmap(self.analysis)
    self.observation_map = observation_map
    self.batch_observation_map = vmap(observation_map)
    self.batch_batch_observation_map = vmap(vmap(observation_map))
    self.jacrev_vmap = vmap(jacrev(lambda x, t, timestep: self.estimate_h_x_0_vmap(x, t, timestep)[0]))
    self.axes = (0,) + tuple(range(len(shape) + 2)[2:]) + (1,)
    self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, (_, x_0) = self.estimate_h_x_0_vmap(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
    grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
    H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
    C_yy = H_grad_H_x_0 + self.noise_std**2 / ratio * jnp.eye(self.num_y)
    f = jnp.linalg.solve(C_yy, y - h_x_0)
    ls = grad_H_x_0.transpose(self.axes_vmap) @ f
    return x_0.squeeze(axis=0) + ls

  def posterior(self, x, t):
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
    beta = self.discrete_betas[timestep]
    m = self.sqrt_alphas_cumprod[timestep]
    v = self.sqrt_1m_alphas_cumprod[timestep]**2
    ratio = v / m
    x_dash = self.batch_analysis_vmap(self.y, x, t, timestep, ratio)
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
    def analysis(self, y, x, t, timestep, ratio):
      h_x_0, vjp_estimate_h_x_0, (_, x_0) = vjp(
          lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
      C_yy = self.observation_map(vjp_estimate_h_x_0(self.observation_map(jnp.ones_like(x)))[0]) + self.noise_std**2 / ratio
      ls = vjp_estimate_h_x_0((y - h_x_0) / C_yy)[0]
      return x_0.squeeze(axis=0) + ls


class KPSMLD(SMLD):
  """Kalman posterior for SMLD Ancestral sampling."""
  def __init__(self, y, observation_map, noise_std, shape, score, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
    super().__init__(score, num_steps, dt, epsilon, sigma_min, sigma_max)
    self.y = y
    self.noise_std = noise_std
    self.num_y = y.shape[1]
    self.estimate_h_x_0 = self.get_estimate_x_0(observation_map)
    self.estimate_h_x_0_vmap = self.get_estimate_x_0_vmap(observation_map)
    self.batch_analysis_vmap = vmap(self.analysis)
    self.observation_map = observation_map
    self.batch_observation_map = vmap(observation_map)
    self.batch_batch_observation_map = vmap(vmap(observation_map))
    self.jacrev_vmap = vmap(jacrev(lambda x, t, timestep: self.estimate_h_x_0_vmap(x, t, timestep)[0]))
    # axes tuple for correct permutation of grad_H_x_0 array
    self.axes = (0,) + tuple(range(len(shape) + 2)[2:]) + (1,)
    self.axes_vmap = tuple(range(len(shape) + 1)[1:]) + (0,)

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, (_, x_0) = self.estimate_h_x_0_vmap(x, t, timestep)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
    grad_H_x_0 = jacrev(lambda _x: self.estimate_h_x_0_vmap(_x, t, timestep)[0])(x)
    H_grad_H_x_0 = self.batch_observation_map(grad_H_x_0)
    C_yy = H_grad_H_x_0 + self.noise_std**2 / ratio * jnp.eye(self.num_y)
    f = jnp.linalg.solve(C_yy, y - h_x_0)
    ls = grad_H_x_0.transpose(self.axes_vmap) @ f
    return x_0.squeeze(axis=0) + ls

  def posterior(self, x, t):
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
    sigma = self.discrete_sigmas[timestep]
    sigma_prev = self.discrete_sigmas_prev[timestep]
    x_0 = self.batch_analysis_vmap(self.y, x, t, timestep, sigma**2)
    x_mean = batch_mul(sigma_prev**2 / sigma**2, x) + batch_mul(1 - sigma_prev**2 / sigma**2, x_0)
    std = jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
    return x_mean, std, x_0

  def update(self, rng, x, t):
    """Return the update of the state and any auxilliary variables."""
    x_mean, std, x_0 = self.posterior(x, t)
    z = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, z)
    return x, x_0


class KPSMLDplus(KPSMLD):
  """Kalman posterior for SMLD Ancestral sampling."""
  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (_, x_0) = vjp(
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    C_yy = self.observation_map(vjp_h_x_0(self.observation_map(jnp.ones_like(x)))[0]) + self.noise_std**2 / ratio
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0) + ls


class KPSMLDdiag(KPSMLD):
  """Kalman posterior for SMLD Ancestral sampling."""

  # def get_grad_estimate_x_0_vmap(self, observation_map):
  #   """https://stackoverflow.com/questions/70956578/jacobian-diagonal-computation-in-jax"""

  #   def estimate_x_0_single(val, i, x, t, timestep):
  #     x_shape = x.shape
  #     x = x.flatten()
  #     x = x.at[i].set(val)
  #     x = x.reshape(x_shape)
  #     x = jnp.expand_dims(x, axis=0)
  #     t = jnp.expand_dims(t, axis=0)
  #     v = self.discrete_sigmas[timestep]**2
  #     s = self.score(x, t)
  #     x_0 = x + v * s
  #     h_x_0 = observation_map(x_0)
  #     return h_x_0[i]
  #   return vmap(value_and_grad(estimate_x_0_single), in_axes=(0, 0, None, None, None))

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (_, x_0) = vjp(
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)

    # There is no natural way to do this with JAX's transforms:
    # you cannot map the input, because in general each diagonal entry of the jacobian depends on all inputs.
    # This seems like the best method, but it is too slow for numerical evaluation, and batch size cannot be large (max size tested was one)
    vec_vjp_h_x_0 = vmap(vjp_h_x_0)
    diag = jnp.diag(self.batch_observation_map(vec_vjp_h_x_0(jnp.eye(y.shape[0]))[0]))

    # # This method gives OOM error
    # idx = jnp.arange(len(y))
    # h_x_0, diag = self.grad_estimate_x_0_vmap(x.flatten(), idx, x, t, timestep)
    # diag = self.observation_map(diag)

    # # This method can't be XLA compiled and is way too slow for numerical evaluation
    # diag = jnp.empty(y.shape[0])
    # for i in range(y.shape[0]):
    #   eye = jnp.zeros(y.shape[0])
    #   diag_i = jnp.dot(self.observation_map(vjp_h_x_0(eye)[0]), eye)
    #   diag = diag.at[i].set(diag_i)

    C_yy = diag + self.noise_std**2 / ratio
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0) + ls


class KPDDPMdiag(KPDDPM):
  """Kalman posterior for DDPM Ancestral sampling."""

  def analysis(self, y, x, t, timestep, ratio):
    h_x_0, vjp_h_x_0, (_, x_0) = vjp(
        lambda x: self.estimate_h_x_0_vmap(x, t, timestep), x, has_aux=True)
    vec_vjp_h_x_0 = vmap(vjp_h_x_0)
    diag = jnp.diag(self.batch_observation_map(vec_vjp_h_x_0(jnp.eye(y.shape[0]))[0]))
    C_yy = diag + self.noise_std**2 / ratio
    ls = vjp_h_x_0((y - h_x_0) / C_yy)[0]
    return x_0.squeeze(axis=0) + ls
