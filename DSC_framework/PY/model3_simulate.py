"""Simulation script for Model 3: Bayesian regression with learned group sparsity.

This module follows the generative process described in Model 3 where the
regression coefficients are modulated by a latent Bernoulli switch that is
shared across all but the first feature. The precision (inverse variance) of the
coefficients and observation noise are both given Gamma priors, while the switch
probability follows a Beta prior.

The DSC framework loads this script as a ``simulate`` module. When executed it
expects the following variables to be defined in the global namespace:

``n``
    Number of observations to generate.
``d``
    Number of covariates (features) per observation.
``seed`` (optional)
    Seed for the random number generator to ensure reproducibility.
``a0, b0, c0, d0, e0, f0`` (all optional)
    Hyperparameters for the Gamma/Beta priors described in the model. Each
    defaults to 1.0, matching the values used by the corresponding analysis
    module.
``noise_std`` (optional)
    Observation noise standard deviation. When provided the noise precision is
    not sampled from its Gamma prior and the supplied value is used instead.
``sparsity_prob`` (optional)
    Probability that the shared Bernoulli switch is ``1`` (i.e. most features
    are active). When omitted the probability is sampled from the Beta prior
    defined by ``e0`` and ``f0``.

On completion the script exposes two NumPy arrays named ``x`` and ``y`` that are
consumed by downstream DSC stages.
"""
import numpy as np


# Create a random number generator that can be seeded from the DSC configuration.
seed = globals().get("seed", None)
rng = np.random.default_rng(seed)

# Retrieve hyperparameters if they were supplied, otherwise fall back to the
# defaults used by the analysis implementation.
a0 = float(globals().get("a0", 1.0))
b0 = float(globals().get("b0", 1.0))
c0 = float(globals().get("c0", 1.0))
d0 = float(globals().get("d0", 1.0))
e0 = float(globals().get("e0", 1.0))
f0 = float(globals().get("f0", 1.0))

# Determine whether an explicit observation noise standard deviation was
# provided by the DSC configuration. When specified we skip sampling ``beta``
# from its Gamma prior and instead use the fixed value so that downstream
# modules know the actual noise level used for this replicate.
noise_std_value = globals().get("noise_std", None)
noise_std = None if noise_std_value is None else float(noise_std_value)

sparsity_prob_value = globals().get("sparsity_prob", None)
if sparsity_prob_value is not None:
    pi_override = float(sparsity_prob_value)
    if not 0.0 <= pi_override <= 1.0:
        raise ValueError("sparsity_prob must be between 0 and 1 inclusive")
else:
    pi_override = None

# Sample global latent variables from their respective priors.
alpha = rng.gamma(shape=a0, scale=1.0 / b0)
if noise_std is None:
    beta = rng.gamma(shape=c0, scale=1.0 / d0)
    noise_scale = np.sqrt(1.0 / beta)
else:
    noise_scale = noise_std
    beta = 1.0 / (noise_scale ** 2)

if pi_override is None:
    pi = rng.beta(e0, f0)
else:
    pi = pi_override

# Bernoulli switch shared by every feature except the first.
gamma_switch = rng.binomial(1, pi)

# Draw regression weights and apply the switch.
omega = rng.normal(loc=0.0, scale=np.sqrt(1.0 / alpha), size=d)
gamma_vector = np.ones(d)
if d > 1:
    gamma_vector[1:] = gamma_switch

# Generate the design matrix and corresponding responses.
x = rng.normal(loc=0.0, scale=1.0, size=(n, d))
signal = x @ (gamma_vector * omega)
noise = rng.normal(loc=0.0, scale=noise_scale, size=n)
y = signal + noise

# Optionally expose latent quantities for downstream inspection.
latent = {
    "alpha": float(alpha),
    "beta": float(beta),
    "pi": float(pi),
    "gamma": int(gamma_switch),
    "omega": (gamma_vector * omega).tolist(),
}
