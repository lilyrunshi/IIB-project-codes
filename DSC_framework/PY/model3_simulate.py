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

On completion the script exposes two NumPy arrays named ``x`` and ``y`` that are
consumed by downstream DSC stages.
"""

from __future__ import annotations

import numpy as np

# Create a random number generator that can be seeded from the DSC configuration.
try:
    rng = np.random.default_rng(seed)  # type: ignore[name-defined]
except NameError:
    rng = np.random.default_rng()

# Retrieve hyperparameters if they were supplied, otherwise fall back to the
# defaults used by the analysis implementation.
try:
    a0  # type: ignore[name-defined]
except NameError:
    a0 = 1.0

try:
    b0  # type: ignore[name-defined]
except NameError:
    b0 = 1.0

try:
    c0  # type: ignore[name-defined]
except NameError:
    c0 = 1.0

try:
    d0  # type: ignore[name-defined]
except NameError:
    d0 = 1.0

try:
    e0  # type: ignore[name-defined]
except NameError:
    e0 = 1.0

try:
    f0  # type: ignore[name-defined]
except NameError:
    f0 = 1.0

# Sample global latent variables from their respective priors.
alpha = rng.gamma(shape=a0, scale=1.0 / b0)
beta = rng.gamma(shape=c0, scale=1.0 / d0)
pi = rng.beta(e0, f0)

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
noise = rng.normal(loc=0.0, scale=np.sqrt(1.0 / beta), size=n)
y = signal + noise

# Optionally expose latent quantities for downstream inspection.
latent = {
    "alpha": float(alpha),
    "beta": float(beta),
    "pi": float(pi),
    "gamma": int(gamma_switch),
    "omega": (gamma_vector * omega).tolist(),
}
