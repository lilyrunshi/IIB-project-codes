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
``seed``
    Seed for the random number generator to ensure reproducibility.
``noise_std``
    Target standard deviation of the observation noise. The script selects
    Gamma hyperparameters ``(c0, d0)`` so that the prior expectation of the
    observation variance matches this value.
``sparsity_prob``
    Desired mean activation probability for the sinusoidal group. The script
    converts this into Beta hyperparameters ``(e0, f0)`` for the Bernoulli
    switch that toggles the sinusoidal coefficients.

The observation noise variance is governed by the latent precision ``β`` which
follows a Gamma prior with hyperparameters ``c0`` (shape) and ``d0`` (rate). The
standard deviation used to draw the observation noise equals the expectation of
``β^{-1}`` under this Gamma distribution (requiring ``c0 > 1``), ensuring the
variance is tied to the prior rather than manually specified.

On completion the script exposes two NumPy arrays named ``x`` and ``y`` that are
consumed by downstream DSC stages along with ``w_true`` containing the
regression coefficients used to synthesise the observations.
"""
import numpy as np
from matplotlib import pyplot as plt


# Create a random number generator that can be seeded from the DSC configuration.
rng = np.random.default_rng(seed)


DEFAULT_ALPHA_SHAPE = 3.0
DEFAULT_ALPHA_RATE = 3.0
NOISE_SHAPE = 5.0
BETA_CONCENTRATION = 10.0


def _gamma_hyperparameters_from_noise_std(target_std, shape=NOISE_SHAPE):
    """Return Gamma(shape, rate) parameters with E[β^{-1}] = target_std.

    Parameters
    ----------
    target_std : float
        Desired standard deviation for the observation noise.
    shape : float, optional
        Shape parameter used to construct the Gamma prior. Must exceed 1.0 so
        that E[β^{-1}] exists.
    """

    if target_std <= 0:
        raise ValueError("noise_std must be strictly positive")
    if shape <= 1.0:
        raise ValueError("shape must exceed 1 to compute E[β^{-1}]")

    rate = (shape - 1.0) / target_std
    return shape, rate


def _beta_hyperparameters_from_probability(probability,
                                           concentration=BETA_CONCENTRATION):
    """Return Beta(alpha, beta) parameters with E[π] = probability."""

    if not 0 <= probability <= 1:
        raise ValueError("sparsity_prob must lie in [0, 1]")
    alpha = probability * concentration
    beta = (1.0 - probability) * concentration
    return alpha, beta

def _build_design_matrix(time_points, frequency=1.0):
    """Construct the sinusoidal design matrix used by Model 3.

    Parameters
    ----------
    time_points : ndarray
        Sampling times in the interval [0, 1).
    frequency : float, optional
        Angular frequency multiplier for the sinusoidal basis, by default 1.0.

    Returns
    -------
    ndarray
        Design matrix with columns ``sin(2πft)``, ``cos(2πft)``, and an
        intercept term.
    """

    angular = 2.0 * np.pi * frequency * time_points
    sin_term = np.sin(angular)
    cos_term = np.cos(angular)
    intercept = np.ones_like(time_points)
    return np.column_stack([sin_term, cos_term, intercept])


def plot_generated_data(time_points, observations, signal=None, filename=None):
    """Plot the generated observations against time.

    Parameters
    ----------
    time_points : ndarray
        Sampling times corresponding to each observation.
    observations : ndarray
        Observed response values including noise.
    signal : ndarray, optional
        Noise-free signal, if available, to overlay on the plot.
    filename : str, optional
        When provided the figure is saved to ``filename`` instead of using the
        default ``model3_generated_data.png``.
    """

    plt.figure(figsize=(8, 4))
    plt.scatter(time_points, observations, color="tab:blue", alpha=0.6,
                label="observations")

    if signal is not None:
        sort_idx = np.argsort(time_points)
        plt.plot(time_points[sort_idx], signal[sort_idx], color="tab:orange",
                 linewidth=2.0, label="signal")

    plt.xlabel("Time")
    plt.ylabel("Response")
    plt.title("Model 3 simulated rhythmic data")
    plt.legend()

    output_path = filename or "model3_generated_data.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



# Derive the prior hyperparameters from the requested noise/sparsity levels.
a0 = DEFAULT_ALPHA_SHAPE
b0 = DEFAULT_ALPHA_RATE
c0, d0 = _gamma_hyperparameters_from_noise_std(noise_std)
e0, f0 = _beta_hyperparameters_from_probability(sparsity_prob)

# Sample global latent variables from their respective priors.
alpha = rng.gamma(shape=a0, scale=1.0 / b0)
beta = rng.gamma(shape=c0, scale=1.0 / d0)
pi = rng.beta(e0, f0)

# Compute the observation noise scale as the expectation of β^{-1} when
# β ~ Gamma(c0, d0). Under the (shape, rate) parameterisation this equals
# d0 / (c0 - 1).
noise_scale = d0 / (c0 - 1.0)

# Bernoulli switch shared by the sinusoidal features; the intercept remains on.
omega_switch = rng.binomial(1, pi)

# Draw regression weights for the sinusoidal basis and apply the switch.
omega = rng.normal(loc=0.0, scale=np.sqrt(1.0 / alpha), size=d)
gamma_vector = np.array([omega_switch, omega_switch, 1.0])

# Generate sampling times and the corresponding design matrix.
time_points = rng.uniform(low=0.0, high=1.0, size=n)
x = _build_design_matrix(time_points)
signal = x @ (gamma_vector * omega)
noise = rng.normal(loc=0.0, scale=noise_scale, size=n)
y = signal + noise

plot_generated_data(time_points, y, signal=signal, filename="model3_generated_data.png")

# Expose the coefficient vector used to generate the data so downstream
# analysis stages and scoring modules can compare posterior expectations with
# the ground truth weights.
w_true = gamma_vector * omega

# Optionally expose latent quantities for downstream inspection.
latent = {
    "alpha": float(alpha),
    "beta": float(beta),
    "noise_std": float(noise_scale),
    "pi": float(pi),
    "omega_switch": int(omega_switch),
    "weights": w_true.tolist(),
    "time": time_points.tolist(),
}
