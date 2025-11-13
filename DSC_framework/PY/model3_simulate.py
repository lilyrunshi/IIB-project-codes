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
``a0, b0, c0, d0, e0, f0``
    Hyperparameters for the Gamma/Beta priors described in the model. In the
    accompanying ``main.dsc`` configuration these are set to (3.0, 3.0, 5.0,
    0.4, 8.0, 2.0) to produce pronounced rhythmic structure with light
    observational noise. The sparsity probability ``π`` is sampled from the
    Beta prior ``Beta(e0, f0)`` and governs whether the sinusoidal group is
    active.

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



# Sample global latent variables from their respective priors.
alpha = rng.gamma(shape=a0, scale=1.0 / b0)
beta = rng.gamma(shape=c0, scale=1.0 / d0)
pi = rng.beta(e0, f0)

if c0 <= 1.0:
    raise ValueError(
        "The shape parameter c0 must be greater than 1 for E[β^{-1}] to exist."
    )

# Compute the observation noise scale as the expectation of β^{-1} when
# β ~ Gamma(c0, d0). Under the (shape, rate) parameterisation this equals
# d0 / (c0 - 1).
expected_inv_beta = d0 / (c0 - 1.0)
noise_std = expected_inv_beta

# Bernoulli switch shared by the sinusoidal features; the intercept remains on.
omega_switch = rng.binomial(1, pi)

# Draw regression weights for the sinusoidal basis and apply the switch.
omega = rng.normal(loc=0.0, scale=np.sqrt(1.0 / alpha), size=d)
gamma_vector = np.array([omega_switch, omega_switch, 1.0])

# Generate sampling times and the corresponding design matrix.
time_points = rng.uniform(low=0.0, high=1.0, size=n)
x = _build_design_matrix(time_points)
signal = x @ (gamma_vector * omega)
noise = rng.normal(loc=0.0, scale=noise_std, size=n)
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
    "expected_inv_beta": float(expected_inv_beta),
    "noise_std": float(noise_std),
    "pi": float(pi),
    "omega_switch": int(omega_switch),
    "weights": w_true.tolist(),
    "time": time_points.tolist(),
}
