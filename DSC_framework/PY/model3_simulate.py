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
    Hyperparameters for the Gamma/Beta priors described in the model. These are
    supplied explicitly in ``main.dsc`` (values tuned to 1.5, 0.75, 3.0, 1.0,
    4.0, 1.0) to emphasise rhythmic features in the simulated data while
    remaining compatible with the analysis module.
``noise_std``
    Observation noise standard deviation. When provided the noise precision is
    not sampled from its Gamma prior and the supplied value is used instead.
``sparsity_prob``
    Probability that the shared Bernoulli switch is ``1`` (i.e. most features
    are active). When omitted the probability is sampled from the Beta prior
    defined by ``e0`` and ``f0``.

On completion the script exposes two NumPy arrays named ``x`` and ``y`` that are
consumed by downstream DSC stages along with ``w_true`` containing the
regression coefficients used to synthesise the observations.
"""
import numpy as np
from matplotlib import pyplot as plt


# Create a random number generator that can be seeded from the DSC configuration.
seed = globals().get("seed", None)
rng = np.random.default_rng(seed)

# Retrieve hyperparameters configured in ``main.dsc``. They are required for the
# simulation to ensure consistent priors across replicates. The DSC runtime
# injects them directly into the module namespace so we can read them without
# additional guards.
a0 = float(globals()["a0"])
b0 = float(globals()["b0"])
c0 = float(globals()["c0"])
d0 = float(globals()["d0"])
e0 = float(globals()["e0"])
f0 = float(globals()["f0"])


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

# Bernoulli switch shared by the sinusoidal features; the intercept remains on.
gamma_switch = rng.binomial(1, pi)

# Enforce the three-feature structure required for the sinusoidal basis.
if d != 3:
    raise ValueError(
        "Model 3 simulation expects d == 3 (sin, cos, intercept features)."
    )

# Draw regression weights for the sinusoidal basis and apply the switch.
omega = rng.normal(loc=0.0, scale=np.sqrt(1.0 / alpha), size=d)
gamma_vector = np.array([gamma_switch, gamma_switch, 1.0])

# Generate sampling times and the corresponding design matrix.
time_points = rng.uniform(low=0.0, high=1.0, size=n)
x = _build_design_matrix(time_points)
signal = x @ (gamma_vector * omega)
noise = rng.normal(loc=0.0, scale=noise_scale, size=n)
y = signal + noise

plot_requested = bool(globals().get("plot_data", False))
plot_filename = globals().get("plot_filename", None)
if plot_requested or plot_filename is not None:
    plot_generated_data(time_points, y, signal=signal, filename=plot_filename)

# Expose the coefficient vector used to generate the data so downstream
# analysis stages and scoring modules can compare posterior expectations with
# the ground truth weights.
w_true = gamma_vector * omega

# Optionally expose latent quantities for downstream inspection.
latent = {
    "alpha": float(alpha),
    "beta": float(beta),
    "pi": float(pi),
    "gamma": int(gamma_switch),
    "omega": w_true.tolist(),
    "time": time_points.tolist(),
}
