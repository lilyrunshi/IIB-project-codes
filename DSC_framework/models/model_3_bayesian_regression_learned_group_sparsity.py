import numpy as np


def _ensure_2d_column(vector):
    arr = np.asarray(vector)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _digamma(x):
    x = float(x)
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += np.log(x) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
    return result


# Inputs provided by DSC: x, y
X = np.asarray(x)
y = _ensure_2d_column(y)

N, M = X.shape

# Hyperparameters
# Prior for the shared weight precision:    alpha ~ Gamma(a0, b0)
# Prior for the observation noise precision: beta  ~ Gamma(c0, d0)
# Prior for the switching probability:       pi    ~ Beta(e0, f0)
a0 = 1.0
b0 = 1.0
c0 = 1.0
d0 = 1.0
e0 = 1.0
f0 = 1.0

max_iter = 200
rtol = 1e-6

# Posterior shape parameters that are fixed by conjugacy.
a_N = a0 + M / 2.0
c_N = c0 + N / 2.0

# Initial expectations for the latent factors.
E_alpha = a0 / b0
E_beta = c0 / d0
p_N = 0.5

e_N = e0 + p_N
f_N = f0 + (1.0 - p_N)
pi_mean = e_N / (e_N + f_N)

m_N = np.zeros((M, 1))
S_N = np.eye(M)

yTy = float((y.T @ y).item())

for _ in range(max_iter):
    precision = E_alpha * np.eye(M) + p_N * E_beta * (X.T @ X)
    try:
        S_N = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        S_N = np.linalg.pinv(precision)

    m_N = p_N * E_beta * S_N @ (X.T @ y)

    m_sq = float((m_N.T @ m_N).item())
    trace_S = float(np.trace(S_N))

    b_N_new = b0 + 0.5 * (m_sq + trace_S)
    E_alpha_new = a_N / b_N_new

    residual = y - X @ m_N
    resid_sq = float((residual.T @ residual).item())
    trace_XSX = float(np.trace(X @ S_N @ X.T))

    expected_sq = (1.0 - p_N) * yTy + p_N * (resid_sq + trace_XSX)
    d_N_new = d0 + 0.5 * expected_sq
    E_beta_new = c_N / d_N_new

    log_prior_odds = _digamma(e_N) - _digamma(f_N)
    log_likelihood_ratio = 0.5 * E_beta_new * (yTy - (resid_sq + trace_XSX))
    log_odds = log_prior_odds + log_likelihood_ratio

    if log_odds >= 0:
        exp_term = np.exp(-log_odds)
        p_N_new = 1.0 / (1.0 + exp_term)
    else:
        exp_term = np.exp(log_odds)
        p_N_new = exp_term / (1.0 + exp_term)

    e_N_new = e0 + p_N_new
    f_N_new = f0 + (1.0 - p_N_new)
    pi_mean_new = e_N_new / (e_N_new + f_N_new)

    if np.allclose(
        [E_alpha, E_beta, p_N, pi_mean],
        [E_alpha_new, E_beta_new, p_N_new, pi_mean_new],
        rtol=rtol,
        atol=0.0,
    ):
        E_alpha, E_beta, p_N, pi_mean = E_alpha_new, E_beta_new, p_N_new, pi_mean_new
        b_N, d_N = b_N_new, d_N_new
        e_N, f_N = e_N_new, f_N_new
        break

    E_alpha, E_beta, p_N, pi_mean = E_alpha_new, E_beta_new, p_N_new, pi_mean_new
    b_N, d_N = b_N_new, d_N_new
    e_N, f_N = e_N_new, f_N_new
else:
    b_N, d_N = b_N_new, d_N_new

# Posterior predictive mean under the mean-field approximation.
y_hat = (p_N * (X @ m_N)).ravel()

fit = {
    "m_N": m_N.ravel().tolist(),
    "S_N": S_N.tolist(),
    "a_N": float(a_N),
    "b_N": float(b_N),
    "c_N": float(c_N),
    "d_N": float(d_N),
    "E_alpha": float(E_alpha),
    "E_beta": float(E_beta),
    "p_N": float(p_N),
    "e_N": float(e_N),
    "f_N": float(f_N),
    "pi_mean": float(pi_mean),
}
