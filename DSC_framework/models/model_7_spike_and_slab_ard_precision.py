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

a0 = 1.0  # shape for each ARD precision (alpha_m)
b0 = 1.0  # rate for each ARD precision
c0 = 1.0  # shape for the observation precision (beta)
d0 = 1.0  # rate for the observation precision
e0 = 1.0  # shape for the Bernoulli probability (pi)
f0 = 1.0  # rate for the Bernoulli probability

max_iter = 200
rtol = 1e-6

a_N = a0 + 0.5
c_N = c0 + N / 2.0

p_N = 0.5

m_N = np.zeros((M, 1))
S_N = np.eye(M)

yTy = float((y.T @ y).item())

diag_S = np.diag(S_N)
E_w2_active = m_N.ravel() ** 2 + diag_S

b_N_vec = b0 + 0.5 * p_N * E_w2_active
E_alpha = np.full(M, a_N / b_N_vec)
E_log_alpha = np.full(M, _digamma(a_N) - np.log(b_N_vec))

d_N = d0 + 0.5 * yTy
E_beta = c_N / d_N
E_log_beta = _digamma(c_N) - np.log(d_N)

e_N = e0 + p_N
f_N = f0 + (1.0 - p_N)
pi_mean = e_N / (e_N + f_N)

for _ in range(max_iter):
    precision = np.diag(E_alpha) + E_beta * (X.T @ X)
    try:
        S_N = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        S_N = np.linalg.pinv(precision)

    m_N = E_beta * S_N @ (X.T @ y)

    diag_S = np.diag(S_N)
    E_w2_active = m_N.ravel() ** 2 + diag_S

    b_N_vec_new = b0 + 0.5 * p_N * E_w2_active
    E_alpha_new = a_N / b_N_vec_new
    E_log_alpha_new = _digamma(a_N) - np.log(b_N_vec_new)

    residual = y - X @ m_N
    resid_sq = float((residual.T @ residual).item())
    trace_term = float(np.trace(X @ S_N @ X.T))
    expected_sq_active = resid_sq + trace_term

    d_N_new = d0 + 0.5 * ((1.0 - p_N) * yTy + p_N * expected_sq_active)
    E_beta_new = c_N / d_N_new
    E_log_beta_new = _digamma(c_N) - np.log(d_N_new)

    log_prior_odds = _digamma(e_N) - _digamma(f_N)
    log_likelihood_odds = -0.5 * E_beta_new * (expected_sq_active - yTy)
    log_precision_odds = 0.5 * np.sum(E_log_alpha_new) - 0.5 * np.sum(E_alpha_new * E_w2_active)
    log_odds = log_prior_odds + log_likelihood_odds + log_precision_odds

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
        [p_N, E_beta, pi_mean],
        [p_N_new, E_beta_new, pi_mean_new],
        rtol=rtol,
        atol=0.0,
    ) and np.allclose(E_alpha, E_alpha_new, rtol=rtol, atol=0.0):
        p_N = p_N_new
        E_beta = E_beta_new
        E_log_beta = E_log_beta_new
        d_N = d_N_new
        E_alpha = E_alpha_new
        E_log_alpha = E_log_alpha_new
        b_N_vec = b_N_vec_new
        e_N, f_N = e_N_new, f_N_new
        pi_mean = pi_mean_new
        break

    p_N = p_N_new
    E_beta = E_beta_new
    E_log_beta = E_log_beta_new
    d_N = d_N_new
    E_alpha = E_alpha_new
    E_log_alpha = E_log_alpha_new
    b_N_vec = b_N_vec_new
    e_N, f_N = e_N_new, f_N_new
    pi_mean = pi_mean_new
else:
    d_N = d_N_new
    b_N_vec = b_N_vec_new
    e_N, f_N = e_N_new, f_N_new
    pi_mean = pi_mean_new

precision = np.diag(E_alpha) + E_beta * (X.T @ X)
try:
    S_N = np.linalg.inv(precision)
except np.linalg.LinAlgError:
    S_N = np.linalg.pinv(precision)

m_N = E_beta * S_N @ (X.T @ y)

y_hat = (p_N * (X @ m_N)).ravel()

fit = {
    "m_N": m_N.ravel().tolist(),
    "S_N": S_N.tolist(),
    "a_N": float(a_N),
    "b_N": b_N_vec.tolist(),
    "c_N": float(c_N),
    "d_N": float(d_N),
    "E_alpha": E_alpha.tolist(),
    "E_beta": float(E_beta),
    "p_N": float(p_N),
    "e_N": float(e_N),
    "f_N": float(f_N),
    "pi_mean": float(pi_mean),
    "E_log_beta": float(E_log_beta),
    "E_log_alpha": E_log_alpha.tolist(),
}
