import numpy as np


def _ensure_2d_column(vector):
    arr = np.asarray(vector)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


# Inputs provided by DSC: x, y
X = np.asarray(x)
y = _ensure_2d_column(y)

N, M = X.shape

# Hyperparameters taken from the Model 4 description
# (shared Gamma priors on the precision parameters).
a0 = 1.0
b0 = 1.0
c0 = 1.0
d0 = 1.0

max_iter = 200
rtol = 1e-6

a_N = a0 + M / 2.0
c_N = c0 + N / 2.0

# Initialise expectations of the precision parameters.
E_alpha = a0 / b0
E_beta = c0 / d0

# Initialise posterior moments of the weight vector.
m_N = np.zeros((M, 1))
S_N = np.eye(M)

for _ in range(max_iter):
    Lambda_N = E_alpha * np.eye(M) + X.T @ X
    try:
        S_N = np.linalg.inv(Lambda_N)
    except np.linalg.LinAlgError:
        S_N = np.linalg.pinv(Lambda_N)
    m_N = S_N @ (X.T @ y)

    m_N_sq = float((m_N.T @ m_N).item())
    trace_S = float(np.trace(S_N))

    # Update q(beta) parameters (Normal-Gamma factor for Model 4).
    residual = y - X @ m_N
    resid_sq = float((residual.T @ residual).item())
    trace_XSX = float(np.trace(X @ S_N @ X.T))

    # Contribution from the prior coupling between alpha and beta.
    coupling_term = E_alpha * (m_N_sq + trace_S)

    d_N_new = d0 + 0.5 * (resid_sq + trace_XSX + coupling_term)
    c_N = c0 + N / 2.0
    E_beta_new = c_N / d_N_new

    # Update q(alpha) parameters using expectations under q(omega, beta).
    b_N_new = b0 + 0.5 * (E_beta_new * m_N_sq + trace_S)
    E_alpha_new = a_N / b_N_new

    if np.allclose([E_alpha, E_beta], [E_alpha_new, E_beta_new], rtol=rtol, atol=0):
        E_alpha, E_beta = E_alpha_new, E_beta_new
        b_N, d_N = b_N_new, d_N_new
        break

    E_alpha, E_beta = E_alpha_new, E_beta_new
    b_N, d_N = b_N_new, d_N_new
else:
    # Store the last updates even if convergence tolerance is not met.
    b_N, d_N = b_N_new, d_N_new

# Posterior predictive mean.
y_hat = (X @ m_N).ravel()

fit = {
    "m_N": m_N.ravel().tolist(),
    "S_N": S_N.tolist(),
    "a_N": float(a_N),
    "b_N": float(b_N),
    "c_N": float(c_N),
    "d_N": float(d_N),
    "E_alpha": float(E_alpha),
    "E_beta": float(E_beta),
    "w_mean": m_N.ravel().tolist(),
    "w_cov": S_N.tolist(),
}

