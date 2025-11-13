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

# Hyperparameters for the conjugate Gamma priors
a0 = 1.0
b0 = 1.0
c0 = 1.0
d0 = 1.0

max_iter = 500
rtol = 1e-6

# Posterior shape parameters remain fixed throughout the iterations
a_N = a0 + M / 2.0
c_N = c0 + N / 2.0

# Initialise the variational expectations
E_alpha = a0 / b0
E_beta = c0 / d0
b_N = b0
d_N = d0

X_T = X.T
XTX = X_T @ X
XTy = X_T @ y

identity = np.eye(M)

for _ in range(max_iter):
    precision = E_beta * XTX + E_alpha * identity
    try:
        S_N = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        S_N = np.linalg.pinv(precision)

    m_N = E_beta * S_N @ XTy

    E_w_sq = float((m_N.T @ m_N).item()) + float(np.trace(S_N))
    b_N_new = b0 + 0.5 * E_w_sq
    E_alpha_new = a_N / b_N_new

    residual = y - X @ m_N
    resid_sq = float((residual.T @ residual).item())
    trace_XSX = float(np.trace(X @ S_N @ X_T))
    E_resid_sq = resid_sq + trace_XSX

    d_N_new = d0 + 0.5 * E_resid_sq
    E_beta_new = c_N / d_N_new

    if np.allclose([E_alpha, E_beta], [E_alpha_new, E_beta_new], rtol=rtol, atol=0.0):
        E_alpha, E_beta = E_alpha_new, E_beta_new
        b_N, d_N = b_N_new, d_N_new
        break

    E_alpha, E_beta = E_alpha_new, E_beta_new
    b_N, d_N = b_N_new, d_N_new
else:
    # Ensure the most recent updates are available if the loop didn't break
    b_N, d_N = b_N_new, d_N_new
    S_N = np.linalg.pinv(E_beta * XTX + E_alpha * identity)
    m_N = E_beta * S_N @ XTy

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

