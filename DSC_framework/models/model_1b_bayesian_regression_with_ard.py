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

# Hyperparameters for the independent Gamma priors on the precisions
a0 = 1.0
b0 = 1.0
c0 = 1.0
d0 = 1.0

max_iter = 500
rtol = 1e-6

a_N = a0 + 0.5
c_N = c0 + N / 2.0

b_N = np.full(M, b0)
d_N = d0

E_alpha = a_N / b_N
E_beta = c_N / d_N

X_T = X.T
XTX = X_T @ X
XTy = X_T @ y

for _ in range(max_iter):
    precision = E_beta * XTX + np.diag(E_alpha)
    try:
        S_N = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        S_N = np.linalg.pinv(precision)

    m_N = E_beta * S_N @ XTy

    m_sq = np.square(m_N.flatten())
    diag_S = np.diag(S_N)
    b_N_new = b0 + 0.5 * (m_sq + diag_S)
    E_alpha_new = a_N / b_N_new

    residual = y - X @ m_N
    resid_sq = float((residual.T @ residual).item())
    trace_XSX = float(np.trace(X @ S_N @ X_T))
    d_N_new = d0 + 0.5 * (resid_sq + trace_XSX)
    E_beta_new = c_N / d_N_new

    if np.allclose(
        np.concatenate([E_alpha, [E_beta]]),
        np.concatenate([E_alpha_new, [E_beta_new]]),
        rtol=rtol,
        atol=0.0,
    ):
        E_alpha = E_alpha_new
        E_beta = E_beta_new
        b_N = b_N_new
        d_N = d_N_new
        break

    E_alpha = E_alpha_new
    E_beta = E_beta_new
    b_N = b_N_new
    d_N = d_N_new
else:
    precision = E_beta * XTX + np.diag(E_alpha)
    S_N = np.linalg.pinv(precision)
    m_N = E_beta * S_N @ XTy

y_hat = (X @ m_N).ravel()

fit = {
    "m_N": m_N.ravel().tolist(),
    "S_N": S_N.tolist(),
    "a_N": float(a_N),
    "b_N": b_N.tolist(),
    "c_N": float(c_N),
    "d_N": float(d_N),
    "E_alpha": E_alpha.tolist(),
    "E_beta": float(E_beta),
    "w_mean": m_N.ravel().tolist(),
    "w_cov": S_N.tolist(),
}
