"""Root mean squared error between posterior mean weights and ground truth."""

import numpy as np


def _extract_w_mean(fit_dict):
    if isinstance(fit_dict, dict):
        if "w_mean" in fit_dict:
            return fit_dict["w_mean"]
        if "m_N" in fit_dict:
            return fit_dict["m_N"]
    raise KeyError("fit dictionary must contain 'w_mean' (or legacy 'm_N')")


# inputs: w_true, fit
w_true_arr = np.asarray(w_true, dtype=float)
w_mean_arr = np.asarray(_extract_w_mean(fit), dtype=float)

if w_mean_arr.ndim > 1:
    w_mean_arr = w_mean_arr.ravel()

diff = w_true_arr - w_mean_arr
e = float(np.sqrt(np.mean(diff ** 2)))
