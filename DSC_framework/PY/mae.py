import numpy as np

# inputs: y_true, y_pred
diff = y_true - y_pred

# Use the symmetric mean absolute percentage error (SMAPE) denominator so that
# values close to zero do not explode the percentage calculation.  When both the
# true and predicted values are zero we treat the relative error as zero.
scale = (np.abs(y_true) + np.abs(y_pred)) / 2.0
valid = scale > np.finfo(float).eps

relative_error = np.zeros_like(diff, dtype=float)
relative_error[valid] = np.abs(diff[valid]) / scale[valid]

# Return the mean absolute percentage error as a 0-200% range quantity.
e = float(np.mean(relative_error) * 100.0)
