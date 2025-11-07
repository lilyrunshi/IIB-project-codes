import numpy as np

# inputs: y_true, y_pred
diff = y_true - y_pred

# Mirror the MAE percentage calculation by using the SMAPE denominator.  This
# keeps the metric bounded even when the true signal approaches zero.
scale = (np.abs(y_true) + np.abs(y_pred)) / 2.0
valid = scale > np.finfo(float).eps

relative_squared_error = np.zeros_like(diff, dtype=float)
relative_squared_error[valid] = (diff[valid] / scale[valid]) ** 2

# Return the root mean squared percentage error
e = float(np.sqrt(np.mean(relative_squared_error)) * 100.0)
