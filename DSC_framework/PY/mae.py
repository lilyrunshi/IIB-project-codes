import numpy as np

# inputs: y_true, y_pred
diff = y_true - y_pred

# Express the mean absolute error as a percentage of the signal magnitude.
# We scale by the larger absolute value between the truth and the prediction
# on each observation. This keeps the percentage well-defined even when the
# truth crosses zero while still reflecting large discrepancies.
scale = np.maximum(np.maximum(np.abs(y_true), np.abs(y_pred)), np.finfo(float).eps)
relative_error = np.abs(diff) / scale

# Return the mean absolute percentage error.
e = float(np.mean(relative_error) * 100.0)
