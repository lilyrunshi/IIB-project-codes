import numpy as np

# inputs: y_true, y_pred
diff = y_true - y_pred

# Express the root mean squared error as a percentage of the signal magnitude.
# Scale each residual by the larger absolute value between the truth and the
# prediction to avoid division-by-zero while keeping the percentage intuitive.
scale = np.maximum(np.maximum(np.abs(y_true), np.abs(y_pred)), np.finfo(float).eps)
relative_squared_error = (diff / scale) ** 2

# Return the root mean squared percentage error.
e = float(np.sqrt(np.mean(relative_squared_error)) * 100.0)
