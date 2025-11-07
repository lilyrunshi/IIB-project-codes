import numpy as np

# inputs: y_true, y_pred
diff = y_true - y_pred

# Avoid division by zero by flooring the denominator at machine epsilon
denominator = np.maximum(np.abs(y_true), np.finfo(float).eps)
relative_error = np.abs(diff) / denominator

# Return the mean absolute percentage error
e = float(np.mean(relative_error) * 100.0)
