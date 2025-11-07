import numpy as np

# inputs: y_true, y_pred
diff = y_true - y_pred

# Avoid division by zero by flooring the denominator at machine epsilon
denominator = np.maximum(np.abs(y_true), np.finfo(float).eps)
relative_squared_error = (diff / denominator) ** 2

# Return the root mean squared percentage error
e = float(np.sqrt(np.mean(relative_squared_error)) * 100.0)
