import numpy as np
from sklearn.linear_model import ARDRegression

# inputs: X, y
ard = ARDRegression(compute_score=False)
ard.fit(X, y)

y_hat = ard.predict(X)
fit = {
    "coef_": ard.coef_.tolist(),
    "intercept_": float(ard.intercept_),
    "alpha_noise_": float(ard.alpha_),    # noise precision
    "lambda_weights_": float(ard.lambda_) # average weights precision
}
