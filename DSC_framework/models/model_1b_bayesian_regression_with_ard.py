import numpy as np
from sklearn.linear_model import ARDRegression

# inputs: x, y
ard = ARDRegression(compute_score=False, fit_intercept=False)
ard.fit(x, y)

y_hat = ard.predict(x)
fit = {
    "coef_": ard.coef_.tolist(),
    "intercept_": float(ard.intercept_),
    "alpha_noise_": float(ard.alpha_),    # noise precision
    "lambda_weights_": ard.lambda_.tolist(),
    "w_mean": ard.coef_.tolist(),
    "w_cov": ard.sigma_.tolist(),
}

