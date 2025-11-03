import numpy as np
from sklearn.linear_model import BayesianRidge

# inputs: x, y
br = BayesianRidge(compute_score=False)
br.fit(x, y)

y_hat = br.predict(x)
fit = {
    "coef_": br.coef_.tolist(),
    "intercept_": float(br.intercept_),
    "alpha_noise_": float(br.alpha_),   # noise precision
    "lambda_weights_": float(br.lambda_),  # weights precision
}

