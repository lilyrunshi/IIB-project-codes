import numpy as np
from sklearn.linear_model import BayesianRidge

# inputs: X, y
br = BayesianRidge(compute_score=False)
br.fit(X, y)

y_hat = br.predict(X)
fit = {
    "coef_": br.coef_.tolist(),
    "intercept_": float(br.intercept_),
    "alpha_noise_": float(br.alpha_),   # noise precision
    "lambda_weights_": float(br.lambda_)  # weights precision
}
