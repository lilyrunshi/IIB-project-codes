import numpy as np

rng = np.random.default_rng()

# inputs: n, d, snr_db
X = rng.normal(size=(n, d))
beta = rng.normal(size=d)

signal = X @ beta
# set noise variance to achieve target SNR (in dB)
snr_lin = 10 ** (snr_db / 10.0)
noise_var = np.var(signal) / snr_lin
y = signal + rng.normal(scale=np.sqrt(noise_var), size=n)
