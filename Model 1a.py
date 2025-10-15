import numpy as np
from scipy.special import digamma 
from scipy.special import gammaln
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- ASSUMPTION: SyntheticData.py is in the same directory ---
# NOTE: The execution environment assumes this import works.
from SyntheticData import SyntheticData 

# --- VBLR CORE FUNCTIONS ---

def calculate_elbo(N, M, X, y, mN, SN, aN, bN, cN, dN, a0, b0, c0, d0):
    """Calculates the Evidence Lower Bound (ELBO) for Model 1a."""
    
    E_log_alpha = digamma(aN) - np.log(bN)
    E_log_beta = digamma(cN) - np.log(dN)
    E_omega_sq = mN.T @ mN + np.trace(SN)
    y_residual = y - X @ mN
    E_residual_sq = y_residual.T @ y_residual + np.trace(X @ SN @ X.T)
    E_alpha = aN / bN
    E_beta = cN / dN
    (sign, logdet_SN) = np.linalg.slogdet(SN)
    
    # E[log p(y, omega, alpha, beta)] terms
    E_log_lik = N/2 * E_log_beta - N/2 * np.log(2 * np.pi) - E_beta/2 * E_residual_sq.item()
    E_log_prior_omega = -M/2 * np.log(2 * np.pi) + M/2 * E_log_alpha - E_alpha/2 * E_omega_sq.item()
    E_log_prior_alpha = a0 * np.log(b0) + (a0 - 1) * E_log_alpha - b0 * E_alpha - gammaln(a0)
    E_log_prior_beta = c0 * np.log(d0) + (c0 - 1) * E_log_beta - d0 * E_beta - gammaln(c0)
    E_log_joint = E_log_lik + E_log_prior_omega + E_log_prior_alpha + E_log_prior_beta

    # -E[log q(omega, alpha, beta)] terms
    neg_E_log_q_omega = M/2 * (1 + np.log(2 * np.pi)) + 0.5 * logdet_SN
    neg_E_log_q_alpha = gammaln(aN) - (aN - 1) * E_log_alpha + aN - aN * np.log(bN)
    neg_E_log_q_beta = gammaln(cN) - (cN - 1) * E_log_beta + cN - cN * np.log(dN)
    neg_E_log_q = neg_E_log_q_omega + neg_E_log_q_alpha + neg_E_log_q_beta

    # Total ELBO
    ELBO = E_log_joint + neg_E_log_q
    return ELBO.item()

def variational_bayes_linear_regression(X, y, a0, b0, c0, d0, max_iter=100, tol=1e-5):
    """Implements Variational Bayes Linear Regression (Model 1a)."""
    N, M = X.shape 

    E_alpha = a0 / b0
    E_beta = c0 / d0

    aN = a0 + M / 2
    cN = c0 + N / 2
    bN = b0
    dN = d0

    SN_inv = E_beta * X.T @ X + E_alpha * np.eye(M)
    SN = np.linalg.inv(SN_inv)
    mN = SN @ (E_beta * X.T @ y)

    elbo_history = []
    i = 0 
    for i in range(max_iter):
        mN_old = mN.copy()

        E_omega_sq = mN.T @ mN + np.trace(SN)
        bN = b0 + 0.5 * E_omega_sq.item()
        E_alpha = aN / bN

        y_residual = y - X @ mN
        E_residual_sq = y_residual.T @ y_residual + np.trace(X @ SN @ X.T)
        dN = d0 + 0.5 * E_residual_sq.item()
        E_beta = cN / dN

        SN_inv = E_beta * X.T @ X + E_alpha * np.eye(M) 
        SN = np.linalg.inv(SN_inv)
        mN = E_beta * SN @ X.T @ y 

        ELBO = calculate_elbo(N, M, X, y, mN, SN, aN, bN, cN, dN, a0, b0, c0, d0)
        elbo_history.append(ELBO)
        
        if i > 0 and np.linalg.norm(mN - mN_old) / np.linalg.norm(mN_old) < tol:
            break

    Estimated_Signal = X @ mN
    
    return Estimated_Signal, elbo_history, i + 1, mN, aN, bN, cN, dN


# --- EXPERIMENT SETUP ---

# 1. Generate Complex Features
N_SAMPLES = 200    # Number of time points (samples)
N_FEATURES = 50    # Number of features (potential predictors)
N_GROUPS = 3       # Rhythmic groups: 0, 1, 2, 3 (harmonics)

data_gen = SyntheticData(n_features=N_FEATURES, n_samples=N_SAMPLES, seed=42)
data_gen.oscillatory_groups(n_groups=N_GROUPS)

# X_features is the FEATURES x SAMPLES matrix
X_features_full, group_ind, _ = data_gen.generate_data(rhy_frac=0.3, normal_dist_times=False)

# Transpose X so VBLR has (SAMPLES x FEATURES) structure (N=200, M=50)
X_VBLR = X_features_full.T 

# 2. Define True Sparse Model (Ground Truth)
# We assume only a few features truly predict the output.
TRUE_MODEL_COEFFS = np.zeros(N_FEATURES)
TRUE_MODEL_COEFFS[4] = 2.5   # Strong predictor (may be rhythmic or arrhythmic)
TRUE_MODEL_COEFFS[15] = -1.0 # Medium predictor
TRUE_MODEL_COEFFS[20] = 0.5  # Weak predictor

# 3. Create Synthetic Target Vector (y)
NOISE_VAR = 0.5 
TRUE_NOISE_PRECISION = 1.0 / NOISE_VAR

# y = X * omega_true + noise
True_Target = X_VBLR @ TRUE_MODEL_COEFFS
Noise = np.sqrt(NOISE_VAR) * np.random.randn(N_SAMPLES)
y_target = (True_Target + Noise).reshape(-1, 1)


# 4. VBLR Priors
a0 = 1.0; b0 = 1.0   # Weak Gamma prior on alpha (E[alpha]=1)
c0 = 2.0; d0 = 1.0   # Semi-informative Gamma prior on beta (E[beta]=2, close to true 2)


# --- 5. RUN VBLR ---
Est_Sig, ELBO_Hist, Iters, mN, aN, bN, cN, dN = variational_bayes_linear_regression(
    X_VBLR, y_target, a0, b0, c0, d0
)

# --- 6. ANALYSIS AND PRINTING ---
E_omega = mN.flatten()
E_alpha = aN / bN
E_beta = cN / dN

print("\n" + "="*60)
print("VBLR Analysis Results on SyntheticData Features")
print("="*60)
print(f"Dataset Dimensions: N_Samples={N_SAMPLES}, N_Features={N_FEATURES}")
print(f"True Noise Variance: {NOISE_VAR}, True Precision: {TRUE_NOISE_PRECISION:.4f}")
print(f"Convergence Iterations: {Iters}")
print("-" * 60)

print("True Non-Zero Coefficients and Estimates (Sparse Recovery Test):")
# Check only the features that were truly used
for idx in [4, 15, 20]:
    print(f"  Feature {idx:2} (True={TRUE_MODEL_COEFFS[idx]:>4.1f}): Estimated={E_omega[idx]:.4f}")

# Check a few non-predictor features (should be close to zero)
print("\nNon-Predictor Feature Estimates (Should be close to zero):")
for idx in [0, 10, 30]:
    print(f"  Feature {idx:2} (True=0.0): Estimated={E_omega[idx]:.4f}")


print("\nHyperparameter Estimates:")
print(f"Estimated Noise Precision (E[β]): {E_beta:.4f} (Target was {TRUE_NOISE_PRECISION:.4f})")
print(f"Estimated Coeff Precision (E[α]): {E_alpha:.4f}")
print(f"Final ELBO: {ELBO_Hist[-1]:.4f}")
print("="*60)


# --- 7. VISUALIZATION ---
plt.figure(figsize=(15, 6))

# Subplot 1: Signal Approximation
ax1 = plt.subplot(1, 2, 1)
sample_times = np.arange(N_SAMPLES)

ax1.plot(sample_times, y_target, 'b.', alpha=0.3, label='Noisy Target ($y$)')
ax1.plot(sample_times, True_Target, 'g-', linewidth=2, label='True Target Signal')
ax1.plot(sample_times, Est_Sig, 'r--', linewidth=2, label='VBLR Estimated Signal')
ax1.set_title('Target Signal Approximation (N=200, M=50)')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Target Amplitude')
ax1.legend()
ax1.grid(True)

# Subplot 2: ELBO Progression
ax2 = plt.subplot(1, 2, 2)
ax2.plot(ELBO_Hist, 'm-', linewidth=2)
ax2.set_title(f'ELBO Progression (Converged in {Iters} Iters)')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('ELBO (Evidence Lower Bound)')
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.grid(True)

plt.tight_layout()
plt.show()