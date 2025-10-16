import numpy as np
from scipy.special import digamma 
from scipy.special import gammaln
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- VBLR CORE FUNCTIONS (Model 1a - Standard VBLR) ---

def calculate_elbo_model1a(N, M, X, y, mN, SN, aN, bN, cN, dN, a0, b0, c0, d0):
    """Calculates the ELBO for Model 1a."""
    
    # E[alpha] is scalar
    E_alpha = aN / bN
    E_beta = cN / dN
    E_log_alpha = digamma(aN) - np.log(bN)
    E_log_beta = digamma(cN) - np.log(dN)
    
    # E[omega^T omega] is scalar
    E_omega_sq = mN.T @ mN + np.trace(SN)
    
    y_residual = y - X @ mN
    E_residual_sq = y_residual.T @ y_residual + np.trace(X @ SN @ X.T)
    (sign, logdet_SN) = np.linalg.slogdet(SN)
    
    # E[log p] terms 
    E_log_lik = N/2 * E_log_beta - N/2 * np.log(2 * np.pi) - E_beta/2 * E_residual_sq.item()
    E_log_prior_omega = -M/2 * np.log(2 * np.pi) + M/2 * E_log_alpha - E_alpha/2 * E_omega_sq.item()
    E_log_prior_alpha = a0 * np.log(b0) + (a0 - 1) * E_log_alpha - b0 * E_alpha - gammaln(a0)
    E_log_prior_beta = c0 * np.log(d0) + (c0 - 1) * E_log_beta - d0 * E_beta - gammaln(c0)
    E_log_joint = E_log_lik + E_log_prior_omega + E_log_prior_alpha + E_log_prior_beta
    
    # -E[log q] terms
    neg_E_log_q_omega = M/2 * (1 + np.log(2 * np.pi)) + 0.5 * logdet_SN
    neg_E_log_q_alpha = gammaln(aN) - (aN - 1) * E_log_alpha + aN - aN * np.log(bN)
    neg_E_log_q_beta = gammaln(cN) - (cN - 1) * E_log_beta + cN - cN * np.log(dN)
    neg_E_log_q = neg_E_log_q_omega + neg_E_log_q_alpha + neg_E_log_q_beta

    ELBO = E_log_joint + neg_E_log_q
    return ELBO.item()

def variational_bayes_linear_regression_standard(X, y, a0, b0, c0, d0, max_iter=200, tol=1e-6):
    """
    Implements Standard VBLR (Model 1a).
    Cannot perform basis selection as it uses a single alpha for all features.
    """
    N, M = X.shape 

    # E[alpha] is a single scalar
    E_alpha = a0 / b0 
    E_beta = c0 / d0

    # aN, cN same as Model 1a
    aN = a0 + M / 2 
    cN = c0 + N / 2
    bN = b0
    dN = d0

    # S_N^-1 uses scalar E[alpha] * I
    SN_inv = E_beta * X.T @ X + E_alpha * np.eye(M)
    SN = np.linalg.inv(SN_inv)
    mN = SN @ (E_beta * X.T @ y)

    elbo_history = []
    i = 0 
    for i in range(max_iter):
        mN_old = mN.copy()

        # 1. Update bN (Scalar)
        E_omega_sq = mN.T @ mN + np.trace(SN)
        bN = b0 + 0.5 * E_omega_sq.item()
        E_alpha = aN / bN

        # 2. Update dN (Scalar)
        y_residual = y - X @ mN
        E_residual_sq = y_residual.T @ y_residual + np.trace(X @ SN @ X.T)
        dN = d0 + 0.5 * E_residual_sq.item()
        E_beta = cN / dN

        # 3. Update SN_inv and mN (Key Difference: Uses scalar E[alpha])
        SN_inv = E_beta * X.T @ X + E_alpha * np.eye(M) 
        SN = np.linalg.inv(SN_inv)
        mN = SN @ (E_beta * X.T @ y)

        # 4. ELBO Calculation 
        ELBO = calculate_elbo_model1a(N, M, X, y, mN, SN, aN, bN, cN, dN, a0, b0, c0, d0)
        elbo_history.append(ELBO)
        
        # 5. Convergence Check 
        if i > 0 and np.linalg.norm(mN - mN_old) / np.linalg.norm(mN_old) < tol:
            break

    return X @ mN, elbo_history, i + 1, mN, E_alpha, E_beta


# --- EXPERIMENT: VBLR (Model 1a) on Overcomplete Dictionary ---

# 1. Setup Time and True Signal (Low Noise settings from previous run)
N_SAMPLES = 400
T = np.linspace(0, 5, N_SAMPLES)
NOISE_VAR = 0.1
TRUE_NOISE_PRECISION = 1.0 / NOISE_VAR

TRUE_FREQS = [2.0, 5.0]
TRUE_COEFFS = [1.5, 0.75, -1.0, 0.5] 

True_Signal = (TRUE_COEFFS[0] * np.sin(2 * np.pi * TRUE_FREQS[0] * T) + TRUE_COEFFS[1] * np.cos(2 * np.pi * TRUE_FREQS[0] * T) +
               TRUE_COEFFS[2] * np.sin(2 * np.pi * TRUE_FREQS[1] * T) + TRUE_COEFFS[3] * np.cos(2 * np.pi * TRUE_FREQS[1] * T)
              )
y_noisy = (True_Signal + np.sqrt(NOISE_VAR) * np.random.randn(N_SAMPLES)).reshape(-1, 1)


# 2. Construct the Overcomplete Dictionary X_dict (The 'Unknown X' proxy)
CANDIDATE_FREQS = np.linspace(1.0, 10.0, 20)
CANDIDATE_FREQS = np.unique(np.sort(np.append(CANDIDATE_FREQS, TRUE_FREQS)))

BASIS_FUNCTIONS = []
BASIS_NAMES = []
FREQS_MAP = [] 

BASIS_FUNCTIONS.append(np.ones(N_SAMPLES))
BASIS_NAMES.append("DC_Offset")
FREQS_MAP.append(0.0)

for f in CANDIDATE_FREQS:
    BASIS_FUNCTIONS.append(np.sin(2 * np.pi * f * T))
    BASIS_FUNCTIONS.append(np.cos(2 * np.pi * f * T))
    BASIS_NAMES.extend([f"Sin(2π*{f:.1f}t)", f"Cos(2π*{f:.1f}t)"])
    FREQS_MAP.extend([f, f])

X_DICT = np.vstack(BASIS_FUNCTIONS).T
M_FEATURES = X_DICT.shape[1]


# 3. VBLR Priors (Using ARD's weak prior, but Model 1a treats it as a single scalar)
a0 = 1e-4; b0 = 1e-4 
c0 = 1.0; d0 = 1.0   


# 4. RUN VBLR (Model 1a)
Est_Sig_1a, ELBO_Hist_1a, Iters_1a, mN_1a, E_alpha_1a, E_beta_1a = variational_bayes_linear_regression_standard(
    X_DICT, y_noisy, a0, b0, c0, d0
)


# --- 5. ANALYSIS AND PRINTING ---

print("\n" + "="*80)
print("Model 1a Results on Overcomplete Dictionary (Expected to Fail Sparsity)")
print("==================================================================")
print(f"Dataset Dimensions: N_Samples={N_SAMPLES}, M_Features={M_FEATURES}")
print(f"Convergence Iterations: {Iters_1a}")
print(f"Estimated Noise Precision (E[β]): {E_beta_1a:.4f} (True: {TRUE_NOISE_PRECISION:.4f})")
print(f"Estimated Single Coeff Precision (E[α]): {E_alpha_1a:.4f}")
print(f"Final ELBO: {ELBO_Hist_1a[-1]:.4f}")
print("-" * 80)

# Check the largest coefficients (Model 1a is expected to shrink *all* of them equally)
# Find the coefficients with the largest magnitude
mN_abs_sorted_indices = np.argsort(np.abs(mN_1a.flatten()))[::-1]

print("Top 10 Coefficients by Magnitude:")
print("{:<15} {:>8} {:>10}".format("Basis Name", "Frequency", "Est. Coeff."))
print("-" * 50)
for i in mN_abs_sorted_indices[:10]:
    print("{:<15} {:>8.1f} {:>10.4f}".format(
        BASIS_NAMES[i][:15], FREQS_MAP[i], mN_1a.flatten()[i]))

print("="*80)


# 6. VISUALIZATION 
plt.figure(figsize=(15, 6))

# Subplot 1: Signal Approximation
ax1 = plt.subplot(1, 2, 1)
ax1.plot(T, y_noisy, 'b.', alpha=0.3, label='Noisy Target ($y$)')
ax1.plot(T, True_Signal, 'g-', linewidth=2, label='True Target Signal')
ax1.plot(T, Est_Sig_1a, 'r--', linewidth=2, label='Model 1a Estimated Signal')
ax1.set_title('Model 1a Fit')
ax1.set_xlabel('Time')
ax1.set_ylabel('Target Amplitude')
ax1.legend()
ax1.grid(True)

# Subplot 2: ELBO Progression
ax2 = plt.subplot(1, 2, 2)
ax2.plot(ELBO_Hist_1a, 'm-', linewidth=2)
ax2.set_title(f'ELBO Progression (Converged in {Iters_1a} Iters)')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('ELBO (Evidence Lower Bound)')
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.grid(True)

plt.tight_layout()
plt.show()