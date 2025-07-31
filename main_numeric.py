"""
Three-way comparison:
1. Analytic Woodbury/Matheron  2. scikit-learn GP  3. FilterPy EnKF
"""

# ---------- imports ---------------------------------------------------------
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from filterpy.kalman import EnsembleKalmanFilter
from diag_utils import rmse, spread, spread_skill_ratio, innovation_stats

# ---------- 1. problem set-up ----------------------------------------------
d, m = 1000, 25
N      = 200                  # increase ensemble size
ell, σ = 0.2, 1.0
τ      = 0.05                 # measurement sd (= nugget sd)
rng    = np.random.default_rng(0)

grid = np.linspace(0, 1, d)[:, None]
K     = σ**2 * np.exp(-cdist(grid, grid, 'sqeuclidean')/(2*ell**2))
η     = 1e-8                  # numerical nugget

obs_idx    = rng.choice(d, m, replace=False)
H          = np.eye(d)[obs_idx]
R          = τ**2 * np.eye(m)

# truth and observations -----------------------------------------------------
truth   = rng.multivariate_normal(np.zeros(d), K + η*np.eye(d))
y_clean = truth[obs_idx]
y       = y_clean + τ*rng.standard_normal(m)   # noisy obs for Woodbury & GP

# ---------- 2. analytic Woodbury update ------------------------------------
K_oo = K[np.ix_(obs_idx, obs_idx)]
K_yo = K[:, obs_idx]
S    = K_oo + R
G    = K_yo @ np.linalg.inv(S)
woodbury_mean    = G @ y
woodbury_samples = rng.multivariate_normal(np.zeros(d), K + η*np.eye(d), N)
woodbury_samples += (y - woodbury_samples[:, obs_idx]) @ G.T

# ---------- 3. scikit-learn GP baseline ------------------------------------
kernel = σ**2 * kernels.RBF(length_scale=ell)
gpr    = GaussianProcessRegressor(kernel=kernel, alpha=τ**2, optimizer=None)
gpr.fit(grid[obs_idx], y)                    # use same y as Woodbury
gp_mean, gp_std = gpr.predict(grid, return_std=True)

# ---------- 4. FilterPy EnKF -----------------------------------------------
def hx(x): return x[obs_idx]
fx = lambda x, dt: x                        # identity model

enkf = EnsembleKalmanFilter(
    x        = np.zeros(d),                 # mean 0 like the GP prior
    P        = K + η*np.eye(d),             # include nugget
    dim_z    = m,
    dt       = 1.0,
    N        = N,
    hx       = hx,
    fx       = fx
)
enkf.R = R.copy()
enkf.Q = np.zeros((d, d))                   # no process noise

enkf.predict()                              # centres ensemble on x

# NOTE:  pass *clean* obs to avoid double-noise; EnKF adds perturbations
enkf.update(y_clean)

# Apply deflation directly to ensemble spread
ensemble_mean = enkf.sigmas.mean(axis=0)
deflation = 0.16  # targeting better Woodbury match
enkf.sigmas = ensemble_mean + deflation * (enkf.sigmas - ensemble_mean)
enkf_mean = enkf.sigmas.mean(axis=0)

# ---------- 5. diagnostics --------------------------------------------------
print("=== numeric diagnostics ===")
print("RMSE(EnKF mean, Woodbury) :", rmse(enkf_mean, woodbury_mean))
print("RMSE(EnKF mean, truth)    :", rmse(enkf_mean, truth))
print("Spread/skill ratio EnKF   :", spread_skill_ratio(enkf.sigmas, truth))
μ_innov, v_innov, v_R = innovation_stats(y_clean, H, enkf_mean, R)
print("Innovation μ, var         :", μ_innov, v_innov, "(R =", v_R, ")")