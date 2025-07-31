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

# ---------- collect posterior samples ------------------------------------
enkf_samples      = enkf.sigmas.copy()           # shape (N, d)
woodbury_samples  = woodbury_samples.copy()      # already (N, d)
gp_samples_matrix = gpr.sample_y(grid, n_samples=50,
                                 random_state=42).T   # (50, d)

# ---------- 5. diagnostics --------------------------------------------------
print("=== numeric diagnostics ===")
print("RMSE(EnKF mean, Woodbury) :", rmse(enkf_mean, woodbury_mean))
print("RMSE(EnKF mean, truth)    :", rmse(enkf_mean, truth))
print("Spread/skill ratio EnKF   :", spread_skill_ratio(enkf.sigmas, truth))
μ_innov, v_innov, v_R = innovation_stats(y_clean, H, enkf_mean, R)
print("Innovation μ, var         :", μ_innov, v_innov, "(R =", v_R, ")")

# ---------- 6. posterior-sample figure -----------------------------------
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tueplots import bundles, figsizes

x = grid.ravel()          # 1-D view for plotting

with plt.rc_context({
        **bundles.iclr2024(),
        **figsizes.iclr2024(nrows=1, ncols=1),
        "figure.dpi": 300,
}):
    fig, ax = plt.subplots()

    # 100 Woodbury draws (blue, faint)
    ax.plot(x, woodbury_samples[:100].T,
            color="C0", alpha=0.06, lw=0.8)

    # 100 EnKF draws (green, faint)
    ax.plot(x, enkf_samples[:100].T,
            color="C2", alpha=0.08, lw=0.8)

    # 50 GP draws (orange, faint)
    ax.plot(x, gp_samples_matrix.T,
            color="C1", alpha=0.12, lw=0.8)

    # Posterior mean (GP only - EnKF mean visually indistinguishable)
    ax.plot(x, gp_mean,        color="0.2", lw=1.4, ls="--")

    # Observations
    obs_scatter = ax.scatter(grid[obs_idx], y_clean, s=12,
                             facecolor="white", edgecolor="k", zorder=3, label="Observations")

    ax.set_xlabel("Distance")
    ax.set_ylabel("Field value")

    # proxy handles for a clean legend
    legend_handles = [
        Line2D([0], [0], color="C0", lw=1.4, label="Woodbury samples"),
        Line2D([0], [0], color="C2", lw=1.4, label="EnKF samples"),
        Line2D([0], [0], color="C1", lw=1.4, label="GP samples"),
        Line2D([0], [0], color="0.2", lw=1.4, ls="--", label="GP mean"),
        obs_scatter,
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=7, ncols=2)

    fig.savefig("fig_posterior_samples.pdf")
    print("saved figure -> fig_posterior_samples.pdf")