"""
Two-way comparison:
1. Analytic EnKF/Matheron (Woodbury identity)  2. scikit-learn GP
"""

# ---------- imports ---------------------------------------------------------
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from diag_utils import rmse

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

# ---------- 2. analytic EnKF update (Woodbury identity) ------------------
K_oo = K[np.ix_(obs_idx, obs_idx)]
K_yo = K[:, obs_idx]
S    = K_oo + R
G    = K_yo @ np.linalg.inv(S)
enkf_mean    = G @ y
enkf_samples = rng.multivariate_normal(np.zeros(d), K + η*np.eye(d), N)
enkf_samples += (y - enkf_samples[:, obs_idx]) @ G.T

# ---------- 3. scikit-learn GP baseline ------------------------------------
kernel = σ**2 * kernels.RBF(length_scale=ell)
gpr    = GaussianProcessRegressor(kernel=kernel, alpha=τ**2, optimizer=None)
gpr.fit(grid[obs_idx], y)                    # use same y as Woodbury
gp_mean, gp_std = gpr.predict(grid, return_std=True)

# ---------- collect posterior samples ------------------------------------
gp_samples_matrix = gpr.sample_y(grid, n_samples=50,
                                 random_state=42).T   # (50, d)

# ---------- 4. diagnostics --------------------------------------------------
print("=== numeric diagnostics ===")
print("RMSE(EnKF mean, GP mean)  :", rmse(enkf_mean, gp_mean))
print("RMSE(EnKF mean, truth)    :", rmse(enkf_mean, truth))
print("RMSE(GP mean, truth)      :", rmse(gp_mean, truth))

# ---------- 5. posterior-sample figure -----------------------------------
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

    # 100 EnKF draws (blue, faint)
    ax.plot(x, enkf_samples[:100].T,
            color="C0", alpha=0.06, lw=0.8)

    # 50 GP draws (orange, faint)
    ax.plot(x, gp_samples_matrix.T,
            color="C1", alpha=0.12, lw=0.8)

    # Posterior means
    ax.plot(x, enkf_mean,      color="0.2", lw=1.4, ls="-")
    ax.plot(x, gp_mean,        color="0.2", lw=1.4, ls="--")

    # Observations
    obs_scatter = ax.scatter(grid[obs_idx], y_clean, s=12,
                             facecolor="white", edgecolor="k", zorder=3, label="Observations")

    ax.set_xlabel("Distance")
    ax.set_ylabel("Field value")

    # proxy handles for a clean legend
    legend_handles = [
        Line2D([0], [0], color="C0", lw=1.4, label="EnKF samples"),
        Line2D([0], [0], color="C1", lw=1.4, label="GP samples"),
        Line2D([0], [0], color="0.2", lw=1.4, ls="-", label="EnKF mean"),
        Line2D([0], [0], color="0.2", lw=1.4, ls="--", label="GP mean"),
        obs_scatter,
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=7, ncols=2)

    fig.savefig("fig_posterior_samples.pdf")
    print("saved figure -> fig_posterior_samples.pdf")