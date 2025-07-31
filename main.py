import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from tueplots import bundles, figsizes

# ---------- 1.  problem set-up ---------------------------------------------
d, N, m = 1000, 100, 25
grid = np.linspace(0.0, 1.0, d)[:, None]
ell, sigma, tau = 0.2, 1.0, 0.05

K = sigma**2 * np.exp(-cdist(grid, grid, 'sqeuclidean') / (2*ell**2))
rng = np.random.default_rng(0)

prior_samples = rng.multivariate_normal(np.zeros(d), K, size=N)      # (N,d)

obs_idx = rng.choice(d, m, replace=False)
true_state = prior_samples[0]                                        # ground truth
y = true_state[obs_idx] + tau * rng.standard_normal(m)

# ---------- 2.  exact Matheron (Woodbury) update ---------------------------
K_oo = K[np.ix_(obs_idx, obs_idx)]             # (m,m)
K_yo = K[:, obs_idx]                           # (d,m)
S = K_oo + tau**2 * np.eye(m)
gain = K_yo @ np.linalg.inv(S)                 # (d,m)

posterior_mean = gain @ y                      # (d,)

# Pathwise update for all N prior samples
posterior_samples = prior_samples + (y - prior_samples[:, obs_idx]) @ gain.T   # (N,d)

# ---------- 3.  classic GP baseline (mean + 50 samples) --------------------
kernel = sigma**2 * kernels.RBF(length_scale=ell)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=tau**2, optimizer=None)
gpr.fit(grid[obs_idx], y)

gp_mean, gp_std = gpr.predict(grid, return_std=True)
gp_samples = gpr.sample_y(grid,
                          n_samples=50,
                          random_state=42).T

# ---------- 4. diagnostics -------------------------------------------------
rmse = np.sqrt(np.mean((posterior_mean - gp_mean)**2))
print(f"RMSE(Matheron mean, GP mean) = {rmse:.2e} (should be ≪ τ={tau})")

# ---------- 5. plotting: 100 Matheron + 50 GP samples ----------------------
import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

with plt.rc_context({
        **bundles.iclr2024(),
        **figsizes.iclr2024(nrows=1, ncols=1),
        "figure.dpi": 300,
}):
    fig, ax = plt.subplots()

    x = grid.ravel()                 # 1-D view for plotting
    # --- Matheron samples (blue)
    ax.plot(x, posterior_samples[:100].T,
            color="C0", alpha=0.08, lw=0.8)
    # --- GP samples (orange, higher alpha so they pop)
    ax.plot(x, gp_samples.T,
            color="C1", alpha=0.12, lw=0.8)

    # --- Posterior means
    # ax.plot(x, posterior_mean, color="C0", lw=1.5)
    ax.plot(x, gp_mean,       color="0.1", lw=1.5, ls="--")

    # --- Observations
    obs_scatter = ax.scatter(grid[obs_idx], y, s=12,
                             facecolor="white", edgecolor="k", zorder=3, label="Observation")

    ax.set_xlabel("Distance")
    ax.set_ylabel("Temperature")

    # ---- single legend entry per group via proxy artists -------------------
    legend_handles = [
        Line2D([0], [0], color="C0", lw=1.5, label="EnKF samples"),
        Line2D([0], [0], color="C1", lw=1.5, label="GP samples"),
        # Line2D([0], [0], color="C0", lw=1.5, label="Matheron mean"),
        Line2D([0], [0], color="0.1", lw=1.5, ls="--", label="GP mean"),
        obs_scatter,
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=7, ncols=2)

    fig.savefig("fig_enkf_vs_gp.pdf")

