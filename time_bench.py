import time, csv, numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

ell, sig, tau, eta = 0.2, 1.0, 0.2, 1e-8  # Increased noise: tau=0.05 -> 0.2
dims = [200, 400, 600, 800]  # Memory-safe dimensions (all < 1000)
n_trials = 5  # Number of trials per dimension

rows = []
for d in dims:
    m = d // 5  # Number of observations scales with dimension
    print(f"Benchmarking d={d}, m={m} ({n_trials} trials)...")
    
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)  # Different seed per trial
    
        # --- common setup ------------------------------------------------------
        grid = np.linspace(0, 1, d)[:, None]
        K = sig**2 * np.exp(-cdist(grid, grid, "sqeuclidean")/(2*ell**2))
        obs_idx = rng.choice(d, m, replace=False)
        H = np.eye(d)[obs_idx]
        R = tau**2 * np.eye(m)

        truth = rng.multivariate_normal(np.zeros(d), K + eta*np.eye(d))
        y = truth[obs_idx] + tau*rng.standard_normal(m)

        # 1. EnKF (Woodbury identity) -------------------------------------------
        t0 = time.perf_counter()
        K_oo = K[np.ix_(obs_idx, obs_idx)]
        K_yo = K[:, obs_idx]
        S = K_oo + R
        gain = K_yo @ np.linalg.inv(S)
        _ = gain @ y                       # mean only, ignore samples
        wall_enkf = time.perf_counter() - t0

        # 2. scikit-learn GP ----------------------------------------------------
        t0 = time.perf_counter()
        gpr = GaussianProcessRegressor(
            kernel=sig**2*kernels.RBF(ell), alpha=tau**2, optimizer=None)
        gpr.fit(grid[obs_idx], y)
        _ = gpr.predict(grid, return_std=False)
        wall_gp = time.perf_counter() - t0


        rows.append({"d": d, "m": m, "trial": trial,
                     "t_enkf": wall_enkf,
                     "t_gp": wall_gp})
    
    # Print summary for this dimension
    df_d = pd.DataFrame([r for r in rows if r['d'] == d])
    enkf_mean, enkf_std = df_d['t_enkf'].mean(), df_d['t_enkf'].std()
    gp_mean, gp_std = df_d['t_gp'].mean(), df_d['t_gp'].std()
    print(f"  EnKF: {enkf_mean:.3f}±{enkf_std:.3f}s, GP: {gp_mean:.3f}±{gp_std:.3f}s")

# ----------- save CSV -------------------------------------------------------
# Save raw trial data
df_raw = pd.DataFrame(rows)
df_raw.to_csv("posterior_timing_raw.csv", index=False)
print("saved -> posterior_timing_raw.csv")

# Compute summary statistics
summary_rows = []
for d in dims:
    df_d = df_raw[df_raw['d'] == d]
    m = df_d['m'].iloc[0]
    
    enkf_mean, enkf_std = df_d['t_enkf'].mean(), df_d['t_enkf'].std()
    gp_mean, gp_std = df_d['t_gp'].mean(), df_d['t_gp'].std()
    
    summary_rows.append({
        "d": d, "m": m,
        "t_enkf_mean": enkf_mean, "t_enkf_std": enkf_std,
        "t_gp_mean": gp_mean, "t_gp_std": gp_std
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv("posterior_timing.csv", index=False)
print("saved -> posterior_timing.csv (summary with error bars)")