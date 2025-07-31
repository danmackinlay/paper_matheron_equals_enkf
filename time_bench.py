import time, csv, numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from filterpy.kalman import EnsembleKalmanFilter
from diag_utils import rmse   # only to re-use the RNG seed helper

rng = np.random.default_rng(0)
ell, sig, tau, eta = 0.2, 1.0, 0.05, 1e-8
m = 25
dims = [500, 1000, 2000, 4000]

rows = []
for d in dims:
    print(f"Benchmarking d={d}...")
    
    # --- common setup ------------------------------------------------------
    grid = np.linspace(0, 1, d)[:, None]
    K = sig**2 * np.exp(-cdist(grid, grid, "sqeuclidean")/(2*ell**2))
    obs_idx = rng.choice(d, m, replace=False)
    H = np.eye(d)[obs_idx]
    R = tau**2 * np.eye(m)

    truth = rng.multivariate_normal(np.zeros(d), K + eta*np.eye(d))
    y = truth[obs_idx] + tau*rng.standard_normal(m)

    # 1. Woodbury -----------------------------------------------------------
    t0 = time.perf_counter()
    K_oo = K[np.ix_(obs_idx, obs_idx)]
    K_yo = K[:, obs_idx]
    S = K_oo + R
    gain = K_yo @ np.linalg.inv(S)
    _ = gain @ y                       # mean only, ignore samples
    wall_wood = time.perf_counter() - t0

    # 2. scikit-learn GP ----------------------------------------------------
    t0 = time.perf_counter()
    gpr = GaussianProcessRegressor(
        kernel=sig**2*kernels.RBF(ell), alpha=tau**2, optimizer=None)
    gpr.fit(grid[obs_idx], y)
    _ = gpr.predict(grid, return_std=False)
    wall_gp = time.perf_counter() - t0

    # 3. FilterPy EnKF ------------------------------------------------------
    N = max(100, d//20)  # Reduce ensemble size for faster benchmarking
    def hx(x): return x[obs_idx]
    fx = lambda x, dt: x
    t0 = time.perf_counter()
    enkf = EnsembleKalmanFilter(
        x=np.zeros(d), P=K+eta*np.eye(d),
        dim_z=m, dt=1.0, N=N, hx=hx, fx=fx)
    enkf.R = R; enkf.Q = np.zeros((d, d))
    enkf.predict(); enkf.update(truth[obs_idx])  # use clean obs for fairness
    _ = enkf.sigmas.mean(axis=0)
    wall_enkf = time.perf_counter() - t0

    rows.append({"d": d, "N": N,
                 "t_woodbury": wall_wood,
                 "t_gp": wall_gp,
                 "t_enkf": wall_enkf})
    
    print(f"  Woodbury: {wall_wood:.3f}s, GP: {wall_gp:.3f}s, EnKF: {wall_enkf:.3f}s")

# ----------- save CSV -------------------------------------------------------
pd.DataFrame(rows).to_csv("posterior_timing.csv", index=False)
print("saved -> posterior_timing.csv")