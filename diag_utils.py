import numpy as np

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b)**2)))

def spread(ens):
    return float(ens.std(axis=0).mean())

def spread_skill_ratio(ens, truth):
    return spread(ens) / rmse(ens.mean(axis=0), truth)

def innovation_stats(y, H, ens_mean, R):
    innov = y - H @ ens_mean
    return float(innov.mean()), float(np.var(innov)), float(np.mean(np.diag(R)))