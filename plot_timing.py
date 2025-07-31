import pandas as pd, matplotlib.pyplot as plt
from tueplots import bundles, figsizes
from matplotlib.ticker import ScalarFormatter

df = pd.read_csv("posterior_timing.csv")

with plt.rc_context({
        **bundles.iclr2024(),
        **figsizes.iclr2024(nrows=1, ncols=1),
        "figure.dpi": 300,
}):
    fig, ax = plt.subplots()
    ax.plot(df.d, df.t_enkf, 'o-', label="EnKF (Woodbury)")
    ax.plot(df.d, df.t_gp,   's--', label="scikit-GP")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("State dimension $d$")
    ax.set_ylabel("Wall time [s]")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend(frameon=False)
    fig.savefig("fig_posterior_timing.pdf")
    print("saved -> fig_posterior_timing.pdf")