import numpy as np

# -- 1D Plots --------------------------------------------------------------------------------------

def plot_objective_1D(ax, x_val, objective):
    ax.plot(x_val, objective(x_val), "C0")

def plot_gp_1D(ax, x_val, gp, y):
    mu, var = gp.predict(y, x_val, return_var=True)
    std = np.sqrt(var)
    ax.plot(x_val, mu, "k")
    ax.fill_between(x_val, mu + std, mu - std, color="k", alpha=0.1)

def plot_sample_1D(ax, x, y):
    ax.plot(x, y, "oC0")

def plot_proposal_1D(ax, x_next, objective):
    ax.plot(x_next, objective(x_next), "oC1")

def plot_acquisition_1D(ax, x_test, a_ei):
    ax2 = ax.twinx()
    ordered = np.argsort(x_test)
    ax2.plot(x_test[ordered], a_ei[ordered], "C1", lw=0.75)
    ax2.set_yticks([])
    return ax2

def plot_estimate_1D(ax, x_val):
    ax.axvline(x_val, color="k", lw=0.75)

def plot_annotations_1D(ax, i):
    ax.annotate(f"{i+1}", xy=(0, 1), xycoords="axes fraction", ha="left", va="top",
        xytext=(5, -5), textcoords="offset points", fontsize=14)

# -- 2D Plots --------------------------------------------------------------------------------------

def plot_objective_2D(ax, x_grid, objective):
    ax.pcolormesh(x_grid[0], x_grid[1], objective(x_grid), cmap="Blues", shading="auto")

def plot_gp_2D(ax, x_grid, gp, y):
    x_val = np.transpose([x_grid[0].ravel(), x_grid[1].ravel()])
    mu = gp.predict(y, x_val, return_cov=False)
    ax.contour(x_grid[0], x_grid[1], mu.reshape(x_grid[0].shape), colors="k", linestyles="solid")

def plot_sample_2D(ax, x_val):
    ax.plot(x_val[:,0], x_val[:,1], "oC0")

def plot_proposal_2D(ax, x_next):
    ax.plot(x_next[0], x_next[1], "oC1")

def plot_estimate_2D(ax, x_val):
    ax.axvline(x_val[0], color="k", lw=0.75)
    ax.axhline(x_val[1], color="k", lw=0.75)

def plot_annotations_2D(*args):
    return plot_annotations_1D(*args)

# ------------------------------------------------------------------------------
