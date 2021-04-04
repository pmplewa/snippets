from george import GP
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from warnings import warn

from .plot import *

__all__ = ["bayes_minimize_1D", "bayes_minimize"]

# --------------------------------------------------------------------------------------------------

def neg_log_likelihood(p, gp, y):
    gp.set_parameter_vector(p)
    grad = gp.grad_log_likelihood(y, quiet=True)
    return -gp.log_likelihood(y, quiet=True), -grad

def expected_improvement(x, gp, y):
    mu, var = gp.predict(y, x, return_var=True)
    std = np.sqrt(var)
    y_min = np.min(y)
    chi = (y_min - mu)/std
    return (y_min - mu)*norm.cdf(chi) + std*norm.pdf(chi)

search_methods = {
    "grid": lambda bounds, n: np.linspace(bounds[0], bounds[1], n),
    "random": lambda bounds, n: np.random.uniform(bounds[0], bounds[1], n)}

# -- 1D Case ---------------------------------------------------------------------------------------

def bayes_minimize_1D(objective, n_iter, bounds, kernel, x0=None, n_init=5, warnings=True,
        search_method="random", n_search=100, rtol=1e-5, full_output=False, plot_args=None):
    assert bounds.ndim == 1
    assert bounds.shape[0] == 2

    if plot_args is not None:
        make_plots = True
        axes, x_val = plot_args
        assert x_val.ndim == 1
        j = 0
    else:
        make_plots = False

    sample = search_methods[search_method]

    if x0 is None:
        x = np.random.uniform(bounds[0], bounds[1], n_init)
    else:
        assert x0.ndim == 1
        x = x0

    y = objective(x)

    gp = GP(np.var(y)*kernel, fit_mean=True)
    gp.compute(x)

    x_last = np.inf
    for i in range(n_iter):
        fit_func = neg_log_likelihood
        fit = minimize(fit_func, gp.get_parameter_vector(), args=(gp, y), jac=True)
        if warnings and not fit.success:
            warn(fit.message)

        x_test = sample(bounds, n_search)
        a_ei = expected_improvement(x_test, gp, y)
        x_next = x_test[np.argmax(a_ei)]

        if make_plots and i < len(axes.flat):
            ax = axes.flat[j]
            j += 1
            plot_objective_1D(ax, x_val, objective)
            plot_gp_1D(ax, x_val, gp, y)
            plot_sample_1D(ax, x, y)
            plot_proposal_1D(ax, x_next, objective)
            ax2 = plot_acquisition_1D(ax, x_test, a_ei)
            plot_estimate_1D(ax, x_last)
            plot_annotations_1D(ax, i)

        x = np.append(x, x_next)
        y = np.append(y, objective(x_next))
        gp.compute(x)

        fit_func = lambda x: gp.predict(y, x, return_cov=False)
        fit = minimize(fit_func, x[np.argmin(y)], bounds=[bounds])
        if warnings and not fit.success:
            warn(fit.message)

        x_min = fit.x

        if i > 0 and np.abs((x_min - x_last)/x_min) < rtol:
            if make_plots:
                ax.cla()
                ax2.cla()
                plot_objective_1D(ax, x_val, objective)
                plot_gp_1D(ax, x_val, gp, y)
                plot_sample_1D(ax, x, y)
                plot_estimate_1D(ax, x_min)
                plot_annotations_1D(ax, i)

            if full_output:
                return x_min, x, y
            else:
                return x_min
        else:
            x_last = x_min

# -- ND Case ---------------------------------------------------------------------------------------

def bayes_minimize(objective, n_iter, bounds_list, kernel, x0=None, n_init=5, warnings=True,
        search_method="random", n_search=100, rtol=1e-5, full_output=False, plot_args=None):
    assert kernel.ndim > 1
    assert bounds_list.ndim == 2
    assert bounds_list.shape[0] == kernel.ndim
    assert bounds_list.shape[1] == 2

    if plot_args is not None:
        make_plots = True
        axes, x_val = plot_args
        assert x_val.ndim == 2
        assert x_val.shape[0] == 2 # only allow 2D plots
        x_grid = np.meshgrid(*x_val)
        j = 0
    else:
        make_plots = False

    sample = search_methods[search_method]

    if x0 is None:
        x = np.transpose([sample(bounds, n_init) for bounds in bounds_list])
    else:
        assert x0.ndim == 2
        assert x0.shape[1] == kernel.ndim
        x = x0

    y = np.array([objective(x_val) for x_val in x])

    gp = GP(np.var(y)*kernel, fit_mean=True)
    gp.compute(x)

    x_last = np.full(kernel.ndim, np.inf)
    for i in range(n_iter):
        fit_func = neg_log_likelihood
        fit = minimize(fit_func, gp.get_parameter_vector(), args=(gp, y), jac=True)
        if warnings and not fit.success:
            warn(fit.message)

        x_test = np.transpose([sample(bounds, n_search) for bounds in bounds_list])
        a_ei = expected_improvement(x_test, gp, y)
        x_next = x_test[np.argmax(a_ei)]

        if make_plots and i < len(axes.flat):
            ax = axes.flat[j]
            j += 1
            plot_objective_2D(ax, x_grid, objective)
            plot_gp_2D(ax, x_grid, gp, y)
            plot_sample_2D(ax, x)
            plot_proposal_2D(ax, x_next)
            plot_estimate_2D(ax, x_last)
            plot_annotations_2D(ax, i)

        x = np.append(x, np.atleast_2d(x_next), axis=0)
        y = np.append(y, objective(x_next))
        gp.compute(x)

        fit_func = lambda x: gp.predict(y, np.atleast_2d(x), return_cov=False)
        fit = minimize(fit_func, x[np.argmin(y)], bounds=bounds_list)
        if warnings and not fit.success:
            warn(fit.message)

        x_min = fit.x

        if i > 0 and np.linalg.norm(np.abs((x_min - x_last)/x_min)) < rtol:
            if make_plots:
                ax.cla()
                plot_objective_2D(ax, x_grid, objective)
                plot_gp_2D(ax, x_grid, gp, y)
                plot_sample_2D(ax, x)
                plot_estimate_2D(ax, x_min)
                plot_annotations_2D(ax, i)

            if full_output:
                return x_min, x, y
            else:
                return x_min
        else:
            x_last = x_min

# --------------------------------------------------------------------------------------------------
