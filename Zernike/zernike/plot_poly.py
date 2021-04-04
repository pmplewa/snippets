import autograd.numpy as np
import matplotlib.pyplot as plt

from numpy import ma

from .poly import vec_poly_P, vec_poly_S


def plot_poly(which, n_grid=3, figsize=(10, 10)):
    fig, ax_list = plt.subplots(n_grid, n_grid, figsize=figsize)

    x_val = np.linspace(-1.1, 1.1, 10)
    x_val, y_val = np.meshgrid(*[x_val]*2)
    mask = np.sqrt(x_val**2 + y_val**2) >= 1

    j = 1
    for k in range(n_grid):
        for l in range(n_grid):
            ax = ax_list[k, l]
            ax.set_aspect("equal")
            ax.set_title(f"j = {j}")

            if which == "S":
                u, v = vec_poly_S(x_val, y_val, j)
            elif which == "P":
                u, v = vec_poly_P(x_val, y_val, j)
            else:
                raise Exception()

            u = ma.array(u, mask=mask, fill_value=0)
            v = ma.array(v, mask=mask, fill_value=0)

            ax.quiver(x_val, y_val, u, v, pivot="middle")
            circle = plt.Circle((0, 0), 1, facecolor="none", edgecolor="k")
            ax.add_artist(circle)

            j += 1

    plt.tight_layout()
    plt.show()
