import autograd.numpy as np
import matplotlib.pyplot as plt

from numpy import ma

from .zernike import eval_zernike, index


def plot_zernike(n_grid=5, figsize=(10, 10)):
    fig, ax_list = plt.subplots(n_grid, n_grid, figsize=figsize)
    
    x_val = np.linspace(-1, 1, 100)
    x_val, y_val = np.meshgrid(*[x_val]*2)
    mask = np.sqrt(x_val**2 + y_val**2) >= 1

    j = 0
    for k in range(n_grid):
        for l in range(n_grid):
            ax = ax_list[k, l]
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if k >= l:
                cj = np.zeros(j + 1)
                cj[j] = 1

                image = eval_zernike(x_val, y_val, cj)
                image = ma.array(image, mask=mask, fill_value=0)

                ax.imshow(image, origin="lower", cmap="RdBu", vmin=-3, vmax=3)

                n, m = index(j)
                ax.set_title(f"j={j} ({n}, {m})")

                j += 1
            else:
                ax.axis("off")

    plt.tight_layout()
    plt.show()
