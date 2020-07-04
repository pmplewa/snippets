import autograd.numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

from .psf import gen_psf
from .zernike import index


def plot_psf(n_grid=5, figsize=(10, 10)):
    fig, ax_list = plt.subplots(n_grid, n_grid, figsize=figsize)
    norm = LogNorm(vmin=0.05, vmax=4)

    j = 0
    for k in range(n_grid):
        for l in range(n_grid):
            ax = ax_list[k, l]   
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])            
            if k >= l:
                cj = np.zeros(j+1)
                cj[j] = 1

                image = gen_psf(cj, norm=False)
                
                ax.imshow(image, origin="lower", cmap="Greys", norm=norm)
                
                n, m = index(j)
                ax.set_title(f"j={j} ({n}, {m})")
                
                j += 1
            else:
                ax.axis("off")
    
    plt.tight_layout()
    plt.show()
