import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import SymLogNorm
from scipy.special import eval_hermite as hermite

from .shapelets import ShapeletPSF


def plot_hermite():
    fig, ax = plt.subplots()

    ax.set_title("Hermite Polynomials")

    x = np.linspace(-2, 2, 100)

    for nval in range(5):
        ax.plot(x, hermite(nval, x))

    ax.set_ylim([-25, 25])

    plt.show()

def plot_phi1d(nmax=5):
    fig, ax = plt.subplots()

    ax.set_title("1D Shapelet Basis Functions")

    psf = ShapeletPSF(nmax=nmax, scale=1)

    x = np.linspace(-5, 5, 100)
    for nval in range(nmax):
        ax.plot(x, psf.phi1d(x, nval))

    plt.show()

def plot_phi2d(nmax=4):    
    fig, ax = plt.subplots(nmax, nmax, figsize=(6, 6))

    plt.suptitle("2D Shapelet Basis Functions")

    psf = ShapeletPSF(nmax=nmax, scale=1)
    
    x = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(*[x]*2)
    
    plot_args = dict(interpolation="none", vmin=-0.5, vmax=0.5, extent=(-9, 9, -9, 9))
    for i in range(nmax):
        for j in range(nmax):
            image = psf.phi2d(x, y, i, j)
            ax[i, j].imshow(image, cmap="PuOr", **plot_args)

            if i != 3 and j != 0:
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            if i == 3 and j > 0:
                ax[i, j].set_yticks([])
            if j == 0 and i < 3:
                ax[i, j].set_xticks([])
    
    plt.show()

def plot_psf_fit(psf, vmin_rel=1e-3):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    
    vmax = np.max(psf.image)
    vmin = vmin_rel * vmax
    norm = SymLogNorm(vmin=vmin, vmax=vmax, linthresh=vmin, base=10)
    plot_args = dict(interpolation="none", origin="lower")

    im0 = ax[0].imshow(psf.image, norm=norm, **plot_args)
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(psf.get_model(), norm=norm, **plot_args)
    fig.colorbar(im1, ax=ax[1])
    
    vabs = np.max(np.abs(psf.get_residuals()))
    im2 = ax[2].imshow(psf.get_residuals(), cmap="RdBu", vmin=-vabs, vmax=vabs, **plot_args)
    cb = fig.colorbar(im2, ax=ax[2])
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()
    
    plt.show()

def plot_psf_matrix(psf):
    fig, ax = plt.subplots()

    coeff_matrix = psf.get_matrix()
    
    vabs = np.max(np.fabs(coeff_matrix))
    norm = SymLogNorm(vmin=-vabs, vmax=vabs, linthresh=vabs / 50, base=10)

    im = ax.matshow(coeff_matrix, cmap="RdBu", norm=norm)
    fig.colorbar(im, ax=ax)
    
    ax.set_xlabel(r"$n_x$")
    ax.set_ylabel(r"$n_y$")

    ax.xaxis.set_label_position("top") 
    
    plt.show()
