import autograd.numpy as np

from numpy import ma

from .zernike import eval_zernike


def otf_to_psf(otf):
    psf = np.abs(np.fft.fft2(otf, norm="ortho"))
    return np.fft.fftshift(psf)

def gen_psf(cj, λ=0.1, n_pix=500, **kwargs):
    """Generate a PSF corresponding to a certain WF abberation."""

    inv_λ = 1/λ

    x = np.linspace(-inv_λ, inv_λ, n_pix)
    x, y = np.meshgrid(x, x)
    mask = np.sqrt(x**2 + y**2) >= 1

    wf = eval_zernike(x, y, cj, **kwargs)
    wf = ma.array(wf, mask=mask, fill_value=0)
    
    return otf_to_psf(np.exp(1j * np.pi * wf).filled())
