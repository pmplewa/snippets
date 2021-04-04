import autograd.numpy as np

from autograd import elementwise_grad

from .zernike import index, Znm


def phi(x, y, j):
    """
    Zhao & Burge (2007, 2008):
    Orthonormal vector polynomials in a unit circle, Part I and II
    """

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    n, m = index(j)

    if n == np.abs(m):
        val = Znm(rho, phi, n, m)
        return val / np.sqrt(2 * n * (n + 1))
    else:
        val = Znm(rho, phi, n, m) - np.sqrt((n + 1) / (n - 1)) * Znm(rho, phi, n - 2, m)
        return val / np.sqrt(4 * n * (n + 1))

def vec_poly_S(x, y, j):
    """S-Polynomials"""

    dphi_dx = elementwise_grad(lambda x, y: phi(x, y, j), 0)
    dphi_dy = elementwise_grad(lambda x, y: phi(x, y, j), 1)

    return dphi_dx(x, y), dphi_dy(x, y)

def vec_poly_P(x, y, j):
    """P-Polynomials"""

    dphi_dx = elementwise_grad(lambda x, y: phi(x, y, j), 0)
    dphi_dy = elementwise_grad(lambda x, y: phi(x, y, j), 1)

    return dphi_dy(x, y), -dphi_dx(x, y)
