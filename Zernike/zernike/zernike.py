import autograd.numpy as np

from scipy.special import binom


def is_odd(x):
    return x & 1

#def is_even(x):
#    return not is_odd(x)

def index(j):
    """
    OSA/ANSI indexing:
    j = (n * (n + 2) + m)/2
    """

    assert j >= 0, "j >= 0"

    n = 0
    while j > n:
        n += 1
        j -= n

    m = 2*j - n

    return n, m

def R(rho, n, abs_m):
    """Radial Zernike Polynomials"""

    val = np.zeros_like(rho)

    if is_odd(n - abs_m):
        return val

    for k in range((n - abs_m) // 2 + 1):
        val += ((-1)**k
            * binom(n - k, k)
            * binom(n - 2*k, (n - abs_m) // 2 - k)
            * rho**(n - 2*k))

    return val

def Znm(rho, phi, n, m, norm=True):
    """Zernike Polynomials (n/m indexing)"""

    abs_m = np.abs(m)

    if norm:
        N = np.sqrt(n + 1) if m == 0 else np.sqrt(2*(n + 1))
    else:
        N = 1

    if m < 0:
        return N * R(rho, n, abs_m) * np.sin(abs_m * phi)
    else:
        return N * R(rho, n, abs_m) * np.cos(abs_m * phi)

def Zj(rho, phi, j, **kwargs):
    """Zernike Polynomials (j indexing)"""

    n, m = index(j)

    return Znm(rho, phi, n, m, **kwargs)

def eval_zernike(x, y, cj, **kwargs):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    val = 0

    for j, c in enumerate(cj):
        val += c * Zj(rho, phi, j, **kwargs)

    return val
