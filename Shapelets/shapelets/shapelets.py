import numpy as np

from astropy.io import fits
from numpy import ma
from scipy.optimize import minimize
from scipy.special import binom, gamma
from scipy.special import eval_hermite as hermite


def is_odd(x):
    return x & 1

def is_even(x):
    return not is_odd(x)

def factorial(n):
    return gamma(n + 1)

class ShapeletPSF():
    def __init__(self, nmax, scale, coeff=None):
        """
        Refregier 2003: Shapelets - I. A method for image analysis
        """

        self._nmax = nmax
        self._scale = scale
        
        # pre-compute parameter indices
        self._index = np.zeros((nmax, nmax), dtype=np.int)
        for nx in range(nmax):
            for ny in range(nmax):
                if nx + ny < nmax:
                    self._index[nx, ny] = nmax -ny - (nx * (1 + nx - 2 * nmax)) // 2 - 1
        
        # pre-compute coefficients for shapelet computation
        self._coeff_phi1d = np.zeros(nmax)
        for n in range(nmax):
            self._coeff_phi1d[n] = (2**n * np.pi**0.5 * factorial(n))**-0.5  
        
        # pre-compute coefficients for flux computation
        self._coeff_flux = np.zeros(self.ncoeff)
        for nx in range(0, nmax, 2):
            for ny in range(0, nmax, 2):
                if nx + ny < nmax:
                    self._coeff_flux[self._index[nx, ny]] = (
                        np.pi**0.5 * 2**(0.5 * (2 - nx - ny))
                      * binom(nx, nx / 2)**0.5
                      * binom(ny, ny / 2)**0.5)
                    
        # pre_compute coefficients for centroid computation
        self._coeff_cen = np.zeros(self.ncoeff)
        for nx in range(nmax):
            for ny in range(nmax):
                if nx + ny < nmax:
                    if is_odd(nx) and is_even(ny):
                        self._coeff_cen[self._index[nx, ny]] = (
                            np.pi**0.5 * (nx + 1)**0.5 * 2**(0.5 * (2 - nx - ny))
                          * binom(nx + 1, (nx + 1)/2)**0.5
                          * binom(ny, ny/2)**0.5)
                    elif is_even(nx) and is_odd(ny):
                        self._coeff_cen[self._index[nx, ny]] = (
                            np.pi**0.5 * (ny + 1)**0.5 * 2**(0.5 * (2 - nx - ny))
                          * binom(ny + 1, (ny + 1) / 2)**0.5
                          * binom(nx, nx / 2)**0.5)

        self.coeff = coeff

    def __repr__(self):
        return f"<ShapeletPSF nmax={self.nmax} scale={self.scale}>"

    @property
    def nmax(self):
        return self._nmax

    @property
    def ncoeff(self):
        return self.nmax + (self.nmax**2 - self.nmax) // 2

    @property
    def scale(self):
        return self._scale

    @property
    def coeff(self):
        return self._coeff

    @coeff.setter
    def coeff(self, value):
        if value is None:
            self._coeff = np.zeros(self.ncoeff)
            self._coeff[self._index[0, 0]] = 1
        else:
            assert len(value) == self.ncoeff
            self._coeff = value
        
    def phi1d(self, x, n):
        """1D shapelet basis function (dimensionless)"""
        return self._coeff_phi1d[n] * hermite(n, x) * np.exp(-x**2 / 2)
    
    #def B1d(self, x, n):
    #    """1D shapelet basis function"""
    #    return self.scale**-0.5 * phi1d(s**-1 * x, n)    
        
    def phi2d(self, x, y, nx, ny):
        """2D shapelet basis function (dimensionless)"""
        return self.phi1d(x, nx) * self.phi1d(y, ny)
    
    def B2d(self, x, y, nx, ny):
        """2D shapelet basis function"""
        return self.scale**-1 * self.phi2d(self.scale**-1 * x, self.scale**-1 * y, nx, ny)      

    def get_value(self, x, y):
        value = 0
        for nx in range(self.nmax):
            for ny in range(self.nmax):
                if nx + ny < self.nmax:
                    value += (self.coeff[self._index[nx, ny]]
                            * self.B2d(x, y, nx, ny))
        return value

    def get_flux(self):
        flux = 0
        for nx in range(0, self.nmax, 2):
            for ny in range(0, self.nmax, 2):
                if nx + ny < self.nmax:
                    flux += (self._coeff_flux[self._index[nx, ny]]
                           * self.coeff[self._index[nx, ny]])
        flux *= self.scale
        return flux
    
    def get_centroid(self):
        pos = np.zeros(2)
        for nx in range(self.nmax):
            for ny in range(self.nmax):
                if nx + ny < self.nmax:
                    if is_odd(nx) and is_even(ny):
                        pos[0] += (self._coeff_cen[self._index[nx, ny]]
                                 * self.coeff[self._index[nx, ny]])
                    elif is_even(nx) and is_odd(ny):
                        pos[1] += (self._coeff_cen[self._index[nx, ny]]
                                 * self.coeff[self._index[nx, ny]])
        pos *= self.scale**2 / self.get_flux()
        return pos
    
    def normalize(self):
        flux = self.get_flux()
        assert flux > 0
        self.coeff /= flux
        
    def get_matrix(self):
        matrix = np.zeros((self.nmax, self.nmax))
        for nx in range(self.nmax):
            for ny in range(self.nmax):
                if nx + ny < self.nmax:
                    matrix[nx, ny] = self.coeff[self._index[nx, ny]]
        return matrix

class FittedShapeletPSF(ShapeletPSF):
    def __init__(self, image, nmax, scale, coeff0=None, bounds=[None, None], sigma=None):
        ShapeletPSF.__init__(self, nmax=nmax, scale=scale, coeff=coeff0)
        
        assert ma.isMaskedArray(image)
        self._image = image
        
        if sigma is None:
            self._sigma = np.ones_like(image) # unweighted fit
        else:
            assert np.array_equal(image.shape, sigma.shape)
            self._sigma = sigma
        
        nx, ny = image.shape
        self._x, self._y = np.indices((nx, ny), dtype=np.float)
        self._x -= (nx - 1)/2
        self._y -= (nx - 1)/2
        
        self._bounds = [bounds] * self.ncoeff

        self.fit()

    def __repr__(self):
        return f"<FittedShapeletPSF shape={self.image.shape} nmax={self.nmax} scale={self.scale}>"

    @property
    def image(self):
        return self._image

    @property
    def sigma(self):
        return self._sigma

    @property
    def mask(self):
        return self.image.mask

    @property
    def fill_value(self):
        return self.image.fill_value

    @property
    def npix(self):
        return np.sum(~self.mask)
        
    def get_model(self):
        image = self.get_value(self._x, self._y)
        return ma.array(image, mask=self.mask, fill_value=self.fill_value)
    
    def get_residuals(self):
        return self.image - self.get_model()
    
    def get_chi2(self):
        return np.sum(self.get_residuals()**2 / self.sigma**2)

    def get_chi2red(self):
        dof = self.npix - self.ncoeff
        return self.get_chi2() / dof
    
    def _objective(self, theta):
        self.coeff = theta
        return self.get_chi2() # logL = -0.5 * chi2
        
    def fit(self):
        fit_args = dict(method="L-BFGS-B", bounds=self._bounds, options={"disp": True})
        fit_results = minimize(self._objective, self.coeff, **fit_args)
        assert fit_results.success, fit_results

def save_model(path, psf):
    assert isinstance(psf, FittedShapeletPSF)

    header = fits.Header()
    header["NMAX"] = psf.nmax
    header["SCALE"] = psf.scale
    header["NCOEFF"] = psf.ncoeff
    for i, c in enumerate(psf.coeff):
        header[f"COEFF{i}"] = c

    hdu_list = fits.HDUList([
        fits.PrimaryHDU(data=psf.get_model().filled(), header=header),
        fits.ImageHDU(data=psf.sigma.filled())])

    hdu_list.writeto(path, overwrite=True)

def load_model(path):
    model_hdu, sigma_hdu = fits.open(path)
    header = model_hdu.header

    nmax = header["NMAX"]
    scale = header["SCALE"]
    ncoeff = header["NCOEFF"]
    coeff = np.zeros(ncoeff)
    for i in range(ncoeff):
        coeff[i] = header[f"COEFF{i}"]

    return ShapeletPSF(nmax=nmax, scale=scale, coeff=coeff)
