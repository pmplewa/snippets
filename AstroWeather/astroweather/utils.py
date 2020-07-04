import numpy as np


def extent(a, axis=0):
    return np.min(a, axis=axis), np.max(a, axis=axis)

def minmax_scale(x, min=0, max=1, axis=0):
    x_min, x_max = extent(x, axis=axis)
    scale = (max - min) / (x_max - x_min)
    return scale * x + min - scale * x_min
