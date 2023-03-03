import random
import numpy as np


def add_noise_for_2d_pts(pts: np.ndarray, scale=1.0):
    return pts + np.random.normal(scale=scale, size=pts.shape)

