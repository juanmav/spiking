from cv2 import getGaborKernel
import numpy as np


def get_gabor_adjusted(k_size, _lambda, theta, gamma, sigma, psi, mode='on'):
    gk = getGaborKernel((k_size, k_size), sigma, theta, _lambda, gamma, psi)
    factor = 1 / (1 + np.sum(gk))
    scale = (lambda x: x if x < 0 else x * factor)
    gk = np.vectorize(scale)(gk)
    return gk if mode == 'on' else -gk
