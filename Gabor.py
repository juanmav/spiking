from cv2 import getGaborKernel
import numpy as np


def get_gabor_adjusted(k_size, _lambda, theta, gamma, sigma, psi, mode='on'):
    base = getGaborKernel((k_size, k_size), sigma, theta, _lambda, gamma, psi)
    gk = base if mode == 'on' else -base
    average = np.average(gk)
    return gk - average


def get_crossed_gabor_pattern(ksize):
    _lambda = 10
    theta = 0
    gamma = 0
    sigma = 20
    psi = 0
    gk = get_gabor_adjusted(ksize, _lambda, theta, gamma, sigma, psi, 'off')
    gk2 = np.transpose(gk)
    return (gk / 2 + gk2 / 2).clip(0, 1).real
