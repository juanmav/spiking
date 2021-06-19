import nest
import nest.topology as topology
import numpy as np
from PIL import Image
from scipy import signal
from Gabor import get_gabor_adjusted
import os


def array_from_image(file_path):
    im = Image.open(file_path).convert('L')
    # im.show()
    (width, height) = im.size
    ia = np.array(list(im.getdata()))
    ia = ia.reshape((height, width))
    print(ia[0])
    return ia


# Set Poisson generators spiking rate using a receptive filed 3x3.
def image_array_to_retina(exposed_pattern, retina, center_on_or_off, max_spiking_rate, min_spiking_rate):
    _lambda = int(os.getenv("LAMBDA", 2))
    theta = int(os.getenv("THETA", 0))
    gamma = int(os.getenv("GAMMA", 3))
    sigma = int(os.getenv("SIGMA", 1))
    psi = int(os.getenv("PSI", 0))
    ksize = int(os.getenv("KSIZE", 5))

    gk = get_gabor_adjusted(ksize, _lambda, theta, gamma, sigma, psi, center_on_or_off)
    gk2 = np.transpose(gk)
    convoluted = signal.convolve2d(exposed_pattern, gk, boundary='symm', mode='same')
    convoluted2 = signal.convolve2d(exposed_pattern, gk2, boundary='symm', mode='same')
    combined = convoluted + convoluted2
    combined = combined.clip(0, 1)
    rated = (combined * (max_spiking_rate - min_spiking_rate)) + min_spiking_rate
    nodes = nest.GetNodes(retina)[0]
    listing = rated.flatten().tolist()
    nest.SetStatus(nodes, 'rate',  listing)


# Set Poisson generators spiking rate using a receptive filed 3x3.
def direct_image_array_to_retina(exposed_pattern, retina, center_on_or_off, max_spiking_rate, min_spiking_rate):
    rated = (exposed_pattern * (max_spiking_rate - min_spiking_rate)) + min_spiking_rate
    nodes = nest.GetNodes(retina)[0]
    print(nodes)
    listing = rated.flatten().tolist()
    nest.SetStatus(nodes, 'rate',  listing)

