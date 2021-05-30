import nest
import nest.topology as topology
import numpy as np
from PIL import Image
from scipy import signal
from Gabor import get_gabor_adjusted
import os


# Takes an image and convert it to a 2D array.
# This method needs an 101x101 image to work properly.
def array_from_image(file_path):
    im = Image.open(file_path).convert('L')
    # im.show()
    (width, height) = im.size
    ia = np.array(list(im.getdata()))
    ia = ia.reshape((height, width))
    print(ia[0])
    return ia


# Set Poisson generators spiking rate using a receptive filed 3x3.
def image_array_to_retina(exposed_pattern, retina, center_on_or_off):
    # TODO Move to gabor filter orientation frequency. 2D gabor functional.

    _lambda = int(os.getenv("LAMBDA", 2))
    theta = int(os.getenv("THETA", 0))
    gamma = int(os.getenv("GAMMA", 3))
    sigma = int(os.getenv("SIGMA", 1))
    psi = int(os.getenv("PSI", 0))
    ksize = int(os.getenv("KSIZE", 5))

    gk = get_gabor_adjusted(ksize, _lambda, theta, gamma, sigma, psi, center_on_or_off)
    gk2 = np.transpose(gk)
    # TODO check exposed_patter and retina have same size (1 to 1 relation)
    height = len(exposed_pattern) - 2
    width = len(exposed_pattern[0]) - 2
    #kernel = center_on_kernel if center_on_or_off == 'on' else center_off_kernel
    convoluted = signal.convolve2d(exposed_pattern, gk, boundary='symm', mode='same')
    convoluted2 = signal.convolve2d(exposed_pattern, gk2, boundary='symm', mode='same')
    # TODO https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
    max_spiking_rate = int(os.getenv("MAX_SPIKING_RATE", 100))
    min_spiking_rate = int(os.getenv("MIN_SPIKING_RATE", 10))
    combined = convoluted + convoluted2
    combined = combined.clip(0, 1)
    rated = (combined * (max_spiking_rate - min_spiking_rate)) + min_spiking_rate
    for row in range(0, height):
        for column in range(0, width):
            ganglion_cells_id = topology.GetElement(retina, (row, column))
            rate = rated[row][column]
            # print(rate)
            nest.SetStatus(ganglion_cells_id, {'rate': rate * 1.0})
