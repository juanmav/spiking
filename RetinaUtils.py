import nest
import nest.topology as topology
import numpy as np
from PIL import Image
from scipy import signal
from skimage.filters import gabor_kernel


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
    gk = gabor_kernel(frequency=1, theta=0)
    # TODO check exposed_patter and retina have same size (1 to 1 relation)
    height = len(exposed_pattern) - 2
    width = len(exposed_pattern[0]) - 2
    kernel = np.real(gk if center_on_or_off == 'on' else -gk)
    convoluted = signal.convolve2d(exposed_pattern, kernel, boundary='symm', mode='same').clip(min=0)
    # TODO https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
    convoluted = (convoluted * 90) + 10
    for row in range(0, height):
        for column in range(0, width):
            ganglion_cells_id = topology.GetElement(retina, (row, column))
            rate = convoluted[row][column] * 20
            nest.SetStatus(ganglion_cells_id, {'rate': rate * 1.0})
