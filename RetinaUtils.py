import nest
import nest.topology as topology
import numpy as np
from PIL import Image


# Takes an image and convert it to a 2D array.
# This method need and 101x101 image to work properly.
def array_from_image(file_path):
    im = Image.open(file_path).convert('L')
    #im.show()
    (width, height) = im.size
    ia = np.array(list(im.getdata()))
    ia = ia.reshape((height, width))
    print(ia[0])
    return ia


# Get neighborhood from an element, 2D array 3 x 3
def get_local_pattern(x, y, exposed_patter):
    local_pattern = np.zeros((3, 3))
    for row in range(x-1, x+2):
        for column in range(y-1, y+2):
            local_pattern[x - row][y - column] = exposed_patter[row + 1][column + 1]
    return local_pattern


# Vertical / Horizontal             Diagonal / Diagonal "wide"
# 000 000  100  111 111 111   0 3 6 9   100 110  110 0 1 3 4 =>
# 000 000  100  000 111 111             000 100  110
# 000 111  100  000 000 111             000 000  000
def calculate_surroundings_average(local_pattern):
    return np.sum(local_pattern)


# 000  111
# 010  101
# 000  111
def calculate_visual_spiking_rate_center_on(local_pattern):
    coverage = calculate_surroundings_average(local_pattern)
    frequencies = {
        1: 20, # This is wrong, it depends where is the 1.
        2: 20,
        3: 20,
        4: 40,
        5: 40,
        6: 60,
        7: 10,
        8: 10,
        9: 10
    }
    return frequencies.get(coverage, 5)


def calculate_visual_spiking_rate_center_off(local_pattern):
    coverage = calculate_surroundings_average(local_pattern)
    frequencies = {
        1: 100,
        2: 100,
        3: 100,
        4: 90,
        5: 80,
        6: 80,
        7: 10,
        8: 10,
        9: 10
    }
    return frequencies.get(coverage, 5)


def calculate_visual_spiking_rate(local_patter, center_on_or_off):
    return calculate_visual_spiking_rate_center_on(local_patter) \
        if center_on_or_off == 'on' \
        else calculate_visual_spiking_rate_center_off(local_patter)


# Set Poisson generators spiking rate using a receptive filed 3x3.
def image_array_to_retina(exposed_pattern, retina, center_on_or_off):
    # TODO check exposed_patter and retina have same size (1 to 1 relation)
    height = len(exposed_pattern) - 2
    width = len(exposed_pattern[0]) - 2
    for row in range(0, height):
        for column in range(0, width):
            ganglion_cells_id = topology.GetElement(retina, (row, column))
            local_pattern = get_local_pattern(row, column, exposed_pattern)
            rate = calculate_visual_spiking_rate(local_pattern, center_on_or_off)
            nest.SetStatus(ganglion_cells_id, {'rate': rate * 1.0})
