from cv2 import getGaborKernel
from matplotlib import pyplot as plt
from scipy import signal
from skimage import io
from Patterns import get_pattern_0, get_pattern_1
import numpy as np
from Gabor import get_gabor_adjusted

image_array_0 = np.pad(np.kron(np.array(get_pattern_0()), np.ones((10, 10))), 1, mode='edge')
io.imshow(image_array_0)
io.show()

_lambda = 3
theta = 0
gamma = 2
sigma = 5
psi = 0
ksize = 10

gk = get_gabor_adjusted(ksize, _lambda, theta, gamma, sigma, psi, 'on')
print('Sum is: ' + str(np.sum(gk)))  # -0.01334812412534191

gk2 = np.transpose(gk)
plt.figure()
io.imshow(gk.real)
io.show()
# io.imshow(gk2.real)
# io.show()
#
# gk3 = (gk/2 + gk2/2).clip(0, 1).real
# print(np.max(gk3))
# io.imshow(gk3)
# io.show()
#
# io.imshow(-(gk3.clip(0, 1).real - 1))
# io.show()
convoluted = signal.convolve2d(image_array_0, gk, boundary='symm', mode='same')
io.imshow(convoluted)
io.show()
convoluted2 = signal.convolve2d(image_array_0, gk2, boundary='symm', mode='same')
io.imshow(convoluted2)
io.show()

result = convoluted + convoluted2

print(np.max(result))

io.imshow(result)
io.show()

clipped = result.clip(0, 1)
io.imshow(clipped)
io.show()
