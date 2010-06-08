#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps

mode=.03
#mean=.03
sigma=.06

scale = (-mode + (mode**2 +4*sigma**2)**.5)/2.
shape = (mode + scale)/scale
#scale = sigma**2/mean
#shape = mean/scale

s = np.random.gamma(shape, scale, 10000)
count, bins, ignored = plt.hist(s, 500, normed=True)
y = bins**(shape-1)*(np.exp(-bins/scale)/(sps.gamma(shape)*scale**shape))
plt.plot(bins, y, linewidth=2, color='r')
plt.show()

