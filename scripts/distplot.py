#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps

def dist1(mode, sigma):
    scale = (-mode + (mode**2 +4*sigma**2)**.5)/2.
    shape = (mode + scale)/scale
    return (shape, scale)

def dist2(mean, sigma):
    scale = sigma**2/mean
    shape = mean/scale
    return (shape, scale)

mode=.34
mean=.64
sigma=.4*1.5

#shape, scale = dist1(mode,sigma)
shape, scale = dist2(mean,sigma)

s = np.random.gamma(shape, scale, 10000)
count, bins, ignored = plt.hist(s, 500, normed=True)
y = bins**(shape-1)*(np.exp(-bins/scale)/(sps.gamma(shape)*scale**shape))
plt.plot(bins, y, linewidth=2, color='r')
plt.show()

