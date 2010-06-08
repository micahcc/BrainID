#!/usr/bin/python 
import sys
import pylab
from numpy import matrix, arange
from mpl_toolkits.mplot3d import Axes3D

print sys.argv[1]
fin = open(sys.argv[1], 'r')
ranges = [float(word) for word in fin.readline().split()]
density = [[float(word) for word in line.split()] for line in fin.readlines()]
print density
pylab.imshow(density, extent=ranges)
pylab.show()
