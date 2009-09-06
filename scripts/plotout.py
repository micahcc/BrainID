#!/usr/bin/python
import sys
import pylab as P
from nifti import *

#t = arange(0.0, 2.0, 0.01)
#s = sin(2*pi*t)
#plot(t, s, linewidth=1.0)
#
#xlabel('time (s)')
#ylabel('voltage (mV)')
#title('About as simple as it gets, folks')
#grid(True)
#show()

truebold = image.NiftiImage(sys.argv[1] . 'bold')
truestat = image.NiftiImage(sys.argv[1] . 'state')

estbold = image.NiftiImage(sys.argv[2] . 'bold')
eststat = image.NiftiImage(sys.argv[2] . 'state')
estcov = image.NiftiImage(sys.argv[2] . 'cov')

line1= P.plot(truebold.data[:, 0,0,0])
line2 = P.plot(estbold.data[:, 0,0,0])

P.show()

names = ['Ts', 'Tf', 'epsilon', 'T0', 'alpha', 'E_0', 'V0', 'Vt', 'Qt', 'St', 'Ft']
for i in range(0,11):
    P.subplot(6, 2, i+1)
    line1 = P.plot(truestat.data[:, 0, i, 0])
    line2 = P.plot(eststat.data[:, 0, i, 0])
    P.ylabel(names[i])

P.show()
