#!/usr/bin/python

import pylab as P
import sys
import nibabel 

#t = arange(0.0, 2.0, 0.01)
#s = sin(2*pi*t)
#plot(t, s, linewidth=1.0)
#
#xlabel('time (s)')
#ylabel('voltage (mV)')
#title('About as simple as it gets, folks')
#grid(True)
#show()

images = list();
for arg in sys.argv[1:]:
    img = nibabel.load(arg)
    images.append(img)

leg = []

for i in range(0, len(images)):
    print images[i]
    if len(images[i].get_shape()) < 3:
        P.plot(images[i].get_data()[:])
        leg = leg + [sys.argv[i+1]+":"+str(iter)]
    else: 
        for iter in range(0,images[i].get_shape()[3]):
            P.plot(images[i].get_data()[:, 0,0,iter])
            leg = leg + [sys.argv[i+1]+":"+str(iter)]
#P.legend(sys.argv[1:])
P.legend(leg)

P.show()
#
#names = ['Ts', 'Tf', 'epsilon', 'T0', 'alpha', 'E_0', 'V0', 'Vt', 'Qt', 'St', 'Ft']
#for i in range(0,11):
#    P.subplot(6, 2, i+1)
#    line1 = P.plot(truestat.data[:, 0, i, 0])
#    line2 = P.plot(eststat.data[:, 0, i, 0])
#    P.ylabel(names[i])
#
#P.show()
