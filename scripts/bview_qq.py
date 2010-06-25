#!/usr/bin/python

import scipy 
from scipy.stats.distributions import norm
from matplotlib.widgets import Slider, Button, RadioButtons
import pylab
import sys
import nibabel
import time
import numpy
import random

class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)

global g_axslice
global g_sliceSlide
global g_timeSlide
global g_dir

def QQplot(observations):
    observations.sort()
    yaxis = list()
    for i in range(1, len(observations)-1):
        yaxis.append(norm.ppf(float(i)/len(observations)))
    pylab.plot(observations[1:-1], yaxis,'.')

    (m, b) = scipy.polyfit(observations[1:-1], yaxis, 1)
    print m, b
    xvals = [observations[0], observations[-1]]
    yvals = [observations[0]*m+b, observations[-1]*m+b]
    pylab.plot(xvals, yvals,'-*')
    pylab.show()

def getequivindex(point, targetimg, img2):
    head1 = targetimg.get_header()
    forward = numpy.matrix([head1['srow_x'][0:3], head1['srow_y'][0:3], head1['srow_z'][0:3]])
    point = forward*(numpy.matrix(point).T)
    point = point + numpy.matrix([head1['qoffset_x'], head1['qoffset_y'], head1['qoffset_z']]).T
    #going to other ...
    head2 = img2.get_header()
    reverse = numpy.matrix([head2['srow_x'][0:3], head2['srow_y'][0:3], head2['srow_z'][0:3]]).I
    point = point - numpy.matrix([head2['qoffset_x'], head2['qoffset_y'], head2['qoffset_z']]).T
    point = reverse*point
    return [round(num) for num in point[:,0]]


def imshowslice(dir, slnum, time):
    try:
        if dir == 'x-y':
             data = [images[0].get_data()[:,i,slnum,time] for i in range(images[0].get_shape()[1])]
        elif dir == 'y-z':
            data = [images[0].get_data()[slnum,:,i,time] for i in range(images[0].get_shape()[2])]
        elif dir == 'x-z':
            data = [images[0].get_data()[:,slnum,i,time] for i in range(images[0].get_shape()[2])]
    except:
        if dir == 'x-y':
             data = [images[0].get_data()[:,i,slnum] for i in range(images[0].get_shape()[1])]
        elif dir == 'y-z':
            data = [images[0].get_data()[slnum,:,i] for i in range(images[0].get_shape()[2])]
        elif dir == 'x-z':
            data = [images[0].get_data()[:,slnum,i] for i in range(images[0].get_shape()[2])]
    pylab.subplot(111)
    pylab.imshow(data,cmap='gray', origin='lower', interpolation='nearest')


#Input
images = list()
for arg in sys.argv[1:]:
    print arg
    images.append(nibabel.load(arg))

#Initialize
g_dir = 'x-y'
fig3d = pylab.figure()
ax = fig3d.add_subplot(111)
pylab.subplots_adjust(left=0.25, bottom=0.25)
imshowslice(g_dir, 0, 0)

# - - - - - - - - - -
# Initialize sliders
# - - - - - - - - - - 
axcolor = 'lightgoldenrodyellow'

#time select
if images[0].get_header()['dim'][0] > 3:
    global g_timeSlide
    def timecall(val):
        global g_sliceSlide
        global g_dir
        imshowslice(g_dir, g_sliceSlide.val, val)
        pylab.draw()
    
    axtime = pylab.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
    g_timeSlide = Slider(axtime, 'Time', 0 , images[0].get_shape()[3]-1, valinit=0, \
                valfmt='%3i')

    g_timeSlide.on_changed(timecall)
else:
    global g_timeSlide
    g_timeSlide = Struct(val = 0)

#slice select
g_axslice = pylab.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
g_sliceSlide = Slider(g_axslice, 'slice', 0 , images[0].get_shape()[2]-1, valinit=0, \
            valfmt='%3i')

# Create callbacks
def slicecall(val):
    global g_dir
    global g_timeSlide
    imshowslice(g_dir, val, g_timeSlide.val)
    pylab.draw()
g_sliceSlide.on_changed(slicecall)

# ------------------------
# Initialize Mouse Buttons
# ------------------------
def mousecall(event):
    global g_dir
    global g_sliceSlide
    if event.inaxes == ax and event.button == 1:
        figplot = pylab.figure()
        print g_dir
        if g_dir == 'y-z':
            x = int(g_sliceSlide.val)
            y = int(event.ydata)
            z = int(event.xdata)
        elif g_dir == 'x-z':
            x = int(event.ydata)
            y = int(g_sliceSlide.val)
            z = int(event.xdata)
        else: #dir == 'x-y':
            x = int(event.xdata)
            y = int(event.ydata)
            z = int(g_sliceSlide.val)

        print x,y,z
        leg = list()

        #get timesteps for x axis
        dt = images[0].get_header()['pixdim'][4]
        stop = (images[0].get_header()['dim'][4]+1)*dt
        mu = numpy.mean(images[0].get_data()[x,y,z,5:])
        samples = [(sample-mu)/mu for sample in images[0].get_data()[x,y,z,5:]]
        xpoints = range(0, stop, dt)[0:images[0].get_header()['dim'][4]-5]
#        samples = numpy.random.randn(100)
#        xpoints = range(0, len(samples))

        #plot
        pylab.subplot(221)
        pylab.plot(xpoints, samples, '*-')
        pylab.xlabel('Time (seconds)')

        print samples
        pylab.subplot(223)
        pylab.acorr(samples, usevlines=True, normed=True, maxlags=None, lw=2)
        pylab.grid(True)
        pylab.axhline(0, color='black', lw=2)

        pylab.subplot(222)
        QQplot(samples)
        
        pylab.title("%u %u %u" % (x, y ,z))
        pylab.show()

cid = fig3d.canvas.mpl_connect('button_press_event', mousecall)

#!======================
#! Interactive - Direction
#!======================
rax = pylab.axes([0.025, 0.3, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('x-y','y-z','x-z'), active=0)
def axesfunc(newaxes):
    global g_sliceSlide
    global g_dir
    global g_axslice
    pylab.delaxes(g_axslice)
    
    g_axslice = pylab.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    if newaxes == 'x-y':
        g_sliceSlide = Slider(g_axslice, 'slice', 0 , images[0].get_shape()[2]-1, valinit=0, \
                    valfmt='%3i')
    elif newaxes == 'y-z':
        g_sliceSlide = Slider(g_axslice, 'slice', 0 , images[0].get_shape()[0]-1, valinit=0, \
                    valfmt='%3i')
    elif newaxes == 'x-z':
        g_sliceSlide = Slider(g_axslice, 'slice', 0 , images[0].get_shape()[1]-1, valinit=0, \
                    valfmt='%3i')
    g_sliceSlide.on_changed(slicecall)
    
    pylab.subplot(111)
    g_dir = newaxes
    slicecall(0)
radio.on_clicked(axesfunc)

pylab.show()
