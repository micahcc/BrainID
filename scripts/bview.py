#!/usr/bin/python

from matplotlib.widgets import Slider, Button, RadioButtons
import pylab
import sys
import nibabel
import time
import numpy

global g_axslice
global g_sliceSlide
global g_timeSlide
global g_dir

def getequivindex(point, targetimg, img2):
#    trans = [targetimg.get_header()['srow_x'], targetimg.get_header()['srow_x'], \
#                targetimg.get_header()['srow_x']]
#    nibabel.orientations.apply_orientation(point, trans)
#    print targetimg, img2
    head1 = targetimg.get_header()
    forward = numpy.matrix([head1['srow_x'][0:3], head1['srow_y'][0:3], head1['srow_z'][0:3]])
    point = forward*(numpy.matrix(point).T)
    point = point + numpy.matrix([head1['qoffset_x'], head1['qoffset_y'], head1['qoffset_z']]).T
#    print point
    #going to other ...
    head2 = img2.get_header()
    reverse = numpy.matrix([head2['srow_x'][0:3], head2['srow_y'][0:3], head2['srow_z'][0:3]]).I
#    print reverse
    point = point - numpy.matrix([head2['qoffset_x'], head2['qoffset_y'], head2['qoffset_z']]).T
#    print point
    point = reverse*point
#    print point
    return [round(num) for num in point[:,0]]


def imshowslice(dir, slnum, time):
    if dir == 'x-y':
        pylab.subplot(111)
        pylab.imshow(images[0].get_data()[:,:,slnum,time])
    elif dir == 'y-z':
        pylab.subplot(111)
        pylab.imshow(images[0].get_data()[slnum,:,:,time])
    elif dir == 'x-z':
        pylab.subplot(111)
        pylab.imshow(images[0].get_data()[:,slnum,:,time])


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
g_axslice = pylab.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
g_sliceSlide = Slider(g_axslice, 'slice', 0 , images[0].get_shape()[2]-1, valinit=0, \
            valfmt='%3i')
axtime = pylab.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
g_timeSlide = Slider(axtime, 'Time', 0 , images[0].get_shape()[3]-1, valinit=0, \
            valfmt='%3i')

# Create callbacks
def slicecall(val):
    global g_dir
    global g_timeSlide
    imshowslice(g_dir, val, g_timeSlide.val)
    pylab.draw()

def timecall(val):
    global g_sliceSlide
    global g_dir
    imshowslice(g_dir, g_sliceSlide.val, val)
    pylab.draw()

# Attach callbacks
g_sliceSlide.on_changed(slicecall)
g_timeSlide.on_changed(timecall)

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

        #Plot each of the images in a single plot
        for i in range(0, len(images)):
            #get position
            pos = getequivindex([x,y,z], images[0], images[i])
            print pos
            if pos[0] < 0 or pos[0] > images[i].get_header()['dim'][1]:
                print "Skipping, point not in image: " + sys.argv[i+1]
                continue
            
            if pos[1] < 0 or pos[1] > images[i].get_header()['dim'][2]:
                print "Skipping, point not in image: " + sys.argv[i+1]
                continue
            
            if pos[2] < 0 or pos[2] > images[i].get_header()['dim'][3]:
                print "Skipping, point not in image: " + sys.argv[i+1]
                continue

            #get timesteps for x axis
            dt = images[i].get_header()['pixdim'][4]
            stop = (images[i].get_header()['dim'][4]+1)*dt
            xpoints = range(0, stop, dt)[0:images[i].get_header()['dim'][4]]

            #plot
            pylab.plot(xpoints, images[i].get_data()[pos[0],pos[1], pos[2],:])
            pylab.xlabel('Time (seconds)')

            leg.append(sys.argv[i+1] + str(pos))
            
        pylab.title("%u %u %u in original image" % (x, y ,z))
        pylab.legend(leg)
        pylab.show()

cid = fig3d.canvas.mpl_connect('button_press_event', mousecall)

#resetax = pylab.axes([0.8, 0.025, 0.1, 0.04])
#button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
#def reset(event):
#    sfreq.reset()
#    samp.reset()
#button.on_clicked(reset)

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
