#!/usr/bin/python

import math
import scipy.signal
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pylab as P
from matplotlib.patches import Circle, Arrow, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

def normalize(arr1, arr2):
    total = 0
    for part in arr1:
        total = total + part[1]
    for part in arr2:
        total = total + part[1]
    return [(part[0], part[1]/total) for part in arr1], [(part[0], part[1]/total) for part in arr2]

def readspline(spl, pos):
    return scipy.signal.cspline1d_eval(spl, pos, dx = 1, x0=0)

def measure(spl, meas, arr1, arr2):
    arr1 = [(part[0], part[1]*math.exp(-.4*abs(meas - readspline(spl, [part[0]])))) for part in arr1]
    arr2 = [(part[0], part[1]*math.exp(-.4*abs(meas - readspline(spl, [part[0]])))) for part in arr2]
    return normalize(arr1, arr2)

def move(dist, left, right):
    left = [(part[0]-1, part[1]) for part in left]
    right= [(part[0]+1, part[1]) for part in right]
    return left, right

#initialize some constants
NUMPART = 20

velocity = 1
startpos = 10

elevmap = [range(NUMPART), [10, 20, 33, 30, 18, 25, 27, 32, 10, 12, \
                       19, 24, 20, 14, 18, 17, 10, 14, 30, 35]]

spline = scipy.signal.cspline1d(np.array(elevmap[1]))
tmpx = [x/1000. for x in range(0, NUMPART*1000)]
tmpy = readspline(spline, tmpx)


def update(left, right, pos, title, meas=False):
    print "Updating"
    print left,right

    fig = P.figure()
#    P.title(title)
    elevax = fig.add_subplot(111)
    elevax.set_ylim(0, 50)
    elevax.set_ylabel("Elevation")
    elevax.set_yticks([])

    elevax.set_xlim(-1, NUMPART+1)
    elevax.set_xlabel("Position")

    superindex = pos*1000;
    elevax.plot(tmpx,tmpy)
    if meas:
#        elevax.plot([-1, NUMPART+1], [tmpy[superindex]+2, tmpy[superindex]+2], 'g--')
        elevax.plot([-1, NUMPART+1], [tmpy[superindex],   tmpy[superindex]], 'r--')
#        elevax.plot([-1, NUMPART+1], [tmpy[superindex]-2, tmpy[superindex]-2], 'g--')
        for i in range(NUMPART):
            elevax.plot([i, i], [tmpy[superindex], tmpy[i*1000]], 'c:')
            
                
    elevax.plot([pos, pos], [0,35], 'm--')
    
    divider = make_axes_locatable(elevax)
    
    pax = divider.append_axes("bottom", 3, pad=0.2, sharex=elevax)

    pax.clear()
    pax.set_axis_off()
    pax.set_ylim(0, 2)
    pax.set_xlim(-1, NUMPART+1)
    elevax.set_xticks([])
    for pp in left:
        pax.vlines(pp[0], 0, pp[1], color='k')
#        pax.add_artist(Circle(xy = (pp[0], 1), radius = math.sqrt(3*pp[1]/3.14159), facecolor='red'))
    for pp in right:
        pax.vlines(pp[0], 1, 1+pp[1], color='k')
#        pax.add_artist(Circle(xy = (pp[0], 2), radius = math.sqrt(3*pp[1]/3.14159), facecolor='red'))
    
#    pRightax.clear()
#    pRightax.set_ylabel("Right")
#    pRightax.set_axis_off()
#    pRightax.set_ylim(-1, 1)
#    pRightax.set_xlim(-1, NUMPART+1)

#main
#initialize particles
posteriorL = zip(range(NUMPART), [1. for i in range(NUMPART)])
posteriorR = zip(range(NUMPART), [1. for i in range(NUMPART)])
posteriorL, posteriorR = normalize(posteriorL, posteriorR)
locat = 1

update(posteriorL, posteriorR, locat, "Initial Distribution")
P.show();

posteriorL, posteriorR = measure(spline, tmpy[locat*1000], posteriorL, posteriorR)
update(posteriorL, posteriorR, locat, "Measurement 1", meas=True)
P.show();

locat = locat+1
posteriorL, posteriorR = move(1, posteriorL, posteriorR)
update(posteriorL, posteriorR, locat, "Time Step 1")
P.show();

posteriorL, posteriorR = measure(spline, tmpy[locat*1000], posteriorL, posteriorR)
update(posteriorL, posteriorR, locat, "Measurement 2", meas=True)
P.show();

locat = locat+1
posteriorL, posteriorR = move(1, posteriorL, posteriorR)
update(posteriorL, posteriorR, locat, "Time Step 2")
P.show();

posteriorL, posteriorR = measure(spline, tmpy[locat*1000], posteriorL, posteriorR)
update(posteriorL, posteriorR, locat, "Measurement 3", meas=True)
P.show()

#fig.subplot
#
#plt.plot(elevmap[0], elevmap[1], 'k')
#
#
#
#ells = [Ellipse(xy=rand(2)*10, width=rand(), height=rand(), angle=rand()*360) \
#            for i in xrange(NUM)]
#
#
#
#
#for e in ells:
#    ax.add_artist(e)
#    e.set_clip_box(ax.bbox)
#    e.set_alpha(rand())
#    e.set_facecolor(rand(3))
#
#ax.set_xlim(0, 10)
#ax.set_ylim(0, 10)
#
#show()
#
#Path = mpath.Path
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#
#pathdata = [
#    (Path.MOVETO, (1.58, -2.57)),
#    (Path.CURVE4, (0.35, -1.1)),
#    (Path.CURVE4, (-1.75, 2.0)),
#    (Path.CURVE4, (0.375, 2.0)),
#    (Path.LINETO, (0.85, 1.15)),
#    (Path.CURVE4, (2.2, 3.2)),
#    (Path.CURVE4, (3, 0.05)),
#    (Path.CURVE4, (2.0, -0.5)),
#    (Path.CLOSEPOLY, (1.58, -2.57)),
#    ]
#
#codes, verts = zip(*pathdata)
#path = mpath.Path(verts, codes)
#patch = mpatches.PathPatch(path, facecolor='red', edgecolor='yellow', alpha=0.5)
#ax.add_patch(patch)
#
#x, y = zip(*path.vertices)
#line, = ax.plot(x, y, 'go-')
#ax.grid()
#ax.set_xlim(-3,4)
#ax.set_ylim(-3,4)
#ax.set_title('spline paths')
#plt.show()

