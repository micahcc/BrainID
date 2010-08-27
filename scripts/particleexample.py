#!/usr/bin/python

import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pylab as P
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

POSITIONS = 20

velocity = 1
startpos = 10

posteriorL = zip(range(POSITIONS), [1./POSITIONS for i in range(POSITIONS)])
posteriorR = zip(range(POSITIONS), [1./POSITIONS for i in range(POSITIONS)])

elevmap = [range(POSITIONS), [10, 20, 33, 30, 18, 25, 27, 32, 10, 12, \
                       19, 24, 20, 14, 18, 17, 10, 14, 30, 35]]

fig = P.figure()
elevax = fig.add_subplot(111)
elevax.set_xlim(-1, POSITIONS+1)
elevax.set_ylim(0, 50)
elevax.plot(elevmap[0], elevmap[1])

divider = make_axes_locatable(elevax)

pLeftax = divider.append_axes("bottom", .5, pad=0.2, sharex=elevax)
pRightax = divider.append_axes("bottom", .5, pad=0, sharex=elevax)

def update(left, right):
    pLeftax.clear()
    pLeftax.set_ylabel("Left")
    pLeftax.set_axis_off()
    pLeftax.set_ylim(-1, 1)
    pLeftax.set_xlim(-1, POSITIONS+1)
    for pp in left:
        pLeftax.add_artist(Circle(xy = (pp[0], 0), radius = pp[1]))
    
    pRightax.clear()
    pRightax.set_ylabel("Right")
    pRightax.set_axis_off()
    pRightax.set_ylim(-1, 1)
    pRightax.set_xlim(-1, POSITIONS+1)
    for pp in right:
        pRightax.add_artist(Circle(xy = (pp[0], 0), radius = pp[1]))

update(posteriorL, posteriorR)
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

