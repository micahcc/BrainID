#!/usr/bin/python

import pylab as P
import sys
from paramtest import State, readout, transition
import random
from scipy.stats.distributions import norm
import scipy

PARAMS=[.98, .33, .34, .03,1.54, 2.46, .54]

def sim(particles, dt, count):
    for i in range(count):
        particles = [transition(p, PARAMS, dt, 0) for p in particles]
    return particles

def QQplot(observations):
    observations.sort()
    yaxis = list()
    for i in range(1, len(observations)-1):
        yaxis.append(norm.ppf(float(i)/len(observations)))
    P.plot(observations[1:-1], yaxis,'.')

    (m, b) = scipy.polyfit(observations[1:-1], yaxis, 1)
    print m, b
    xvals = [observations[0], observations[-1]]
    yvals = [observations[0]*m+b, observations[-1]*m+b]
    P.plot(xvals, yvals,'-*')

particles = [State() for i in range(1000)]
for var in particles:
    var.F = random.normalvariate(1, .3)
    var.S = random.normalvariate(1, .3)
    var.V = random.normalvariate(1, .3)
    var.Q = random.normalvariate(1, .3)
#P.hist([p.Q for p in particles])
out = sim(particles, .0001, 10000)
P.subplot(321)
P.title("f - normalized blood flow")
fs = [p.F for p in particles]
QQplot(fs)

P.subplot(322)
P.title("q - normalized deoxy/oxyhemoglobin ratio")
qs = [p.Q for p in particles]
QQplot(qs)

P.subplot(323)
P.title("v - normalized blood volume")
vs = [p.V for p in particles]
QQplot(vs)

P.subplot(324)
P.title("s - flow inducing signal")
ss = [p.S for p in particles]
QQplot(ss)

P.subplot(325)
P.title("y - BOLD signal")
ys = [readout(p, PARAMS) for p in particles]
QQplot(ys)

P.show()
