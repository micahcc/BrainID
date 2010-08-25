#!/usr/bin/python

import pylab as P
import sys
from paramtest import State, readout, transition
import random
from scipy.stats.distributions import norm
import scipy

END = 20.
dt = 1./2000
print dt

PARAMS=[.98, .33, .34, .03, 1.54, 2.46, .54]
def sim(particles, dt, count):
    for i in range(count):
        particles = [transition(p, PARAMS, dt, 0) for p in particles]
    return particles

def plotem(times, uts, sts, fts, qts, vts):
    P.subplot(2, 3, 1)
    P.plot(times, uts)
    P.axis([0, END, 0, 2])
    P.xlabel("Seconds")
    P.title("Stimulus")
    
    P.subplot(2, 3, 2)
    P.plot(times, sts)
    P.xlabel("Seconds")
    P.title("s")
    
    P.subplot(2, 3, 3)
    P.plot(times, fts)
    P.xlabel("Seconds")
    P.title("f")
    
    P.subplot(2, 3, 4)
    P.plot(times, qts)
    P.xlabel("Seconds")
    P.title("q")
    
    P.subplot(2, 3, 5)
    P.plot(times, vts)
    P.xlabel("Seconds")
    P.title("v")
    
    P.subplot(2, 3, 6)
    P.plot(times, bts)
    P.xlabel("Seconds")
    P.title("BOLD")
    P.show()

local = State() 
local.F = 1;
local.S = 0;
local.V = 1;
local.Q = 1;

times = [i*dt for i in range(END/dt)]
vts = [0 for i in range(END/dt)]
sts = [0 for i in range(END/dt)]
fts = [0 for i in range(END/dt)]
qts = [0 for i in range(END/dt)]
bts = [0 for i in range(END/dt)]


uts = [0 for i in range(END/dt)]
for i in range(END/dt):
    if i*dt > 1 and i*dt < 3:
        uts[i] = 1;
#    elif i*dt > 2 and i*dt < 2.25:
#        uts[i] = 1;
    else:
        uts[i] = 0;

for i in range(END/dt):
    vts[i] = local.V;
    qts[i] = local.Q;
    sts[i] = local.S;
    fts[i] = local.F;
    transition(local, PARAMS, dt, uts[i])
    bts[i] = readout(local, PARAMS)

plotem(times, uts, sts, fts, qts, vts)

