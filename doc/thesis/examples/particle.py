#!/usr/bin/python

import sys
from scipy.stats.distributions import norm, cauchy
import scipy
import numpy 
import pylab as P
from math import sqrt
import random

def plus(a,b):
    return a+b
def multiply(a,b):
    return a*b

def normalize(arr):
    tot = sum(arr)*20/1500
    return [val/tot for val in arr]

def sample(pos, weights):
    sm = sum(weights)
    weights = [w/sm for w in weights]
    var = random.uniform(0,1)
    agg = sum(weights)
    i = 0
    while agg > var:
        agg = agg - weights[i]
        i = i+1
    return pos[i]

def smooth(pos, weights):
    tog = sorted(zip(pos, weights), key=lambda val: val[0])
    xvals = scipy.linspace(-10,10,num=1500)
    pdf = [0 for val in xvals]
    j = 0
    for i in range(0, len(pdf)):
        while j < len(tog) and xvals[i] > tog[j][0]:
            pdf[i] = pdf[i] + tog[j][1]
            j = j + 1
    gauss = [norm.pdf(i/100., 2, .2) for i in range(0,400)]
    return (xvals, numpy.convolve(pdf, gauss, mode="same"))

#time 1
P.figure(1)
P.subplot(411)
P.title("Update 1")
xvals = scipy.linspace(-10,10,num=1500)
measmean = 1
meas = normalize([norm.pdf(i, measmean, 2) for i in xvals])
P.plot([val for val in xvals], meas)

state = normalize([cauchy.pdf(i, -2, 1) +cauchy.pdf(i, 2, 1) for i in xvals])
P.plot([val for val in xvals], state)

state = normalize(map(multiply, state, meas))
P.plot([val for val in xvals], state)
P.hist([val for val in xvals], weights=[-1./3000 for val in xvals], color="b")
P.legend(["Measurement", "State Predict", "Updated State", "Support"], loc=2)

#time 2
P.subplot(412)
P.title("Update 2")
#meas = normalize([norm.pdf(i, measmean, 2) for i in xvals])
P.plot([val for val in xvals], meas)

#state = normalize([cauchy.pdf(i, -2, 1) +cauchy.pdf(i, 2, 1) for i in xvals])
P.plot([val for val in xvals], state)

state = normalize(map(multiply, state, meas))
P.plot([val for val in xvals], state)
P.hist([val for val in xvals], weights=[-1./3000 for val in xvals], color="b")
P.legend(["Measurement", "State Predict", "Updated State", "Support"], loc=2)

#time 3
P.subplot(413)
P.title("Resample Support")
P.plot([val for val in xvals], state)
part_points = [sample(xvals, state) for val in xvals]
state = [1./len(xvals)  for val in xvals]

P.hist(part_points, bins=100, weights=[-1./3000 for val in xvals], normed=True)
P.legend(["Posterior Distribution", "New Support"], loc=2)

#time 4
P.subplot(414)
P.title("Update 3")
tmpx, tmpy = smooth(part_points, state)
tmpy = normalize(tmpy)
meas = normalize([norm.pdf(i, measmean-2, 2) for i in xvals])
P.plot([val for val in xvals], meas)

P.plot([val for val in tmpx], tmpy)

reweight = [norm.pdf(point, measmean-2, 2) for point in part_points]
state = map(multiply, state, reweight)
tmpx, tmpy = smooth(part_points, state)
tmpy = normalize(tmpy)
P.plot([val for val in tmpx], tmpy)

P.hist([val for val in part_points], weights=[-10./len(part_points) for val in part_points], bins=100, color="b")

P.legend(["Measurement", "State Predict", "Updated State", "Support"], loc=2)

P.show()

