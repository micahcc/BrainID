#!/usr/bin/python

from scipy.stats.distributions import norm, cauchy
import scipy
import numpy 
import pylab as P
from math import sqrt

def plus(a,b):
    return a+b

#time 1
P.subplot(311)
xvals = scipy.linspace(-10,10,num=1500)
measmean = 2
meas = [norm.pdf(i, measmean, .5) for i in xvals]
P.plot([-val for val in xvals], meas)

state = [cauchy.pdf(i, -2, 1)/2. +cauchy.pdf(i, 2, 1)/2. for i in xvals]
P.plot([-val for val in xvals], state)

statew = [var/sum(state) for var in state]
statemu = numpy.average(xvals, weights=statew)
statestd = sqrt(sum([pair1[1]*(pair1[0] - statemu)**2 for pair1 in zip(xvals, statew)]))
statest = [norm.pdf(i, statemu, statestd) for i in xvals]
P.plot([-val for val in xvals], statest)

combined_mean = (statemu + measmean)/2.
combined_var = sum([var[0]*(var[1] - combined_mean)**2 for var in zip(statest, xvals)])
combined_var = combined_var/sum(statest) + sum([var[0]*(var[1] - combined_mean)**2 for var in zip(meas, xvals)])/sum(meas)
print combined_var/2, sqrt(combined_var/2)
state = [norm.pdf(i, combined_mean, sqrt(combined_var/2)) for i in xvals]
P.plot([-val for val in xvals], state)
P.legend(["Measurement", "State Predict", "State Precit (Norm)", "Updated State"])

#time 2
P.subplot(312)
xvals = scipy.linspace(-10,10,num=1500)
measmean = 2
meas = [norm.pdf(i, measmean, .5) for i in xvals]
P.plot([10 - val for val in xvals], meas)

state2 = map(plus, state, [norm.pdf(i, 0, 20) for i in xvals])
state = state2
P.plot([10-val for val in xvals], state)

statew = [var/sum(state) for var in state]
statemu = numpy.average(xvals, weights=statew)
statestd = sqrt(sum([pair1[1]*(pair1[0] - statemu)**2 for pair1 in zip(xvals, statew)]))
statest = [norm.pdf(i, statemu, statestd) for i in xvals]
P.plot([10-val for val in xvals], statest)

combined_mean = (statemu + measmean)/2.
combined_var = sum([var[0]*(var[1] - combined_mean)**2 for var in zip(statest, xvals)])
combined_var = combined_var/sum(statest) + sum([var[0]*(var[1] - combined_mean)**2 for var in zip(meas, xvals)])/sum(meas)
print combined_var/2, sqrt(combined_var/2)
state = [norm.pdf(i, combined_mean, sqrt(combined_var/2)) for i in xvals]
P.plot([10-val for val in xvals], state)
P.legend(["Measurement", "State Predict", "State Precit (Norm)", "Updated State"])

#time 3
P.subplot(313)
xvals = scipy.linspace(-10,10,num=1500)
measmean = 2
meas = [norm.pdf(i, measmean, .5) for i in xvals]
P.plot([-3 - val for val in xvals], meas)

state2 = map(plus, state, [cauchy.pdf(i, 0, .1)/10 for i in xvals])
state = state2
P.plot([-3 - val for val in xvals], state)

statew = [var/sum(state) for var in state]
statemu = numpy.average(xvals, weights=statew)
statestd = sqrt(sum([pair1[1]*(pair1[0] - statemu)**2 for pair1 in zip(xvals, statew)]))
statest = [norm.pdf(i, statemu, statestd) for i in xvals]
P.plot([-3 - val for val in xvals], statest)

combined_mean = (statemu + measmean)/2.
combined_var = sum([var[0]*(var[1] - combined_mean)**2 for var in zip(statest, xvals)])
combined_var = combined_var/sum(statest) + sum([var[0]*(var[1] - combined_mean)**2 for var in zip(meas, xvals)])/sum(meas)
print combined_var/2, sqrt(combined_var/2)
state = [norm.pdf(i, combined_mean, sqrt(combined_var/2)) for i in xvals]
P.plot([-3 - val for val in xvals], state)
P.legend(["Measurement", "State Predict", "State Precit (Norm)", "Updated State"])

P.show()
